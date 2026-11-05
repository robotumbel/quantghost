"""Payload embedding into float32 model weights.

X-LSB embedding overwrites the X least-significant bits of each float32 weight's
IEEE-754 bit pattern. For X <= 23 these are mantissa bits, so the numerical
perturbation is tiny and model accuracy is preserved. This is the EvilModel /
Model X-Ray "X-LSB-Attack" baseline.

All functions operate on flat float32 numpy arrays and are pure / side-effect
free (they return a new array).
"""
from __future__ import annotations
import numpy as np

MANTISSA_BITS = 23  # IEEE-754 single precision


def as_uint32(w: np.ndarray) -> np.ndarray:
    """View a float32 array as uint32 (same bits)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    return w.view(np.uint32)


def as_float32(u: np.ndarray) -> np.ndarray:
    """View a uint32 array back as float32."""
    return np.ascontiguousarray(u, dtype=np.uint32).view(np.float32)


def random_payload_bits(n_bits: int, seed: int = 0) -> np.ndarray:
    """A reproducible uniform-random payload as a {0,1} uint8 array."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


def bytes_to_bits(payload: bytes) -> np.ndarray:
    """Unpack a byte payload (e.g. a malware binary) to an MSB-first bit array."""
    arr = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(arr)  # MSB-first, length 8*len


def embed_xlsb(w: np.ndarray, payload_bits: np.ndarray, x: int,
               fill: bool = True) -> np.ndarray:
    """Embed `payload_bits` into the X LSBs of each weight (X-LSB-Fill).

    Parameters
    ----------
    w : float32 array (flattened weights)
    payload_bits : {0,1} array; cycled if shorter than capacity when fill=True
    x : number of least-significant bits used per weight (1..23 sensible)
    fill : if True, every weight carries X payload bits (max capacity, the
           detectable regime). If False, only the first ceil(len/x) weights are
           touched (partial payload).

    Returns a new float32 array with identical exponent/sign bits.
    """
    if not (1 <= x <= 32):
        raise ValueError("x must be in 1..32")
    u = as_uint32(w).copy()
    n = u.shape[0]
    capacity = n * x
    if fill:
        # tile payload to fill the whole capacity
        reps = int(np.ceil(capacity / max(len(payload_bits), 1)))
        bits = np.tile(payload_bits, reps)[:capacity].astype(np.uint32)
    else:
        bits = payload_bits[:capacity].astype(np.uint32)
        if bits.shape[0] < capacity:
            bits = np.concatenate([bits, np.zeros(capacity - bits.shape[0],
                                                  dtype=np.uint32)])
    # pack x bits into an integer per weight (MSB-first within the x-bit field)
    bits = bits.reshape(n, x)
    weights_field = np.zeros(n, dtype=np.uint32)
    for i in range(x):
        weights_field = (weights_field << 1) | bits[:, i]
    mask = np.uint32((1 << x) - 1)
    u = (u & ~mask) | (weights_field & mask)
    return as_float32(u)


def extract_xlsb(w: np.ndarray, x: int) -> np.ndarray:
    """Recover the embedded X-bit fields (inverse of embed; for round-trip tests)."""
    u = as_uint32(w)
    mask = np.uint32((1 << x) - 1)
    field = u & mask
    bits = np.zeros((u.shape[0], x), dtype=np.uint8)
    for i in range(x):
        bits[:, x - 1 - i] = (field >> i) & 1
    return bits.reshape(-1)


def perturbation_stats(w_clean: np.ndarray, w_stego: np.ndarray) -> dict:
    """Quantify how little the weights moved (sanity that accuracy is preserved)."""
    d = np.abs(w_clean.astype(np.float64) - w_stego.astype(np.float64))
    denom = np.abs(w_clean.astype(np.float64)) + 1e-12
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float((d / denom).max()),
        "mean_rel": float((d / denom).mean()),
        "frac_changed": float((d > 0).mean()),
    }
