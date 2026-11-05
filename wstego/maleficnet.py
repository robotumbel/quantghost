"""Faithful MaleficNet-style baseline: direct-sequence spread-spectrum (DSSS / CDMA)
payload embedding in neural-network weights (Hitaj et al., MaleficNet, ESORICS 2022).

Unlike naive X-LSB (EvilModel/StegoNet), MaleficNet does NOT edit mantissa bits. It
spreads each payload bit over the whole carrier with a pseudo-random Gaussian sequence
and ADDS a small-amplitude signal to the weights:

    w' = w + gamma * (1/sqrt(K)) * sum_k b_k * s_k ,   b_k in {-1,+1},  s_k ~ N(0,1)^n

The payload is recovered blindly by correlation, exploiting that the host weights are
uncorrelated with the spreading sequences:

    corr_k = <w', s_k> / n  ~=  gamma * b_k / sqrt(K)  +  host-noise (mean 0)
    b_hat_k = 1 if corr_k > 0 else 0

The spreading gain gives robustness to SMALL weight perturbations (light pruning,
fine-tuning), which is MaleficNet's design goal. The amplitude gamma is kept well below
the per-tensor INT8 quantization step so that FP32 accuracy is preserved -- which is
exactly why INT8 quantization, whose step is coarser than the signal, erases it. This
module lets us MEASURE that on the same pipeline as QuantGhost, rather than quote it from
the method's reported design.

The embedding operates on a flattened concatenation of weight tensors; recovery mirrors
it. Error correction (MaleficNet uses LDPC) is intentionally omitted: we report the raw
correlator bit-recovery, the fair per-bit survival signal, since an LDPC layer cannot
repair a channel already at chance.
"""
from __future__ import annotations
import numpy as np


def embed(w: np.ndarray, payload_bits: np.ndarray, gamma: float,
          seed: int = 0) -> np.ndarray:
    """Add a DSSS payload signal to the flat weight vector `w`.

    gamma is the absolute per-weight signal std (so the perturbation std equals gamma).
    Returns the stego weight vector (same shape as `w`)."""
    w = np.ascontiguousarray(w, dtype=np.float64)
    bits = np.asarray(payload_bits).astype(np.float64)
    b = bits * 2.0 - 1.0                      # {0,1} -> {-1,+1}
    K, n = b.size, w.size
    rng = np.random.default_rng(seed)
    signal = np.zeros(n, dtype=np.float64)
    for k in range(K):
        sk = rng.standard_normal(n)           # Gaussian spreading sequence for bit k
        signal += b[k] * sk
    signal /= np.sqrt(K)                       # unit per-weight signal variance
    return (w + gamma * signal).astype(np.float32)


def extract(w: np.ndarray, n_bits: int, seed: int = 0) -> np.ndarray:
    """Blind correlation decode of `n_bits` payload bits from flat weight vector `w`.

    Regenerates the same spreading sequences (same seed/order) and correlates."""
    w = np.ascontiguousarray(w, dtype=np.float64)
    n = w.size
    rng = np.random.default_rng(seed)
    out = np.zeros(n_bits, dtype=np.uint8)
    for k in range(n_bits):
        sk = rng.standard_normal(n)
        out[k] = 1 if float(np.dot(w, sk)) > 0.0 else 0
    return out


def recovery(w_stego: np.ndarray, payload_bits: np.ndarray, seed: int = 0) -> float:
    """Convenience: bit-recovery rate of the DSSS channel (chance 0.5)."""
    rec = extract(w_stego, len(payload_bits), seed=seed)
    return float((rec == np.asarray(payload_bits).astype(np.uint8)).mean())
