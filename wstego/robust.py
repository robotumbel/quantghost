"""Quantization-robust payload embedding (Angle D, Phase 3).

Naive X-LSB hides in FP32 mantissa LSBs, which FP32->INT8 quantization erases
(measured: 0% survival). The quantization-AWARE scheme instead anticipates the
victim's per-tensor INT8 quantization and writes the payload into the INT8 LSBs
*by construction*: set each FP32 weight to a value that quantizes exactly to an
INT8 code whose low bit(s) carry the payload. After the victim quantizes, the
payload is read back from the INT8 codes -> survives by design.

This is the foundational survival proof. Capacity = nbits per weight (1 bit/weight
already gives MB-scale for multi-million-param models). Stealth and accuracy
trade-offs are handled in later phases.
"""
from __future__ import annotations
import numpy as np


def quant_scale(w: np.ndarray, qmax: int = 127) -> float:
    """Per-tensor symmetric INT8 scale (standard PTQ)."""
    m = np.abs(w).max()
    return float(m / qmax) if m > 0 else 1.0


def quantize(w: np.ndarray, scale: float, qmax: int = 127) -> np.ndarray:
    return np.clip(np.round(w / scale), -qmax, qmax).astype(np.int32)


def qaware_embed(w: np.ndarray, payload_bits: np.ndarray, nbits: int = 1,
                 qmax: int = 127) -> tuple[np.ndarray, float, np.ndarray]:
    """Embed payload into the INT8 LSBs that survive quantization.

    Returns (fp32_weights, scale, carrier_mask). carrier_mask marks weights that
    actually carry payload (the scale-defining max-magnitude weights are PINNED to
    keep |q|=qmax so the victim's recomputed per-tensor scale matches ours -> the
    INT8 grid is identical and re-quantization is exact -> ~100% survival).
    """
    scale = quant_scale(w, qmax)
    q = quantize(w, scale, qmax)
    n = w.size
    cap = n * nbits
    reps = int(np.ceil(cap / max(len(payload_bits), 1)))
    bits = np.tile(payload_bits, reps)[:cap].reshape(n, nbits).astype(np.int32)
    field = np.zeros(n, dtype=np.int32)
    for i in range(nbits):
        field = (field << 1) | bits[:, i]
    mask = (1 << nbits) - 1
    sign = np.sign(q); sign[sign == 0] = 1
    mag = np.abs(q)
    mag_emb = (mag & ~mask) | (field & mask)
    mag_emb = np.clip(mag_emb, 0, qmax)
    # PIN the scale: keep every max-magnitude weight at qmax (no payload there)
    pinned = (mag >= qmax)
    mag_emb[pinned] = qmax
    carrier = ~pinned
    q_emb = (sign * mag_emb).astype(np.int32)
    w_emb = (q_emb.astype(np.float32) * scale)
    return w_emb, scale, carrier


def qaware_embed_stealthy(w: np.ndarray, payload_bits: np.ndarray, nbits: int = 1,
                          qmax: int = 127, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Survival + STEALTH: place each weight at a RANDOM point inside the INT8 bin
    that maps to its payload-bearing code, instead of exactly on the grid.

    Quantization maps the whole bin [(q-0.5)s,(q+0.5)s) -> code q, so the payload
    still survives, but the sub-quantum part (the FP32 mantissa) stays natural ->
    weights are NOT grid-aligned and mantissa entropy is preserved. One max weight
    is pinned to fix the victim's per-tensor scale.
    """
    w_grid, scale, carrier = qaware_embed(w, payload_bits, nbits, qmax)
    q = np.round(w_grid / scale).astype(np.int32)
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.49, 0.49, size=w.size).astype(np.float32)
    argmax = int(np.argmax(np.abs(q)))
    noise[argmax] = 0.0                       # pin scale-defining weight exactly
    w_st = (q.astype(np.float32) + noise) * scale
    # ensure nothing exceeds the pinned magnitude (keep victim scale == scale)
    cap = qmax * scale
    w_st = np.clip(w_st, -cap, cap)
    return w_st, scale, carrier


def qaware_embed_minimal(w: np.ndarray, payload_bits: np.ndarray, qmax: int = 127
                         ) -> tuple[np.ndarray, float, np.ndarray]:
    """Distribution-preserving 1-bit embed: change a weight ONLY if its INT8 code's
    LSB already mismatches the payload bit; then shift it by exactly ONE quantum
    while keeping its natural sub-quantum offset. ~half the weights are untouched,
    the rest move by 1 code -> the FP32 distribution is preserved (stealthy) and
    the payload still survives quantization (correct bin by construction).
    """
    scale = quant_scale(w, qmax)
    q = quantize(w, scale, qmax)
    n = w.size
    bits = np.tile(payload_bits, int(np.ceil(n / max(len(payload_bits), 1))))[:n].astype(np.int32)
    mag = np.abs(q); sign = np.sign(q); sign[sign == 0] = 1
    frac = w / scale - q                                  # natural sub-quantum offset
    mism = (mag & 1) != bits                              # codes whose LSB is wrong
    # move mismatched codes by +1 (or -1 at the ceiling) to flip the LSB
    step = np.where(mag >= qmax, -1, 1)
    mag_new = np.where(mism, mag + step, mag)
    mag_new = np.clip(mag_new, 0, qmax)
    # pin the scale: keep one max weight exactly at qmax
    pinned = (mag >= qmax)
    mag_new[pinned] = qmax
    carrier = ~pinned
    q_new = sign * mag_new
    w_new = (q_new.astype(np.float32) + frac.astype(np.float32)) * scale  # keep natural frac
    cap = qmax * scale
    w_new = np.clip(w_new, -cap, cap)
    return w_new, scale, carrier


def qaware_extract(w: np.ndarray, scale: float, nbits: int = 1,
                   qmax: int = 127) -> np.ndarray:
    """Recover payload bits from (re-)quantized weights."""
    q = quantize(w, scale, qmax)
    mag = np.abs(q).astype(np.int32)
    mask = (1 << nbits) - 1
    field = mag & mask
    bits = np.zeros((w.size, nbits), dtype=np.uint8)
    for i in range(nbits):
        bits[:, nbits - 1 - i] = (field >> i) & 1
    return bits.reshape(-1)


def survival_rate(recovered: np.ndarray, payload_bits: np.ndarray,
                  nbits: int = 1, carrier: np.ndarray | None = None) -> float:
    """Bit-survival over carrier weights. `recovered` is the per-weight bit field
    flattened (n*nbits). carrier (length n) marks payload-bearing weights."""
    cap = recovered.size
    reps = int(np.ceil(cap / max(len(payload_bits), 1)))
    truth = np.tile(payload_bits, reps)[:cap].astype(np.uint8)
    if carrier is not None:
        keep = np.repeat(carrier, nbits)
        recovered, truth = recovered[keep], truth[keep]
    return float((recovered == truth).mean())


# --- B2/B3: per-channel quantization + arbitrary qmax (INT8 qmax=127, INT4 qmax=7) ---
#
# Real PTQ often uses PER-CHANNEL (per output-channel) scales, not one per tensor.
# The survival-by-construction argument is identical, applied independently per row:
# each channel pins its own max-magnitude weight so the victim's recomputed
# per-channel scale matches ours. Reshape a weight tensor to 2D (C_out, rest) first.

def quant_scale_pc(w2d: np.ndarray, qmax: int = 127) -> np.ndarray:
    """Per-row (per output-channel) symmetric scale. w2d: (C, K) -> scales (C,)."""
    m = np.abs(w2d).max(axis=1)
    return np.where(m > 0, m / qmax, 1.0).astype(np.float32)


def quantize_pc(w2d: np.ndarray, scales: np.ndarray, qmax: int = 127) -> np.ndarray:
    return np.clip(np.round(w2d / scales[:, None]), -qmax, qmax).astype(np.int32)


def qaware_embed_stealthy_pc(w2d: np.ndarray, payload_bits: np.ndarray, nbits: int = 1,
                             qmax: int = 127, seed: int = 0
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel analogue of qaware_embed_stealthy. Returns (w_st 2D, scales (C,),
    carrier 2D bool). Each row pins its own scale-defining weight; payload lands in
    the per-channel INT codes' LSBs and survives per-channel quantization."""
    C, K = w2d.shape
    n = C * K
    scales = quant_scale_pc(w2d, qmax)
    q = quantize_pc(w2d, scales, qmax)                       # (C,K)
    cap = n * nbits
    reps = int(np.ceil(cap / max(len(payload_bits), 1)))
    bits = np.tile(payload_bits, reps)[:cap].reshape(n, nbits).astype(np.int32)
    field = np.zeros(n, dtype=np.int32)
    for i in range(nbits):
        field = (field << 1) | bits[:, i]
    mask = (1 << nbits) - 1
    qf = q.reshape(-1)
    sign = np.sign(qf); sign[sign == 0] = 1
    mag = np.abs(qf)
    mag_emb = (mag & ~mask) | (field & mask)
    mag_emb = np.clip(mag_emb, 0, qmax)
    # pin each row's max-magnitude weight (fixes that channel's scale)
    pin_col = np.abs(q).argmax(axis=1)                       # (C,)
    pin_flat = pin_col + np.arange(C) * K
    mag_emb[pin_flat] = qmax
    carrier = np.ones(n, dtype=bool); carrier[pin_flat] = False
    q_emb = (sign * mag_emb).astype(np.int32)
    # stealth: random sub-quantum offset inside each code's bin (pinned weights exact)
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.49, 0.49, size=n).astype(np.float32)
    noise[pin_flat] = 0.0
    scal_rep = np.repeat(scales, K)
    w_st = (q_emb.astype(np.float32) + noise) * scal_rep
    cap_clip = qmax * scal_rep
    w_st = np.clip(w_st, -cap_clip, cap_clip)
    return w_st.reshape(C, K), scales, carrier.reshape(C, K)


def qaware_extract_pc(w2d: np.ndarray, scales: np.ndarray, nbits: int = 1,
                      qmax: int = 127) -> np.ndarray:
    """Recover payload bits from per-channel (re-)quantized weights (flattened)."""
    q = quantize_pc(w2d, scales, qmax)
    mag = np.abs(q).reshape(-1).astype(np.int32)
    mask = (1 << nbits) - 1
    field = mag & mask
    bits = np.zeros((mag.size, nbits), dtype=np.uint8)
    for i in range(nbits):
        bits[:, nbits - 1 - i] = (field >> i) & 1
    return bits.reshape(-1)


# --- C3/C4 stealth fix: PER-TENSOR embedding on a flattened multi-tensor vector ---
#
# The OSL/value-distribution detectors (Model X-Ray's img_pp + weight_value_dist)
# do not just look at mantissa bytes -- they see the whole FP32 value. Embedding a
# flat multi-tensor weight vector with ONE GLOBAL per-tensor scale is the bug: the
# global max (dominated by the largest tensor) defines a coarse grid that crushes
# small-magnitude tensors into a few bins near zero, shifting the value
# distribution detectably (measured: real OSL centroid 0.747, weight_value_dist
# 0.980 -> DETECTED). Embedding each constituent tensor with ITS OWN scale keeps
# every tensor's value distribution intact while preserving survival-by-construction
# per tensor (each tensor pins its own scale-defining weight). Capacity is unchanged.

# Homogeneous CIFAR zoo (scripts/03_train_zoo.py SmallCNN): the dim>=2 params, in
# model.parameters() order, that flat_weights() concatenates. Sum = 140,384.
ZOO_SMALLCNN_TENSOR_SIZES = (864, 9216, 18432, 36864, 73728, 1280)


def qaware_embed_stealthy_segmented(
    w_flat: np.ndarray, payload_bits: np.ndarray, sizes, nbits: int = 1,
    qmax: int = 127, seed: int = 0,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """Per-tensor qaware-stealthy embed over a flattened multi-tensor vector.

    `sizes` is the ordered list of constituent tensor lengths (must sum to
    w_flat.size). Each segment is embedded independently with its own per-tensor
    scale (so its value distribution is preserved) carrying the SAME payload bits.
    Returns (w_st_flat, per_segment_scales, carrier_flat)."""
    w_flat = np.ascontiguousarray(w_flat, dtype=np.float32)
    if int(np.sum(sizes)) != w_flat.size:
        raise ValueError(f"sizes sum {int(np.sum(sizes))} != flat length {w_flat.size}")
    out = np.empty_like(w_flat)
    carrier = np.empty(w_flat.size, dtype=bool)
    scales: list[float] = []
    off = 0
    for j, n in enumerate(sizes):
        seg = w_flat[off:off + n]
        w_st, scale, car = qaware_embed_stealthy(seg, payload_bits, nbits=nbits,
                                                 qmax=qmax, seed=seed + j)
        out[off:off + n] = w_st
        carrier[off:off + n] = car
        scales.append(scale)
        off += n
    return out, scales, carrier


def qaware_extract_segmented(
    w_flat: np.ndarray, sizes, scales, nbits: int = 1, qmax: int = 127,
) -> np.ndarray:
    """Recover payload bits from a per-tensor-embedded flat vector. Each segment is
    re-quantized with its own scale (mirrors the victim's per-tensor PTQ)."""
    w_flat = np.ascontiguousarray(w_flat, dtype=np.float32)
    rec = []
    off = 0
    for n, scale in zip(sizes, scales):
        seg = w_flat[off:off + n]
        rec.append(qaware_extract(seg, scale, nbits=nbits, qmax=qmax))
        off += n
    return np.concatenate(rec)


def mantissa_lowbyte_entropy(w: np.ndarray) -> float:
    """Shannon entropy (bits) of the low mantissa byte of FP32 weights. Natural
    trained weights ~8.0; grid-aligned (non-stealthy) embedding collapses it. The
    stealth signal Model X-Ray-class detectors look for."""
    a = np.ascontiguousarray(w, dtype=np.float32)
    lb = a.view(np.uint8).reshape(-1, 4)[:, 0]              # little-endian low byte
    hist = np.bincount(lb, minlength=256).astype(np.float64)
    p = hist / hist.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
