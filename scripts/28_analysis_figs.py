"""Deep-analysis data for the paper figures (run on the CIFAR carrier zoo):
  (A) mantissa-byte entropy DISTRIBUTION across models: benign vs naive X-LSB vs ours
      -> shows ours overlaps benign while naive sits at the max-entropy ceiling (stealth).
  (B) sub-quantum residual SHAPE: benign (peaked at 0) vs ours-basic (uniform) vs adaptive
      -> the mechanism the detector exploits.
  (C) survival vs quantization bit-width: ours (embed at matching grid) vs naive X-LSB
      -> robustness across INT8..INT3, not just INT8.
Saves results/ANL_*.npz and prints summary stats so we can sanity-check before plotting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wstego import robust

QMAX = 127
BINS = 40


def naive_xlsb_embed(w, bits, nbit=8):
    """Write nbit low mantissa bits of each float32 weight with payload (EvilModel-style)."""
    u = w.astype(np.float32).view(np.uint32).copy()
    nb = len(u)
    pay = bits[:nb * nbit]
    if len(pay) < nb * nbit:
        pay = np.concatenate([pay, np.zeros(nb * nbit - len(pay), np.uint8)])
    chunks = pay.reshape(nb, nbit)
    vals = np.zeros(nb, np.uint32)
    for k in range(nbit):
        vals |= (chunks[:, k].astype(np.uint32) << k)
    mask = np.uint32((1 << nbit) - 1)
    u = (u & ~mask) | vals
    return u.view(np.float32)


def naive_xlsb_extract(w, nbit=8):
    u = w.astype(np.float32).view(np.uint32)
    out = []
    for k in range(nbit):
        out.append(((u >> k) & 1).astype(np.uint8))
    return np.stack(out, axis=1).reshape(-1)


def residual(w, qmax=QMAX):
    s = robust.quant_scale(w, qmax)
    return w / s - np.round(w / s)


def main():
    Z = np.load("data/zoo_cifar.npz"); W = Z["weights"].astype(np.float32)
    print(f"zoo {W.shape}", flush=True)
    rng = np.random.default_rng(0)
    pbits = rng.integers(0, 2, W.shape[1] * 8, dtype=np.uint8)

    # (A) mantissa entropy distribution -------------------------------------
    ent_benign, ent_naive, ent_ours = [], [], []
    for w in W:
        ent_benign.append(robust.mantissa_lowbyte_entropy(w))
        ent_naive.append(robust.mantissa_lowbyte_entropy(naive_xlsb_embed(w, pbits, 8)))
        wo, _, _ = robust.qaware_embed_stealthy(w, pbits[:w.size], 1, QMAX, seed=0)
        ent_ours.append(robust.mantissa_lowbyte_entropy(wo))
    A = dict(benign=np.array(ent_benign), naive=np.array(ent_naive), ours=np.array(ent_ours))
    np.savez("results/ANL_entropy.npz", **A)
    print("\n(A) mantissa-byte entropy (bits, max=8):")
    for k, v in A.items():
        print(f"   {k:8s} mean={v.mean():.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}", flush=True)

    # (B) PER-TENSOR residual shape (the signal the detector exploits) -------
    SIZES = robust.ZOO_SMALLCNN_TENSOR_SIZES
    edges = np.linspace(-0.5, 0.5, BINS + 1)

    def per_tensor_resid_hist(w):
        hs, off = [], 0
        for n in SIZES:
            seg = w[off:off + n]; off += n
            s = robust.quant_scale(seg, QMAX)
            r = seg / s - np.round(seg / s)
            h, _ = np.histogram(r, edges, density=True)
            hs.append(h)
        return np.mean(hs, axis=0)

    def per_tensor_embed_resid_hist(w, noise, pin):
        hs, off = [], 0
        for n in SIZES:
            seg = w[off:off + n]; off += n
            wb, scale, carrier = robust.qaware_embed(seg, pbits, 1, QMAX)
            q = np.round(wb / scale).astype(np.int32)
            u = rng.uniform(-noise, noise, size=n).astype(np.float32)
            u[int(np.argmax(np.abs(q)))] = 0.0
            if pin:
                u[~carrier] = 0.0          # non-carrier weights stay exactly on the grid
            r = (q + u) - np.round(q + u)
            h, _ = np.histogram(r, edges, density=True)
            hs.append(h)
        return np.mean(hs, axis=0)

    hb = np.mean([per_tensor_resid_hist(w) for w in W], axis=0)
    h1 = np.mean([per_tensor_embed_resid_hist(w, 0.49, True) for w in W], axis=0)
    h2 = np.mean([per_tensor_embed_resid_hist(w, 0.499, False) for w in W], axis=0)
    np.savez("results/ANL_residual.npz", edges=edges, benign=hb, basic=h1, adaptive=h2)
    cen = 0.5 * (edges[:-1] + edges[1:])
    pk = lambda h: h[np.abs(cen) < 0.05].sum() / h.sum()
    print("\n(B) per-tensor residual peakedness (frac mass |r|<0.05; uniform ~0.10):")
    for nm, h in [("benign", hb), ("basic", h1), ("adaptive", h2)]:
        print(f"   {nm:8s} {pk(h):.3f}", flush=True)

    # (C) survival vs bit-width (mirror script 26: embed+extract at same grid)
    grids = [(127, "INT8"), (63, "INT7"), (31, "INT6"), (15, "INT5"), (7, "INT4"), (3, "INT3")]
    surv_ours, surv_naive = [], []
    pb0 = pbits[:w0.size] if (w0 := W[0]).size else pbits
    for qmax, lbl in grids:
        wo, _, _ = robust.qaware_embed_stealthy(w0, pb0, 1, qmax, seed=0)
        s = robust.quant_scale(wo, qmax)
        wq = robust.quantize(wo, s, qmax).astype(np.float32) * s
        rec = robust.qaware_extract(wq, robust.quant_scale(wq, qmax), 1, qmax)
        surv_ours.append((rec == pb0).mean())
        wn = naive_xlsb_embed(w0, pbits, 1)
        sN = robust.quant_scale(wn, qmax)
        wnq = robust.quantize(wn, sN, qmax).astype(np.float32) * sN
        recN = naive_xlsb_extract(wnq, 1)
        surv_naive.append((recN[:w0.size] == pbits[:w0.size]).mean())
    np.savez("results/ANL_survbits.npz",
             labels=np.array([l for _, l in grids]),
             ours=np.array(surv_ours), naive=np.array(surv_naive))
    print("\n(C) survival vs bit-width (ours embed at matching grid; chance 0.5):")
    for (qmax, lbl), so, sn in zip(grids, surv_ours, surv_naive):
        print(f"   {lbl}: ours={so:.3f}  naive={sn:.3f}", flush=True)


if __name__ == "__main__":
    main()
