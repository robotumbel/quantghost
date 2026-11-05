"""Evaluate the sub-quantum residual detector on the DIVERSE GHRP STL-10 zoo
(3000 models, varied seeds/activations/LR/init/dropout), not the narrow CIFAR carrier
zoo. Closes the "homogeneous benign population" gap for our proposed defense.

GHRP small CNN (use_bias=True, layer_lst=conv,conv,conv,fc,fc), 2864 params per model.
Derived weight-tensor layout (interleaved weight,bias per layer; verified to sum to 2864):
  conv1 Conv2d(3,8,5): w[0:600]   b[600:608]
  conv2 Conv2d(8,6,5): w[608:1808] b[1808:1814]
  conv3 Conv2d(6,4,2): w[1814:1910] b[1910:1914]
  fc1   Linear(36,20): w[1914:2634] b[2634:2654]
  fc2   Linear(10,20): w[2654:2854] b[2854:2864]

  python scripts/30_detector_ghrp.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wstego import robust
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve

# weight-tensor segments (dim>=2 only), as (start, stop) in the 2864-vector
WSEGS = [(0, 600), (608, 1808), (1814, 1910), (1914, 2634), (2654, 2854)]
QMAX, BINS = 127, 40
EDGES = np.linspace(-0.5, 0.5, BINS + 1)


def feat(w):
    fs = []
    for a, b in WSEGS:
        seg = w[a:b]
        s = robust.quant_scale(seg, QMAX)
        r = seg / s - np.round(seg / s)
        h, _ = np.histogram(r, EDGES)
        fs.append(h / max(h.sum(), 1))
    return np.mean(fs, axis=0)


def embed_qg(w, payload, seed, noise=0.49, pin_exact=True):
    out = w.copy()
    for j, (a, b) in enumerate(WSEGS):
        seg = w[a:b].astype(np.float32); n = b - a
        w_grid, scale, carrier = robust.qaware_embed(seg, payload[:n], 1, QMAX)
        q = np.round(w_grid / scale).astype(np.int32)
        rng = np.random.default_rng(seed + j)
        u = rng.uniform(-noise, noise, size=n).astype(np.float32)
        u[int(np.argmax(np.abs(q)))] = 0.0
        if pin_exact:
            u[~carrier] = 0.0
        out[a:b] = np.clip((q + u) * scale, -QMAX * scale, QMAX * scale)
    return out


def auc_seeds(W, Xb, kind, seeds=5):
    aucs = []
    y = np.r_[np.zeros(len(W)), np.ones(len(W))]
    for sd in range(seeds):
        pay = np.random.default_rng(sd).integers(0, 2, 4096, dtype=np.uint8)
        noise, pin = (0.49, True) if kind == "basic" else (0.499, False)
        Xs = np.array([feat(embed_qg(w, pay, 1000 * sd + i, noise, pin)) for i, w in enumerate(W)])
        aucs.append(cross_val_score(LogisticRegression(max_iter=2000),
                                    np.r_[Xb, Xs], y, cv=5, scoring="roc_auc").mean())
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    Z = np.load("data/ghrp_stl10_sub.npz")["weights"].astype(np.float32)
    # sanity: weight-segment magnitudes should look like trained weights (not all-zero)
    print(f"GHRP zoo {Z.shape}", flush=True)
    seg_std = [float(np.std(Z[:, a:b])) for a, b in WSEGS]
    print("per-segment weight std (sanity):", [f"{s:.3f}" for s in seg_std], flush=True)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z), size=min(400, len(Z)), replace=False)
    W = Z[idx]
    Xb = np.array([feat(w) for w in W])

    print("\nResidual detector on DIVERSE GHRP zoo (5-fold CV, 5 seeds, mean+/-std):", flush=True)
    mb, sb = auc_seeds(W, Xb, "basic")
    ma, sa = auc_seeds(W, Xb, "adaptive")
    print(f"  residual detector, basic attack    {mb:.3f} +/- {sb:.3f}", flush=True)
    print(f"  residual detector, adaptive attack {ma:.3f} +/- {sa:.3f}", flush=True)
    print(f"  (CIFAR carrier zoo was 1.000 / 0.754 -- compare generalization)", flush=True)

    # ROC for the figure
    y = np.r_[np.zeros(len(W)), np.ones(len(W))]
    pay = np.random.default_rng(0).integers(0, 2, 4096, dtype=np.uint8)
    roc = {}
    for kind in ("basic", "adaptive"):
        noise, pin = (0.49, True) if kind == "basic" else (0.499, False)
        Xs = np.array([feat(embed_qg(w, pay, i, noise, pin)) for i, w in enumerate(W)])
        sc = cross_val_predict(LogisticRegression(max_iter=2000), np.r_[Xb, Xs], y, cv=5,
                               method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, sc)
        roc[f"{kind}_fpr"], roc[f"{kind}_tpr"] = fpr, tpr
    np.savez("results/DET_ghrp_roc.npz", **roc)
    print("\nROC -> results/DET_ghrp_roc.npz", flush=True)


if __name__ == "__main__":
    main()
