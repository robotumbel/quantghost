"""Proposed targeted detector for QuantGhost + adaptive-attacker analysis.

Detector model: per weight tensor we form the histogram of the sub-quantum residual
r = w/s - round(w/s) (40 bins over [-0.5,0.5)); the per-model feature is the mean
histogram across tensors, and a logistic-regression classifier separates benign from
stego (5-fold cross-validated ROC-AUC). The mantissa-byte entropy feature (which
QuantGhost defeats) is the baseline.

We then test an ADAPTIVE attacker that widens the sub-quantum offset to the full bin and
removes the exact-zero pin signature, to see whether the residual signal is fundamental
or merely an artefact of the basic construction.

  python scripts/25_detector.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wstego import robust
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

SIZES = robust.ZOO_SMALLCNN_TENSOR_SIZES
QMAX = 127
BINS = 40


def split_tensors(w):
    out, off = [], 0
    for n in SIZES:
        out.append(w[off:off + n]); off += n
    return out


def residual_hist_feature(w):
    feats = []
    for seg in split_tensors(w):
        s = robust.quant_scale(seg, QMAX)
        r = seg / s - np.round(seg / s)
        h, _ = np.histogram(r, bins=BINS, range=(-0.5, 0.5))
        feats.append(h / max(h.sum(), 1))
    return np.mean(feats, axis=0)


def embed_quantghost(w, payload, seed, noise=0.49, pin_exact=True):
    """QuantGhost per-tensor; noise controls sub-quantum offset range; pin_exact toggles
    the exact-zero pin (adaptive attacker sets noise->~0.5 and pin_exact=False except the
    single scale-defining weight, which must stay exact for survival)."""
    out = np.empty_like(w); off = 0
    for j, n in enumerate(SIZES):
        seg = w[off:off + n].astype(np.float32)
        w_grid, scale, carrier = robust.qaware_embed(seg, payload, 1, QMAX)
        q = np.round(w_grid / scale).astype(np.int32)
        rng = np.random.default_rng(seed + j)
        u = rng.uniform(-noise, noise, size=n).astype(np.float32)
        amax = int(np.argmax(np.abs(q)))
        u[amax] = 0.0                      # scale-defining weight pinned exactly (required)
        if pin_exact:                      # basic variant also leaves other pins exact
            u[~carrier] = 0.0
        w_st = (q.astype(np.float32) + u) * scale
        cap = QMAX * scale
        out[off:off + n] = np.clip(w_st, -cap, cap); off += n
    return out


def cv_auc(X, y):
    clf = LogisticRegression(max_iter=2000)
    return cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()


def main():
    Z = np.load("data/zoo_cifar.npz"); W = Z["weights"].astype(np.float32)
    print(f"zoo {W.shape}  (benign carriers)\n", flush=True)
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, 1 << 16, dtype=np.uint8)

    Xb = np.array([residual_hist_feature(w) for w in W])
    eb = np.array([robust.mantissa_lowbyte_entropy(w) for w in W]).reshape(-1, 1)

    # basic QuantGhost (offset +/-0.49, exact pins) and adaptive (near-full range, no extra pins)
    Wb = np.array([embed_quantghost(w, payload, i, noise=0.49, pin_exact=True) for i, w in enumerate(W)])
    Wa = np.array([embed_quantghost(w, payload, i, noise=0.499, pin_exact=False) for i, w in enumerate(W)])
    Xs_b = np.array([residual_hist_feature(w) for w in Wb])
    Xs_a = np.array([residual_hist_feature(w) for w in Wa])
    es = np.array([robust.mantissa_lowbyte_entropy(w) for w in Wb]).reshape(-1, 1)

    y = np.r_[np.zeros(len(W)), np.ones(len(W))]
    print(f"{'detector / attacker':42s} {'CV ROC-AUC':>10}", flush=True)
    print(f"{'mantissa entropy   vs basic QuantGhost':42s} {cv_auc(np.r_[eb, es], y):>10.3f}", flush=True)
    print(f"{'residual histogram vs basic QuantGhost':42s} {cv_auc(np.r_[Xb, Xs_b], y):>10.3f}", flush=True)
    print(f"{'residual histogram vs ADAPTIVE QuantGhost':42s} {cv_auc(np.r_[Xb, Xs_a], y):>10.3f}", flush=True)
    print("\nbaca: detektor residual menangkap QuantGhost dasar; lawan attacker adaptif "
          "(offset rentang penuh) sinyalnya melemah -> arms race, memotivasi pertahanan "
          "struktural (permutasi + inspeksi statistik invarian).", flush=True)


if __name__ == "__main__":
    main()
