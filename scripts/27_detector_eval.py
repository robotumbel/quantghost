"""Thorough evaluation of the sub-quantum residual detector (TIFS-grade):
  (a) multiple-seed cross-validated ROC-AUC (mean +/- std) for basic & adaptive attacks
      and the mantissa-entropy baseline;
  (b) ROC curves (out-of-fold scores) -> results/DET_roc.npz for a figure;
  (c) adaptive arms-race sweep: detector AUC vs the attacker's sub-quantum offset width
      delta (the wider the offset, the more uniform the residual, the lower the AUC),
      quantifying the stealth-vs-detectability frontier.

  python scripts/27_detector_eval.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wstego import robust
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

SIZES = robust.ZOO_SMALLCNN_TENSOR_SIZES
QMAX, BINS = 127, 40


def split(w):
    out, off = [], 0
    for n in SIZES:
        out.append(w[off:off+n]); off += n
    return out


def feat(w):
    fs = []
    for seg in split(w):
        s = robust.quant_scale(seg, QMAX)
        r = seg/s - np.round(seg/s)
        h, _ = np.histogram(r, bins=BINS, range=(-0.5, 0.5))
        fs.append(h/max(h.sum(), 1))
    return np.mean(fs, axis=0)


def embed_qg(w, payload, seed, noise=0.49, pin_exact=True):
    out = np.empty_like(w); off = 0
    for j, n in enumerate(SIZES):
        seg = w[off:off+n].astype(np.float32)
        w_grid, scale, carrier = robust.qaware_embed(seg, payload, 1, QMAX)
        q = np.round(w_grid/scale).astype(np.int32)
        rng = np.random.default_rng(seed+j)
        u = rng.uniform(-noise, noise, size=n).astype(np.float32)
        u[int(np.argmax(np.abs(q)))] = 0.0
        if pin_exact: u[~carrier] = 0.0
        out[off:off+n] = np.clip((q.astype(np.float32)+u)*scale, -QMAX*scale, QMAX*scale); off += n
    return out


def auc_seeds(W, Xb, kind, seeds=5):
    aucs = []
    for sd in range(seeds):
        payload = np.random.default_rng(sd).integers(0, 2, 1 << 16, dtype=np.uint8)
        noise, pin = (0.49, True) if kind == "basic" else (0.499, False)
        Xs = np.array([feat(embed_qg(w, payload, 1000*sd + i, noise, pin)) for i, w in enumerate(W)])
        X = np.r_[Xb, Xs]; y = np.r_[np.zeros(len(W)), np.ones(len(W))]
        aucs.append(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())
    return np.mean(aucs), np.std(aucs)


def main():
    np.random.seed(0)
    Z = np.load("data/zoo_cifar.npz"); W = Z["weights"].astype(np.float32)
    print(f"zoo {W.shape}\n", flush=True)
    Xb = np.array([feat(w) for w in W])
    eb = np.array([robust.mantissa_lowbyte_entropy(w) for w in W]).reshape(-1, 1)

    # (a) mean +/- std AUC over seeds
    print("Detector ROC-AUC (5-fold CV, 5 seeds, mean +/- std):", flush=True)
    payload0 = np.random.default_rng(0).integers(0, 2, 1 << 16, dtype=np.uint8)
    es = np.array([robust.mantissa_lowbyte_entropy(embed_qg(w, payload0, i)) for i, w in enumerate(W)]).reshape(-1, 1)
    y = np.r_[np.zeros(len(W)), np.ones(len(W))]
    a_ent = cross_val_score(LogisticRegression(max_iter=2000), np.r_[eb, es], y, cv=5, scoring="roc_auc").mean()
    mb, sb = auc_seeds(W, Xb, "basic"); ma, sa = auc_seeds(W, Xb, "adaptive")
    print(f"  mantissa entropy (baseline)      {a_ent:.3f}", flush=True)
    print(f"  residual detector, basic attack  {mb:.3f} +/- {sb:.3f}", flush=True)
    print(f"  residual detector, adaptive      {ma:.3f} +/- {sa:.3f}\n", flush=True)

    # (b) ROC curves (out-of-fold scores)
    roc = {}
    for kind in ("basic", "adaptive"):
        noise, pin = (0.49, True) if kind == "basic" else (0.499, False)
        Xs = np.array([feat(embed_qg(w, payload0, i, noise, pin)) for i, w in enumerate(W)])
        X = np.r_[Xb, Xs]
        sc = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, sc)
        roc[f"{kind}_fpr"], roc[f"{kind}_tpr"] = fpr, tpr
    np.savez("results/DET_roc.npz", **roc)
    print("ROC curves -> results/DET_roc.npz", flush=True)

    # (c) adaptive arms-race sweep: AUC vs offset width delta
    print("\nArms-race sweep (attacker sub-quantum offset width delta vs detector AUC):", flush=True)
    deltas = [0.10, 0.20, 0.30, 0.40, 0.49, 0.499]
    sweep = []
    for d in deltas:
        Xs = np.array([feat(embed_qg(w, payload0, i, noise=d, pin_exact=False)) for i, w in enumerate(W)])
        auc = cross_val_score(LogisticRegression(max_iter=2000), np.r_[Xb, Xs], y, cv=5, scoring="roc_auc").mean()
        sweep.append((d, auc)); print(f"  delta={d:.3f}  AUC={auc:.3f}", flush=True)
    np.savez("results/DET_sweep.npz", delta=np.array([d for d, _ in sweep]), auc=np.array([a for _, a in sweep]))
    print("\nbaca: detektor menangkap basic (AUC~1.0); attacker melebarkan offset -> residual "
          "makin seragam -> AUC turun mendekati chance, tapi terbatas oleh pin & bentuk benign.", flush=True)


if __name__ == "__main__":
    main()
