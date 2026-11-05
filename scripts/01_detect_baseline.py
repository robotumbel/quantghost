"""Milestone 1: does a torch SRNet detect naive X-LSB embedding at high capacity?

Build benign vs X-LSB-stego GF byte-image tiles from a torchvision zoo, split by
MODEL (cross-model / few-shot setting), train SRNet, report held-out ROC-AUC.
Expectation (reproducing Model X-Ray): high detection at X>=8.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from wstego import embed, gf_image, detector, zoo


def auc(scores, labels):
    """ROC-AUC via rank statistic (no sklearn dependency)."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def build_tiles(weights, x, tile, per_model, seed):
    """Return tiles (N,tile,tile) and labels: clean=0, stego=1, plus model ids."""
    Xs, ys, mids = [], [], []
    for mid, (name, w) in enumerate(weights.items()):
        payload = embed.random_payload_bits(1 << 16, seed=seed + mid)
        w_st = embed.embed_xlsb(w, payload, x=x, fill=True)
        tc = gf_image.weights_to_tiles(w, tile=tile, max_tiles=per_model,
                                       shuffle_seed=seed + mid)
        ts = gf_image.weights_to_tiles(w_st, tile=tile, max_tiles=per_model,
                                       shuffle_seed=seed + mid)
        Xs += [tc, ts]; ys += [np.zeros(len(tc)), np.ones(len(ts))]
        mids += [np.full(len(tc), mid), np.full(len(ts), mid)]
    return (np.concatenate(Xs), np.concatenate(ys).astype(np.int64),
            np.concatenate(mids))


def batches(X, y, bs, dev, shuffle=True):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        xb = torch.from_numpy(X[j].astype(np.float32)[:, None] / 255.0).to(dev)
        yb = torch.from_numpy(y[j]).to(dev)
        yield xb, yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=int, default=8, help="LSB bits embedded")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--per_model", type=int, default=60)
    ap.add_argument("--n_test_models", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--time_budget", type=float, default=540, help="seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== loading zoo (X={args.x}, tile={args.tile}) ===", flush=True)
    W = zoo.load_zoo()
    names = list(W.keys())
    print(f"loaded {len(names)} models", flush=True)

    # cross-model split: last n_test_models held out
    test_names = names[-args.n_test_models:]
    train_names = names[:-args.n_test_models]
    Wtr = {n: W[n] for n in train_names}
    Wte = {n: W[n] for n in test_names}
    print(f"train models: {train_names}\ntest models : {test_names}", flush=True)

    print("=== building tiles ===", flush=True)
    Xtr, ytr, _ = build_tiles(Wtr, args.x, args.tile, args.per_model, args.seed)
    Xte, yte, _ = build_tiles(Wte, args.x, args.tile, args.per_model, args.seed + 100)
    print(f"train tiles {Xtr.shape} (pos={ytr.sum()}) | test {Xte.shape} (pos={yte.sum()})",
          flush=True)

    net = detector.make_detector(width=args.width, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time(); best = 0.0
    for ep in range(1, args.epochs + 1):
        net.train()
        tot = 0.0
        for xb, yb in batches(Xtr, ytr, args.bs, dev):
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        # eval
        net.eval(); sc = []
        with torch.no_grad():
            for xb, yb in batches(Xte, yte, args.bs, dev, shuffle=False):
                p = torch.softmax(net(xb), 1)[:, 1].cpu().numpy()
                sc.append(p)
        sc = np.concatenate(sc)
        a = auc(sc, yte)
        acc = ((sc > 0.5).astype(int) == yte).mean()
        best = max(best, a)
        el = time.time() - t0
        print(f"ep{ep:02d} loss={tot/len(Xtr):.4f} test_AUC={a:.3f} test_acc={acc:.3f} "
              f"[{el:.0f}s]", flush=True)
        if el > args.time_budget:
            print("time budget reached, stopping", flush=True); break
    print(f"\nBEST cross-model test AUC = {best:.3f}  (X={args.x})", flush=True)


if __name__ == "__main__":
    main()
