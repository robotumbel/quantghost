"""Milestone 1b: reproduce the OSL detector. Target ~0.9 detection at X=8 on
RANDOM-fill, cross-model (matching Model X-Ray's fsl_osl.csv).

Uses the homogeneous CNN zoo (scripts/03_train_zoo.py output). Trains an embedding
net with triplet loss on TRAIN models (benign + X-LSB-stego tiles), evaluates
centroid+1NN on HELD-OUT models. Sweeps X to reproduce the detection-vs-X curve.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from wstego import embed, gf_image, osl


def build(weights_rows, x, tile, per_model, seed):
    """weights_rows: (M,P) zoo. Returns tiles, labels(0/1), model-id."""
    Xs, ys, mids = [], [], []
    for mid, w in enumerate(weights_rows):
        pl = embed.random_payload_bits(1 << 16, seed=seed + mid)
        wst = embed.embed_xlsb(w, pl, x=x, fill=True) if x > 0 else w
        tc = gf_image.weights_to_tiles(w,   tile=tile, max_tiles=per_model, shuffle_seed=seed+mid)
        ts = gf_image.weights_to_tiles(wst, tile=tile, max_tiles=per_model, shuffle_seed=seed+mid+777)
        Xs += [tc, ts]; ys += [np.zeros(len(tc)), np.ones(len(ts))]
        mids += [np.full(len(tc), mid), np.full(len(ts), mid)]
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int64), np.concatenate(mids)


def episodic_train(net, Xtr, ytr, dev, steps, bs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xtr)
    net.train()
    for s in range(steps):
        # balanced batch: half benign half stego
        i0 = np.random.choice(np.where(ytr == 0)[0], bs // 2)
        i1 = np.random.choice(np.where(ytr == 1)[0], bs // 2)
        idx = np.concatenate([i0, i1])
        xb = torch.from_numpy(Xtr[idx].astype(np.float32)[:, None] / 255.).to(dev)
        yb = torch.from_numpy(ytr[idx]).to(dev)
        z = net(xb)
        loss = osl.online_triplet_loss(z, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="data/zoo_cifar.npz")
    ap.add_argument("--x_range", type=int, nargs="+", default=[1, 4, 8, 12, 16, 23])
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--per_model", type=int, default=40)
    ap.add_argument("--n_test_models", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Z = np.load(a.zoo)
    W = Z["weights"]
    print(f"zoo: {W.shape}  mean_acc={Z['acc'].mean():.3f}", flush=True)
    M = len(W)
    test_ids = np.arange(M - a.n_test_models, M)
    train_ids = np.arange(0, M - a.n_test_models)
    print(f"train models={len(train_ids)} test models={len(test_ids)}", flush=True)

    print(f"\n{'X':>3} {'centroid':>9} {'1nn':>7}  (cross-model held-out)", flush=True)
    for x in a.x_range:
        t0 = time.time()
        Xtr, ytr, _ = build(W[train_ids], x, a.tile, a.per_model, a.seed)
        Xte, yte, _ = build(W[test_ids], x, a.tile, a.per_model, a.seed + 5000)
        net = osl.EmbeddingNet(emb=a.emb, width=a.width).to(dev)
        net = episodic_train(net, Xtr, ytr, dev, a.steps, a.bs, a.lr)
        cent, nn_ = osl.centroid_1nn_eval(net, Xtr, ytr, Xte, yte, dev)
        print(f"{x:>3} {cent:>9.3f} {nn_:>7.3f}   [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
