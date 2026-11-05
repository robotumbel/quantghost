"""Train a HOMOGENEOUS zoo of small CNNs (same arch, different seeds), FAST.

GPU-resident data (no DataLoader bottleneck): CIFAR-10 is loaded once to the GPU
as uint8; batches are sliced and normalized on the fly with cheap flip
augmentation. A zoo is just a carrier set, so ~70% accuracy is plenty. Weights
are saved flattened to one .npz for the OSL detector reproduction.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
torch.backends.cudnn.benchmark = True


class SmallCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        def blk(i, o): return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            blk(3, 32), blk(32, 32), nn.MaxPool2d(2),
            blk(32, 64), blk(64, 64), nn.MaxPool2d(2),
            blk(64, 128), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, nc)

    def forward(self, x):
        return self.head(self.f(x).flatten(1))


def flat_weights(model):
    return np.concatenate([p.detach().cpu().numpy().astype(np.float32).ravel()
                           for p in model.parameters() if p.dim() >= 2])


def load_cifar_gpu(root, dev):
    tr = tv.datasets.CIFAR10(root, train=True, download=True)
    te = tv.datasets.CIFAR10(root, train=False, download=True)
    def to_gpu(ds):
        x = torch.from_numpy(ds.data).to(dev).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        y = torch.tensor(ds.targets, device=dev)
        return x, y
    return to_gpu(tr), to_gpu(te)


def train_one(seed, data, dev, epochs, bs, subset=0):
    (Xtr, ytr), (Xte, yte) = data
    if subset and subset < len(Xtr):                  # fewer iters: WDDM is launch-bound
        sel = torch.randperm(len(Xtr), device=dev)[:subset]
        Xtr, ytr = Xtr[sel], ytr[sel]
    torch.manual_seed(seed)
    net = SmallCNN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    n = len(Xtr)
    g = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb = Xtr[idx].float() / 255.0            # no per-iter CPU sync
            yb = ytr[idx]
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(Xte), 1000):
            xb = Xte[i:i+1000].float() / 255.0
            correct += (net(xb).argmax(1) == yte[i:i+1000]).sum().item()
    return flat_weights(net), correct / len(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--out", default="data/zoo_cifar.npz")
    ap.add_argument("--data", default="data/cifar")
    ap.add_argument("--subset", type=int, default=10000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    print("loading CIFAR to GPU...", flush=True)
    data = load_cifar_gpu(a.data, dev)
    P = flat_weights(SmallCNN()).size
    print(f"arch weight-params={P:,} bytes={4*P:,} (100x100 GF ok: {4*P>=10000})", flush=True)

    Ws, accs, t0 = [], [], time.time()
    for i in range(a.n_models):
        w, acc = train_one(1000 + i, data, dev, a.epochs, a.bs, a.subset)
        Ws.append(w); accs.append(acc)
        print(f"model {i+1:02d}/{a.n_models} acc={acc:.3f} [{time.time()-t0:.0f}s]", flush=True)
    W = np.stack(Ws).astype(np.float32)
    np.savez_compressed(a.out, weights=W, acc=np.array(accs))
    print(f"\nsaved {a.out}: {W.shape} mean_acc={np.mean(accs):.3f}", flush=True)


if __name__ == "__main__":
    main()
