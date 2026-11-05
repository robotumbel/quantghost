"""B5: capacity-vs-robustness FRONTIER through fine-tuning.

Fine-tuning moves weights, so payload survival depends on REDUNDANCY (copies per
message bit, recovered by majority vote). We sweep message length n_msg (=> redundancy
P/n_msg) and measure message-recovery after fine-tuning at two intensities:
  - light transfer  (2 epochs, lr 2e-4)
  - heavy transfer  (8 epochs, lr 5e-4)   <- harder: weights move more
Output = recovery vs redundancy curve. The crossover (recovery -> 1.0) marks the
usable capacity at a given deployment fine-tuning budget.
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from wstego import robust
torch.backends.cudnn.benchmark = True


class SmallCNN(nn.Module):
    def __init__(s, nc=10):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        s.f = nn.Sequential(blk(3, 32), blk(32, 32), nn.MaxPool2d(2),
                            blk(32, 64), blk(64, 64), nn.MaxPool2d(2),
                            blk(64, 128), nn.AdaptiveAvgPool2d(1))
        s.h = nn.Linear(128, 10)
    def forward(s, x): return s.h(s.f(x).flatten(1))


def cifar(dev):
    tr = tv.datasets.CIFAR10("data/cifar", train=True, download=True)
    te = tv.datasets.CIFAR10("data/cifar", train=False, download=True)
    f = lambda d: (torch.from_numpy(d.data).to(dev).permute(0, 3, 1, 2).contiguous(),
                   torch.tensor(d.targets, device=dev))
    return f(tr), f(te)


def acc(net, Xte, yte, dev):
    net.eval(); c = 0
    with torch.no_grad():
        for i in range(0, len(Xte), 1000):
            c += (net(Xte[i:i + 1000].float() / 255.).argmax(1) == yte[i:i + 1000]).sum().item()
    return c / len(Xte)


def train(net, Xtr, ytr, dev, epochs, lr, bs=512, sub=10000):
    sel = torch.randperm(len(Xtr), device=dev)[:sub]; Xtr, ytr = Xtr[sel], ytr[sel]
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    for _ in range(epochs):
        p = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), bs):
            idx = p[i:i + bs]; opt.zero_grad()
            F.cross_entropy(net(Xtr[idx].float() / 255.), ytr[idx]).backward(); opt.step()


def weight_tensors(net):
    return [p for p in net.parameters() if p.dim() >= 2]


def embed_model(net, msg, seed=0):
    g0 = 0
    with torch.no_grad():
        for p in weight_tensors(net):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            bits = msg[(np.arange(g0, g0 + n)) % len(msg)].astype(np.uint8)
            w2, s, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed + g0)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device)); g0 += n


def extract_model(net, n_msg):
    votes = np.zeros((n_msg, 2), dtype=np.int64); g0 = 0
    with torch.no_grad():
        for p in weight_tensors(net):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            s = robust.quant_scale(w); bits = robust.qaware_extract(w, s, nbits=1)
            pos = (np.arange(g0, g0 + n)) % n_msg
            np.add.at(votes, (pos, bits), 1); g0 += n
    return (votes[:, 1] > votes[:, 0]).astype(np.uint8)


def main():
    np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (Xtr, ytr), (Xte, yte) = cifar(dev)
    base = SmallCNN().to(dev); train(base, Xtr, ytr, dev, epochs=6, lr=1e-3)
    P = sum(p.numel() for p in weight_tensors(base)); a0 = acc(base, Xte, yte, dev)
    print(f"baseline acc={a0:.3f}, embeddable weights={P:,}\n", flush=True)
    rng = np.random.default_rng(1)
    FT = [("light 2ep", 2, 2e-4), ("heavy 8ep", 8, 5e-4)]

    print(f"{'n_msg':>7} {'redund':>8} {'capacity':>10} " +
          " ".join(f"{lbl:>12}" for lbl, _, _ in FT), flush=True)
    for n_msg in [16, 64, 256, 1024, 4096, 16384]:
        msg = rng.integers(0, 2, n_msg, dtype=np.uint8); red = P // n_msg
        net = copy.deepcopy(base); embed_model(net, msg)
        recs = []
        for _, ep, lr in FT:
            netf = copy.deepcopy(net); train(netf, Xtr, ytr, dev, epochs=ep, lr=lr)
            recs.append((extract_model(netf, n_msg) == msg).mean())
        cap = f"{n_msg/8:.0f}B"
        print(f"{n_msg:>7} {red:>8} {cap:>10} " +
              " ".join(f"{r:>12.3f}" for r in recs), flush=True)
    print("\nbaca: redundansi tinggi (n_msg kecil) -> recovery 1.0 bahkan setelah fine-tune "
          "berat; trade-off kapasitas vs robustness = kurva ini.", flush=True)


if __name__ == "__main__":
    main()
