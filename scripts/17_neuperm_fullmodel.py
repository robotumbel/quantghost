"""B6: full-model faithful NeuPerm. Confirms (1) the permutation preserves the
network function (max|Δlogit|~0, accuracy identical), and (2) which payloads
survive it across the WHOLE model:
  - flat/positional payload (global weight order)  -> DIES (~chance)
  - spatial-invariant payload (indexed by kh,kw)   -> SURVIVES (~1.0)
Aggregate spatial-invariant capacity = sum of kh*kw over conv tensors (each slot
replicated across all in/out channels = built-in redundancy).
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from wstego import robust, neuperm
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


def train(net, Xtr, ytr, dev, epochs=6, lr=1e-3, bs=512, sub=10000):
    sel = torch.randperm(len(Xtr), device=dev)[:sub]; Xtr, ytr = Xtr[sel], ytr[sel]
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    for _ in range(epochs):
        p = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), bs):
            idx = p[i:i + bs]; opt.zero_grad()
            F.cross_entropy(net(Xtr[idx].float() / 255.), ytr[idx]).backward(); opt.step()


def conv_weights(net):
    return [p for p in net.parameters() if p.dim() == 4]


# ---- flat/positional payload (global weight order) ----
def embed_flat(net, msg, seed=0):
    g0 = 0
    with torch.no_grad():
        for p in conv_weights(net):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            bits = msg[(np.arange(g0, g0 + n)) % len(msg)].astype(np.uint8)
            w2, _, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed + g0)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device)); g0 += n


def extract_flat(net, n_msg):
    votes = np.zeros((n_msg, 2), np.int64); g0 = 0
    with torch.no_grad():
        for p in conv_weights(net):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            s = robust.quant_scale(w); bits = robust.qaware_extract(w, s, nbits=1)
            np.add.at(votes, ((np.arange(g0, g0 + n)) % n_msg, bits), 1); g0 += n
    return (votes[:, 1] > votes[:, 0]).astype(np.uint8)


# ---- spatial-invariant payload (indexed by kh,kw; each tensor its own chunk) ----
def embed_spatial(net, rng, seed=0):
    msgs = []
    with torch.no_grad():
        for p in conv_weights(net):
            O, I, kh, kw = p.shape; S = kh * kw
            m = rng.integers(0, 2, S, dtype=np.uint8); msgs.append(m)
            w = p.detach().cpu().numpy().astype(np.float32)
            spatial_idx = (np.arange(O * I * S) % S)
            bits = m[spatial_idx]
            w2, _, _ = robust.qaware_embed_stealthy(w.ravel(), bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device))
    return msgs


def extract_spatial(net):
    out = []
    with torch.no_grad():
        for p in conv_weights(net):
            O, I, kh, kw = p.shape; S = kh * kw
            w = p.detach().cpu().numpy().astype(np.float32).ravel()
            s = robust.quant_scale(w); bits = robust.qaware_extract(w, s, nbits=1)
            votes = np.zeros((S, 2), np.int64)
            np.add.at(votes, (np.arange(O * I * S) % S, bits), 1)
            out.append((votes[:, 1] > votes[:, 0]).astype(np.uint8))
    return out


def main():
    np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (Xtr, ytr), (Xte, yte) = cifar(dev)
    base = SmallCNN().to(dev); train(base, Xtr, ytr, dev)
    a0 = acc(base, Xte, yte, dev)

    # (1) faithful permutation preserves the function
    permd = neuperm.permute_smallcnn(copy.deepcopy(base), seed=7)
    xb = Xte[:256].float() / 255.
    d = neuperm.max_logit_diff(base.eval(), permd.eval(), xb)
    ap = acc(permd, Xte, yte, dev)
    print(f"baseline acc={a0:.3f} | after NeuPerm acc={ap:.3f} | max|d_logit|={d:.2e} "
          f"(=0 => function-preserving)\n", flush=True)

    cap = sum(p.shape[2] * p.shape[3] for p in conv_weights(base))
    print(f"conv tensors={len(conv_weights(base))}  spatial-invariant capacity "
          f"= {cap} bits (sum kh*kw)\n", flush=True)

    # (2) flat payload: survives nothing-changed, dies after permutation
    rng = np.random.default_rng(1); msg = rng.integers(0, 2, 256, dtype=np.uint8)
    nf = copy.deepcopy(base); embed_flat(nf, msg)
    s0 = (extract_flat(nf, 256) == msg).mean()
    nfp = neuperm.permute_smallcnn(copy.deepcopy(nf), seed=7)
    s1 = (extract_flat(nfp, 256) == msg).mean()

    # (3) spatial-invariant payload: survives the permutation
    rng2 = np.random.default_rng(2)
    ns = copy.deepcopy(base); msgs = embed_spatial(ns, rng2)
    rec0 = extract_spatial(ns); t0 = np.mean([ (a == b).mean() for a, b in zip(rec0, msgs) ])
    nsp = neuperm.permute_smallcnn(copy.deepcopy(ns), seed=7)
    rec1 = extract_spatial(nsp); t1 = np.mean([ (a == b).mean() for a, b in zip(rec1, msgs) ])

    print(f"{'scheme':18s} {'no-perm':>9} {'after-PERM':>11}", flush=True)
    print(f"{'flat-positional':18s} {s0:>9.3f} {s1:>11.3f}", flush=True)
    print(f"{'spatial-invariant':18s} {t0:>9.3f} {t1:>11.3f}", flush=True)
    print("\nharapan: flat hancur (~0.5) after-PERM; spatial-invariant bertahan (~1.0). "
          f"kapasitas invarian agregat = {cap} bit (toy; skala penuh = sum kh*kw model nyata).", flush=True)


if __name__ == "__main__":
    main()
