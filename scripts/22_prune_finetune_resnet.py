"""B4/B5 at REAL scale: payload survival through PRUNING and FINE-TUNING on a real
pretrained CIFAR-10 ResNet-56, reporting payload recovery AND model top-1 together.

Unlike 09/16 (toy SmallCNN, acc ~0.47), this uses a real model so the survival-vs-
accuracy trade-off is meaningful: a transform only "defends" if it destroys the
payload without destroying the model. Redundancy-coded 1-bit/weight embedding
(qaware-stealthy) + majority-vote decode; redundancy = (#weights)/n_msg.

  python scripts/22_prune_finetune_resnet.py
  python scripts/22_prune_finetune_resnet.py --arch cifar10_resnet20 --ft_epochs 2
"""
import sys, os, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T
from wstego import robust

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def weight_tensors(model):
    return [p for _, p in model.named_parameters() if p.dim() >= 2]


def embed_redundant(model, msg, seed=0):
    """Tile msg across all embeddable weights (qaware-stealthy), write FP32 back."""
    cur = 0
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            bits = msg[(np.arange(cur, cur + n)) % len(msg)].astype(np.uint8)
            w2, _, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device, p.dtype)); cur += n


def extract_redundant(model, n_msg):
    """Majority-vote decode over all redundant copies of each message bit."""
    votes = np.zeros((n_msg, 2), np.int64); cur = 0
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            s = robust.quant_scale(w); bits = robust.qaware_extract(w, s, nbits=1)
            np.add.at(votes, ((np.arange(cur, cur + n)) % n_msg, bits), 1); cur += n
    return (votes[:, 1] > votes[:, 0]).astype(np.uint8)


def prune_unstructured(model, ratio):
    """Global magnitude pruning: zero the smallest-|w| `ratio` fraction of weights."""
    with torch.no_grad():
        allw = np.concatenate([p.detach().cpu().numpy().ravel() for p in weight_tensors(model)])
        thr = np.quantile(np.abs(allw), ratio)
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy()
            p.copy_(torch.from_numpy(np.where(np.abs(w) <= thr, 0.0, w).astype(np.float32)).to(p.device, p.dtype))


def prune_structured(model, ratio):
    """Structured: zero the smallest-L2 `ratio` fraction of output channels per conv tensor."""
    with torch.no_grad():
        for p in weight_tensors(model):
            if p.dim() != 4:
                continue
            w = p.detach().cpu().numpy()
            l2 = np.sqrt((w.reshape(w.shape[0], -1) ** 2).sum(1))
            k = int(ratio * w.shape[0])
            if k > 0:
                drop = np.argsort(l2)[:k]; w[drop] = 0.0
                p.copy_(torch.from_numpy(w.astype(np.float32)).to(p.device, p.dtype))


def cifar_loaders(root, bs=256, workers=2):
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr = tv.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    te = tv.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    L = lambda d, sh: torch.utils.data.DataLoader(d, batch_size=bs, shuffle=sh, num_workers=workers, pin_memory=True)
    return L(tr, True), L(te, False)


@torch.no_grad()
def top1(model, loader, dev):
    model.eval().to(dev); c = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
    return c / n


def finetune(model, loader, dev, epochs, lr=1e-3):
    model.train().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cifar10_resnet56")
    ap.add_argument("--data", default="data/cifar")
    ap.add_argument("--ft_epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} arch={a.arch}", flush=True)

    base = torch.hub.load("chenyaofo/pytorch-cifar-models", a.arch, pretrained=True).to(dev)
    P = sum(p.numel() for p in weight_tensors(base))
    trL, teL = cifar_loaders(a.data)
    acc0 = top1(base, teL, dev)
    print(f"embeddable weights={P:,}  base top-1={acc0*100:.2f}%\n", flush=True)

    rng = np.random.default_rng(1)

    # ---- PRUNING sweep (one redundancy) : recovery + accuracy ----
    n_msg = 256; msg = rng.integers(0, 2, n_msg, dtype=np.uint8)
    redund = P // n_msg
    emb = copy.deepcopy(base); embed_redundant(emb, msg, seed=a.seed)
    rec0 = (extract_redundant(emb, n_msg) == msg).mean()
    print(f"PRUNING (n_msg={n_msg}, redundancy={redund}x)", flush=True)
    print(f"{'transform':22s} {'recovery':>9} {'top-1':>8}", flush=True)
    print(f"{'(embed, no prune)':22s} {rec0:>9.3f} {top1(emb, teL, dev)*100:>7.2f}%", flush=True)
    for ratio in (0.3, 0.5, 0.7, 0.9):
        for kind, fn in (("unstruct", prune_unstructured), ("struct", prune_structured)):
            m = copy.deepcopy(emb); fn(m, ratio)
            rec = (extract_redundant(m, n_msg) == msg).mean()
            print(f"{'prune '+kind+f' {int(ratio*100)}%':22s} {rec:>9.3f} {top1(m, teL, dev)*100:>7.2f}%", flush=True)

    # ---- FINE-TUNING (two redundancy levels) : recovery + accuracy ----
    print(f"\nFINE-TUNING ({a.ft_epochs} epochs, real CIFAR-10 train)", flush=True)
    print(f"{'n_msg':>7} {'redund':>8} {'recovery':>9} {'top-1':>8}", flush=True)
    for n_msg in (256, 4096):
        msg = rng.integers(0, 2, n_msg, dtype=np.uint8)
        m = copy.deepcopy(base); embed_redundant(m, msg, seed=a.seed)
        m = finetune(m, trL, dev, a.ft_epochs)
        rec = (extract_redundant(m, n_msg) == msg).mean()
        print(f"{n_msg:>7} {P//n_msg:>7}x {rec:>9.3f} {top1(m, teL, dev)*100:>7.2f}%", flush=True)

    print(f"\nbaca: pruning unstruct selamat ~30% lalu meluruh; structured lebih cepat mati. "
          f"fine-tune = transform tersulit -> redundansi tinggi (n_msg kecil) menang. "
          f"recovery dilaporkan berdampingan dgn top-1 di model NYATA.", flush=True)


if __name__ == "__main__":
    main()
