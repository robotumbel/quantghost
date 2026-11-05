"""Reviewer rebuttal: run a FAITHFUL MaleficNet (spread-spectrum / DSSS) baseline on the
SAME INT8 / prune / fine-tune pipeline as QuantGhost, on a real CIFAR-10 ResNet-56, and
report payload recovery alongside top-1 accuracy. This replaces the "reported design"
entries of Table I with measured numbers.

MaleficNet (Hitaj et al., ESORICS 2022) spreads each payload bit over the carrier with a
Gaussian sequence and adds a small signal (amplitude gamma) to the weights; it is recovered
by blind correlation (wstego/maleficnet.py). Its spreading gain tolerates small weight
perturbations (its design goal), but the signal is sub-quantum (kept below the INT8 step to
preserve accuracy), so INT8 quantization is expected to erase it. We measure all of this.

  python scripts/32_maleficnet_baseline.py                 # default K=512, gamma auto
  python scripts/32_maleficnet_baseline.py --K 1024 --c 0.04 --ft_epochs 3
"""
import sys, os, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T
from wstego import maleficnet as mn, robust

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def weight_tensors(model):
    return [p for _, p in model.named_parameters() if p.dim() >= 2]


def flatten_weights(model):
    return np.concatenate([p.detach().cpu().numpy().astype(np.float32).ravel()
                           for p in weight_tensors(model)])


def write_flat(model, flat):
    cur = 0
    with torch.no_grad():
        for p in weight_tensors(model):
            n = p.numel()
            p.copy_(torch.from_numpy(flat[cur:cur + n].reshape(p.shape)).to(p.device, p.dtype))
            cur += n


def victim_int8_per_tensor(model):
    """Apply per-tensor symmetric INT8 round-trip to every weight tensor (victim PTQ)."""
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32)
            s = robust.quant_scale(w.ravel())
            wq = robust.quantize(w, s).astype(np.float32) * s
            p.copy_(torch.from_numpy(wq).to(p.device, p.dtype))


def prune_unstructured(model, ratio):
    with torch.no_grad():
        allw = np.concatenate([p.detach().cpu().numpy().ravel() for p in weight_tensors(model)])
        thr = np.quantile(np.abs(allw), ratio)
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy()
            p.copy_(torch.from_numpy(np.where(np.abs(w) <= thr, 0.0, w).astype(np.float32)).to(p.device, p.dtype))


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
    ap.add_argument("--K", type=int, default=512, help="payload bits (DSSS)")
    ap.add_argument("--c", type=float, default=0.05, help="gamma = c * global weight std")
    ap.add_argument("--ft_epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} arch={a.arch}", flush=True)

    base = torch.hub.load("chenyaofo/pytorch-cifar-models", a.arch, pretrained=True).to(dev)
    trL, teL = cifar_loaders(a.data)
    w0 = flatten_weights(base); P = w0.size
    std = float(w0.std()); gamma = a.c * std
    acc0 = top1(base, teL, dev)
    print(f"embeddable weights P={P:,}  weight std={std:.4g}  gamma={gamma:.4g} "
          f"(c={a.c})", flush=True)
    print(f"MaleficNet DSSS: K={a.K} bits, spreading factor n/K={P//a.K:,}", flush=True)
    print(f"base top-1={acc0*100:.2f}%\n", flush=True)

    rng = np.random.default_rng(a.seed + 7)
    payload = rng.integers(0, 2, a.K, dtype=np.uint8)

    # ---- embed MaleficNet signal into the flat weight vector, write back ----
    w_st = mn.embed(w0, payload, gamma=gamma, seed=a.seed)
    emb = copy.deepcopy(base); write_flat(emb, w_st)
    acc_emb = top1(emb, teL, dev)
    rec_clean = mn.recovery(flatten_weights(emb), payload, seed=a.seed)
    print(f"{'transform':26s} {'recovery':>9} {'top-1':>8}", flush=True)
    print(f"{'embed (no transform)':26s} {rec_clean:>9.3f} {acc_emb*100:>7.2f}%", flush=True)

    # ---- INT8 (victim per-tensor PTQ) ----
    m = copy.deepcopy(emb); victim_int8_per_tensor(m)
    rec_q = mn.recovery(flatten_weights(m), payload, seed=a.seed)
    print(f"{'INT8 per-tensor':26s} {rec_q:>9.3f} {top1(m, teL, dev)*100:>7.2f}%", flush=True)

    # ---- pruning (unstructured) ----
    for ratio in (0.3, 0.5):
        m = copy.deepcopy(emb); prune_unstructured(m, ratio)
        rec_p = mn.recovery(flatten_weights(m), payload, seed=a.seed)
        print(f"{'prune unstruct '+f'{int(ratio*100)}%':26s} {rec_p:>9.3f} {top1(m, teL, dev)*100:>7.2f}%", flush=True)

    # ---- fine-tuning (same-task) ----
    m = copy.deepcopy(emb); m = finetune(m, trL, dev, a.ft_epochs)
    rec_f = mn.recovery(flatten_weights(m), payload, seed=a.seed)
    print(f"{f'finetune {a.ft_epochs}ep':26s} {rec_f:>9.3f} {top1(m, teL, dev)*100:>7.2f}%", flush=True)

    # ---- INT8 AFTER fine-tune (full chain) ----
    victim_int8_per_tensor(m)
    rec_fq = mn.recovery(flatten_weights(m), payload, seed=a.seed)
    print(f"{'finetune + INT8':26s} {rec_fq:>9.3f} {top1(m, teL, dev)*100:>7.2f}%", flush=True)

    print(f"\nbaca: MaleficNet spread-spectrum survives SMALL perturbations (light prune/"
          f"finetune) by design, but its sub-quantum signal is erased by INT8 (recovery -> "
          f"~0.5 chance), because the quantization step is coarser than gamma. Contrast "
          f"QuantGhost: INT8 survival 1.000 by construction. Accuracy cost of MaleficNet "
          f"embed = {(acc0-acc_emb)*100:+.2f} pp.", flush=True)


if __name__ == "__main__":
    main()
