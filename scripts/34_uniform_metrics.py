"""Reviewer #3: a UNIFORM metric set for every transform condition, on real ResNet-56.

For each transform we report, at one fixed operating point (payload P_bytes, redundancy r),
all four levels the reviewer asked for, averaged over independent trials:
  - raw BER      : per-carrier code-LSB bit-error rate, BEFORE majority vote
  - post-vote BER: coded-bit error rate AFTER majority vote over the r carriers
  - post-ECC BER : message-bit error rate AFTER LT-fountain erasure decode
  - file OK      : fraction of trials whose recovered bytes SHA-256-match the original

Chain: payload bytes -> message bits -> LT encode (N coded bits) -> tile across the P
weights (qaware-stealthy, 1 bit/weight) -> victim transform -> read code LSBs -> majority
vote per coded bit -> LT decode -> bytes -> hash. Trials vary the payload and embed seed.

  python scripts/34_uniform_metrics.py --pbytes 256 --trials 3 --ft_epochs 3
"""
import sys, os, copy, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T
from wstego import robust, ecc

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465); CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def weight_tensors(model):
    return [p for _, p in model.named_parameters() if p.dim() >= 2]


def embed_coded(model, coded, seed):
    """Tile coded bits across all weights (qaware-stealthy per tensor). Returns a global
    carrier mask and the per-weight global coded-index (g % N) for later de-tiling."""
    N = coded.size; cur = 0
    carrier_parts, idx_parts = [], []
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            gidx = (np.arange(cur, cur + n)) % N
            bits = coded[gidx].astype(np.uint8)
            w2, _, car = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device, p.dtype))
            carrier_parts.append(car); idx_parts.append(gidx); cur += n
    return np.concatenate(carrier_parts), np.concatenate(idx_parts)


def read_raw(model):
    """Per-weight recovered code-LSB after whatever transform (per-tensor re-quant)."""
    out = []
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32).ravel()
            s = robust.quant_scale(w); out.append(robust.qaware_extract(w, s, nbits=1))
    return np.concatenate(out)


def prune_unstructured(model, ratio):
    with torch.no_grad():
        allw = np.concatenate([p.detach().cpu().numpy().ravel() for p in weight_tensors(model)])
        thr = np.quantile(np.abs(allw), ratio)
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy()
            p.copy_(torch.from_numpy(np.where(np.abs(w) <= thr, 0.0, w).astype(np.float32)).to(p.device, p.dtype))


def victim_int8(model):
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32); s = robust.quant_scale(w.ravel())
            p.copy_(torch.from_numpy((robust.quantize(w, s).astype(np.float32) * s)).to(p.device, p.dtype))


def cifar_loaders(root, bs=256):
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr = tv.datasets.CIFAR10(root, True, download=True, transform=tf)
    te = tv.datasets.CIFAR10(root, False, download=True, transform=tf)
    L = lambda d, sh: torch.utils.data.DataLoader(d, batch_size=bs, shuffle=sh, num_workers=2, pin_memory=True)
    return L(tr, True), L(te, False)


def finetune(model, loader, dev, epochs, lr=1e-3):
    model.train().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    return model


def metrics_for(model, carrier, gidx, coded, msg_bits, K, ecc_seed, N):
    """Compute the four-level metric tuple for a transformed, embedded model."""
    rec = read_raw(model)
    car = carrier
    # raw BER over carriers
    truth = coded[gidx].astype(np.uint8)
    raw_ber = float((rec[car] != truth[car]).mean())
    # majority vote per coded bit (carriers only)
    votes = np.zeros((N, 2), np.int64)
    np.add.at(votes, (gidx[car], rec[car]), 1)
    have = (votes.sum(1) > 0)
    voted = (votes[:, 1] > votes[:, 0]).astype(np.uint8)
    post_vote_ber = float((voted[have] != coded[have]).mean())
    # LT decode: erased = coded bits with no carrier or a tied vote
    tie = (votes[:, 0] == votes[:, 1])
    received = have & ~tie
    dec, ok = ecc.decode(voted, received, K, ecc_seed)
    post_ecc_ber = float((dec != msg_bits).mean())
    # file hash match
    def bits_to_bytes(b):
        b = b[:(b.size // 8) * 8]
        return np.packbits(b).tobytes()
    file_ok = (hashlib.sha256(bits_to_bytes(dec)).digest()
               == hashlib.sha256(bits_to_bytes(msg_bits)).digest())
    return raw_ber, post_vote_ber, post_ecc_ber, bool(file_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cifar10_resnet56")
    ap.add_argument("--data", default="data/cifar")
    ap.add_argument("--pbytes", type=int, default=256, help="payload size (bytes)")
    ap.add_argument("--overhead", type=float, default=0.25, help="LT coding overhead")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--ft_epochs", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = torch.hub.load("chenyaofo/pytorch-cifar-models", a.arch, pretrained=True).to(dev)
    trL, teL = cifar_loaders(a.data)
    P = sum(p.numel() for p in weight_tensors(base))
    K = a.pbytes * 8
    N = int(np.ceil(K * (1 + a.overhead)))
    r = P // N
    print(f"device={dev}  P={P:,}  payload={a.pbytes} B (K={K} bits)  "
          f"LT coded N={N}  redundancy r=P/N={r}x  trials={a.trials}\n", flush=True)

    conds = ["embed (clean)", "INT8", "prune 30%", "prune 50%", "prune 70%",
             f"finetune {a.ft_epochs}ep", "finetune+INT8"]
    acc = {c: [] for c in conds}

    for t in range(a.trials):
        rng = np.random.default_rng(1000 + t)
        msg = rng.integers(0, 2, K, dtype=np.uint8)
        coded, Nc = ecc.encode(msg, overhead=a.overhead, seed=7 + t); assert Nc == N
        emb = copy.deepcopy(base)
        carrier, gidx = embed_coded(emb, coded, seed=t)

        def do(m): return metrics_for(m, carrier, gidx, coded, msg, K, 7 + t, N)
        acc["embed (clean)"].append(do(emb))
        m = copy.deepcopy(emb); victim_int8(m); acc["INT8"].append(do(m))
        for ratio, name in ((0.3, "prune 30%"), (0.5, "prune 50%"), (0.7, "prune 70%")):
            m = copy.deepcopy(emb); prune_unstructured(m, ratio); acc[name].append(do(m))
        m = copy.deepcopy(emb); finetune(m, trL, dev, a.ft_epochs)
        acc[f"finetune {a.ft_epochs}ep"].append(do(m))
        victim_int8(m); acc["finetune+INT8"].append(do(m))
        print(f"  trial {t+1}/{a.trials} done", flush=True)

    print(f"\n{'transform':16s} {'raw BER':>9} {'post-vote':>10} {'post-ECC':>9} {'file OK':>9}", flush=True)
    for c in conds:
        A = np.array([[x[0], x[1], x[2], float(x[3])] for x in acc[c]])
        rb, pv, pe, fo = A.mean(0)
        print(f"{c:16s} {rb:>9.4f} {pv:>10.4f} {pe:>9.4f} {int(round(fo*a.trials))}/{a.trials:<d}", flush=True)
    print(f"\nbaca: metrik seragam 4-aras tiap transform, purata {a.trials} cubaan bebas, "
          f"pada SATU operating point (r={r}x). raw BER = sebelum undi; post-vote = selepas "
          f"undi majoriti; post-ECC = selepas LT; file OK = padanan SHA-256.", flush=True)


if __name__ == "__main__":
    main()
