"""B1 (real-model accuracy, no manual download): embed a payload into a REAL
pretrained CIFAR-10 ResNet, then measure top-1 accuracy across four conditions to
separate the cost of embedding from the cost of quantization:

    orig FP32 | clean INT8 | embed FP32 | embed INT8 (victim)

Unlike 14_mb_payload.py (ResNet-50, needs ImageNet-val for accuracy), this uses a
pretrained CIFAR-10 model from torch.hub (chenyaofo/pytorch-cifar-models) and the
CIFAR-10 test set, both auto-downloaded -- so the "accuracy preserved" number is
fully reproducible with zero manual data setup. Same qaware-stealthy 1-bit/weight
embedding + per-tensor INT8 victim quantization as the paper's main method.

  python scripts/20_cifar_accuracy.py                 # default cifar10 resnet56
  python scripts/20_cifar_accuracy.py --arch cifar10_resnet20
"""
import sys, os, hashlib, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision as tv
import torchvision.transforms as T
from wstego import robust

EMBED_MIN_DIM = 2          # embed into / quantize conv+fc weight tensors (dim>=2)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def weight_tensors(model):
    return [(n, p) for n, p in model.named_parameters() if p.dim() >= EMBED_MIN_DIM]


def embed_into_model(model, payload_bits, seed=0):
    """Embed payload sequentially across weight tensors (per-tensor qaware-stealthy),
    writing FP32 weights back in place. Same protocol as 14_mb_payload.py."""
    cur = 0
    with torch.no_grad():
        for _, p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32).ravel()
            n = w.size
            if cur + n <= len(payload_bits):
                bits = payload_bits[cur:cur + n]
            else:
                bits = np.concatenate([payload_bits[cur:],
                                       np.zeros(cur + n - len(payload_bits), np.uint8)])
            w2, _, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device, p.dtype))
            cur += n


def quantize_model_inplace(model):
    """Victim per-tensor symmetric INT8 PTQ on the weight tensors."""
    with torch.no_grad():
        for _, p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32)
            s = robust.quant_scale(w.ravel())
            wq = robust.quantize(w, s).astype(np.float32) * s
            p.copy_(torch.from_numpy(wq).to(p.device, p.dtype))


def extract_from_model(model, total_bits):
    rec = []
    for _, p in weight_tensors(model):
        w = p.detach().cpu().numpy().astype(np.float32).ravel()
        s = robust.quant_scale(w)
        rec.append(robust.qaware_extract(w, s, nbits=1))
    return np.concatenate(rec)[:total_bits]


@torch.no_grad()
def top1(model, loader, dev):
    model.eval().to(dev)
    correct = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        correct += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return correct / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cifar10_resnet56",
                    help="torch.hub chenyaofo/pytorch-cifar-models arch "
                         "(cifar10_resnet20/32/44/56, cifar10_vgg16_bn, ...)")
    ap.add_argument("--data", default="data/cifar")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  arch={a.arch}", flush=True)

    base = torch.hub.load("chenyaofo/pytorch-cifar-models", a.arch, pretrained=True)
    cap_bits = sum(p.numel() for _, p in weight_tensors(base))
    print(f"weight tensors={len(weight_tensors(base))}, "
          f"capacity = {cap_bits:,} bits = {cap_bits/8/1e3:.1f} KB\n", flush=True)

    # CIFAR-10 test set (auto-download)
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = tv.datasets.CIFAR10(a.data, train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                                         num_workers=a.workers, pin_memory=True)

    # payload (encrypted-malware stand-in: uniform random bytes filling capacity)
    n_bytes = cap_bits // 8
    rng = np.random.default_rng(12345)
    payload = rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()
    payload_bits = np.unpackbits(np.frombuffer(payload, np.uint8)).astype(np.uint8)
    print(f"payload {len(payload)/1e3:.1f} KB  sha256={hashlib.sha256(payload).hexdigest()[:16]}...\n",
          flush=True)

    model_q = copy.deepcopy(base); quantize_model_inplace(model_q)        # clean INT8
    model_e = copy.deepcopy(base); embed_into_model(model_e, payload_bits, seed=a.seed)  # embed FP32
    model_eq = copy.deepcopy(model_e); quantize_model_inplace(model_eq)   # embed + INT8

    rec_bits = extract_from_model(model_eq, len(payload_bits))
    bit_acc = (rec_bits == payload_bits).mean()
    print(f"bit-recovery after INT8 = {bit_acc:.6f}\n", flush=True)

    print(f"{'condition':14s} {'top-1':>8} {'n':>7}", flush=True)
    accs = {}
    for lbl, m in [("orig FP32", base), ("clean INT8", model_q),
                   ("embed FP32", model_e), ("embed INT8", model_eq)]:
        acc, n = top1(m, loader, dev)
        accs[lbl] = acc
        print(f"{lbl:14s} {acc*100:>7.2f}% {n:>7}", flush=True)

    d_quant = (accs["orig FP32"] - accs["clean INT8"]) * 100
    d_embed = (accs["clean INT8"] - accs["embed INT8"]) * 100
    d_total = (accs["orig FP32"] - accs["embed INT8"]) * 100
    print(f"\ndelta: quant-only={d_quant:+.2f}pp  embedding-added={d_embed:+.2f}pp  "
          f"total(orig->embed+INT8)={d_total:+.2f}pp", flush=True)
    print(f"\nVERDICT: {cap_bits/8/1e3:.0f} KB payload @ {bit_acc*100:.4f}% bit-recovery; "
          f"accuracy change orig->embed+INT8 = {d_total:+.2f}pp on a real CIFAR-10 {a.arch}.",
          flush=True)


if __name__ == "__main__":
    main()
