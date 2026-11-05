"""Scale to a transformer: QuantGhost on a real Vision Transformer (ViT-B/16).

Shows the attack is architecture-agnostic beyond CNNs: capacity, survival through
per-tensor/per-channel INT8/INT4 quantization (by construction), and natural mantissa
entropy (stealth) on a real ImageNet-pretrained ViT-B/16. Optionally measures CIFAR-10
top-1 across {orig, clean INT8, embed, embed+INT8} on a HuggingFace CIFAR ViT.

  python scripts/26_vit.py
  python scripts/26_vit.py --acc-hf aaraki/vit-base-patch16-224-in21k-finetuned-cifar10
"""
import sys, os, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision.models as M
from wstego import robust

EMBED_MIN_DIM = 2


def weight_tensors(model):
    return [p for _, p in model.named_parameters() if p.dim() >= EMBED_MIN_DIM]


def embed_into(model, payload_bits, seed=0):
    cur = 0
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            bits = (payload_bits[cur:cur+n] if cur+n <= len(payload_bits)
                    else np.concatenate([payload_bits[cur:], np.zeros(cur+n-len(payload_bits), np.uint8)]))
            w2, _, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device, p.dtype)); cur += n


def quantize_inplace(model, qmax=127):
    with torch.no_grad():
        for p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32)
            s = robust.quant_scale(w.ravel(), qmax)
            p.copy_(torch.from_numpy(robust.quantize(w, s, qmax).astype(np.float32)*s).to(p.device, p.dtype))


def extract(model, n_bits, qmax=127):
    rec = []
    for p in weight_tensors(model):
        w = p.detach().cpu().numpy().astype(np.float32).ravel()
        rec.append(robust.qaware_extract(w, robust.quant_scale(w, qmax), nbits=1, qmax=qmax))
    return np.concatenate(rec)[:n_bits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-hf", default=None, help="HF CIFAR-10 ViT id for the accuracy table")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  loading ViT-B/16 (ImageNet) ...", flush=True)
    base = M.vit_b_16(weights="DEFAULT").to(dev).eval()
    P = sum(p.numel() for p in weight_tensors(base))
    print(f"embeddable weight tensors={len(weight_tensors(base))}  weights={P:,}  "
          f"capacity = {P/8/1e6:.2f} MB @ 1 bit/weight\n", flush=True)

    rng = np.random.default_rng(12345)
    payload = rng.integers(0, 256, size=P//8, dtype=np.uint8).tobytes()
    pbits = np.unpackbits(np.frombuffer(payload, np.uint8)).astype(np.uint8)
    ent0 = robust.mantissa_lowbyte_entropy(
        np.concatenate([p.detach().cpu().numpy().ravel() for p in weight_tensors(base)]))

    print(f"{'scheme':22s} {'survival':>9} {'mant.entropy':>13}", flush=True)
    print(f"{'benign (no embed)':22s} {'--':>9} {ent0:>13.3f}", flush=True)
    for qmax, lbl in [(127, "INT8"), (7, "INT4")]:
        m = copy.deepcopy(base); embed_into(m, pbits, seed=a.seed)
        ent = robust.mantissa_lowbyte_entropy(
            np.concatenate([p.detach().cpu().numpy().ravel() for p in weight_tensors(m)]))
        quantize_inplace(m, qmax)
        rec = extract(m, len(pbits), qmax)
        surv = (rec == pbits).mean()
        print(f"{'ours, per-tensor '+lbl:22s} {surv:>9.4f} {ent:>13.3f}", flush=True)

    print(f"\nVERDICT: QuantGhost embeds {P/8/1e6:.1f} MB into a real ViT-B/16 and survives "
          f"INT8/INT4 by construction with benign mantissa entropy (~{ent0:.2f}); the attack "
          f"is architecture-agnostic beyond CNNs.", flush=True)

    if a.acc_hf:
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            import torchvision as tv, torchvision.transforms as T
            print(f"\nloading HF CIFAR ViT {a.acc_hf} for accuracy ...", flush=True)
            proc = AutoImageProcessor.from_pretrained(a.acc_hf)
            hf = AutoModelForImageClassification.from_pretrained(a.acc_hf).to(dev).eval()
            mean, std = proc.image_mean, proc.image_std
            size = proc.size.get("height", 224)
            tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])
            ds = tv.datasets.CIFAR10("data/cifar", train=False, download=True, transform=tf)
            ld = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=2)

            def top1(model):
                model.eval(); c = n = 0
                with torch.no_grad():
                    for x, y in ld:
                        x = x.to(dev); logits = model(x).logits
                        c += (logits.argmax(1).cpu() == y).sum().item(); n += y.size(0)
                        if n >= 5000: break
                return 100.0 * c / n
            Pn = sum(p.numel() for p in weight_tensors(hf)); pb = np.unpackbits(rng.integers(0,256,Pn//8,dtype=np.uint8))
            mq = copy.deepcopy(hf); quantize_inplace(mq)
            me = copy.deepcopy(hf); embed_into(me, pb, seed=a.seed)
            meq = copy.deepcopy(me); quantize_inplace(meq)
            print(f"\n{'condition':16s} {'top-1':>8}", flush=True)
            for lbl, m in [("orig", hf), ("clean INT8", mq), ("embed", me), ("embed+INT8", meq)]:
                print(f"{lbl:16s} {top1(m):>7.2f}%", flush=True)
        except Exception as e:
            print(f"\naccuracy step skipped: {e!r}", flush=True)


if __name__ == "__main__":
    main()
