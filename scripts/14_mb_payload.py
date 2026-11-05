"""Scale experiment #2: embed a real MULTI-MB payload in a real model (ResNet-50),
verify recovery after INT8 quantization, AND measure ImageNet top-1/top-5 accuracy.

Sequential 1-bit/weight embedding (frameshift-free): per tensor, weight i carries
payload bit i. Pinned scale-defining weights (~1/tensor) carry a forced bit -> a
negligible error rate that the redundancy/ECC layer (see 09) removes for
byte-exact delivery.

Accuracy (B1): we now write the embedded weights back into the model and report
top-1/top-5 on ImageNet-val for four conditions to separate the cost of embedding
from the cost of quantization:
  orig FP32 | clean INT8 | embed FP32 | embed INT8 (victim).
Provide data via --val PATH or the IMAGENET_VAL env var. Without it, the accuracy
section is skipped and the bit-recovery part still runs.
"""
import sys, os, hashlib, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision.models as M
from wstego import robust, imagenet

EMBED_MIN_DIM = 2          # embed into / quantize weight tensors with dim>=2 (conv+fc)


def weight_tensors(model):
    """Named parameters we embed into and the victim quantizes (conv/linear weights)."""
    return [(n, p) for n, p in model.named_parameters() if p.dim() >= EMBED_MIN_DIM]


def embed_into_model(model, payload_bits, seed=0):
    """Embed the payload sequentially across all weight tensors, writing FP32
    qaware-stealthy weights back into the model in place."""
    cur = 0
    with torch.no_grad():
        for _, p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32).ravel()
            n = w.size
            if cur + n <= len(payload_bits):
                bits = payload_bits[cur:cur + n]
            else:                                   # zero-pad the tail of the last tensors
                bits = np.concatenate([payload_bits[cur:],
                                       np.zeros(cur + n - len(payload_bits), np.uint8)])
            w2, _, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device, p.dtype))
            cur += n


def quantize_model_inplace(model):
    """Simulate the victim's per-tensor symmetric INT8 PTQ on the weight tensors."""
    with torch.no_grad():
        for _, p in weight_tensors(model):
            w = p.detach().cpu().numpy().astype(np.float32)
            s = robust.quant_scale(w.ravel())
            wq = robust.quantize(w, s).astype(np.float32) * s
            p.copy_(torch.from_numpy(wq).to(p.device, p.dtype))


def extract_from_model(model, total_bits):
    """Recover payload bits from the (quantized) model weights, in embed order."""
    rec = []
    for _, p in weight_tensors(model):
        w = p.detach().cpu().numpy().astype(np.float32).ravel()
        s = robust.quant_scale(w)
        rec.append(robust.qaware_extract(w, s, nbits=1))
    return np.concatenate(rec)[:total_bits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=None, help="ImageNet val dir (else IMAGENET_VAL env)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="limit batches for a quick accuracy estimate (default: full val)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_accuracy", action="store_true", help="skip ImageNet eval")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    print("loading ResNet-50...", flush=True)
    base = M.resnet50(weights="DEFAULT")

    cap_bits = sum(p.numel() for _, p in weight_tensors(base))
    print(f"weight tensors={len(weight_tensors(base))}, "
          f"capacity = {cap_bits:,} bits = {cap_bits/8/1e6:.2f} MB\n", flush=True)

    # --- build payload (encrypted-malware stand-in: uniform random bytes) ---
    n_bytes = cap_bits // 8
    rng = np.random.default_rng(12345)
    payload = rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()
    payload_bits = np.unpackbits(np.frombuffer(payload, np.uint8)).astype(np.uint8)
    sha_in = hashlib.sha256(payload).hexdigest()
    print(f"payload {len(payload)/1e6:.2f} MB  sha256={sha_in[:16]}...\n", flush=True)

    # --- four model variants ---
    model_q = copy.deepcopy(base); quantize_model_inplace(model_q)        # clean INT8
    model_e = copy.deepcopy(base); embed_into_model(model_e, payload_bits, seed=a.seed)  # embed FP32
    model_eq = copy.deepcopy(model_e); quantize_model_inplace(model_eq)   # embed + victim INT8

    # --- bit recovery from the victim-quantized embedded model ---
    rec_bits = extract_from_model(model_eq, len(payload_bits))
    bit_acc = (rec_bits == payload_bits).mean()
    nbytes_ok = len(rec_bits) // 8 * 8
    rec_bytes = np.packbits(rec_bits[:nbytes_ok]).tobytes()
    byte_err = (np.frombuffer(rec_bytes, np.uint8)
                != np.frombuffer(payload[:len(rec_bytes)], np.uint8)).mean()
    print(f"embedded {cap_bits/8/1e6:.2f} MB | bit-recovery after INT8 = {bit_acc:.6f}", flush=True)
    print(f"byte-error rate = {byte_err:.6f}  (residual = pinned scale weights; ECC layer -> byte-exact)\n",
          flush=True)

    # --- accuracy ---
    if a.no_accuracy:
        print("accuracy: skipped (--no_accuracy)", flush=True)
    else:
        loader = imagenet.val_loader(a.val, batch_size=a.batch_size, workers=a.workers)
        if loader is None:
            d = a.val or os.environ.get("IMAGENET_VAL")
            print("accuracy: SKIPPED (no ImageNet val dir).", flush=True)
            print(f"  set --val PATH or IMAGENET_VAL env (looked at: {d!r}).", flush=True)
            print("  dir must be ImageFolder layout: one synset subfolder per class.", flush=True)
        else:
            scope = "full val" if a.max_batches is None else f"{a.max_batches} batches"
            print(f"evaluating ImageNet-val top-1/top-5 ({scope})...\n", flush=True)
            print(f"{'condition':14s} {'top-1':>8} {'top-5':>8} {'n':>8}", flush=True)
            for lbl, m in [("orig FP32", base), ("clean INT8", model_q),
                           ("embed FP32", model_e), ("embed INT8", model_eq)]:
                t1, t5, n = imagenet.top1(m, loader, dev, max_batches=a.max_batches)
                print(f"{lbl:14s} {t1*100:>7.3f}% {t5*100:>7.3f}% {n:>8}", flush=True)
            print("\nbaca: (orig->clean INT8) = biaya kuantisasi murni; "
                  "(clean INT8->embed INT8) = biaya tambahan embedding.", flush=True)

    print(f"\nVERDICT: {cap_bits/8/1e6:.1f} MB payload survives FP32->INT8 quantization at "
          f"{bit_acc*100:.4f}% bit fidelity on a real ResNet-50.", flush=True)


if __name__ == "__main__":
    main()
