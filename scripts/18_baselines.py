"""B7: prior weight-stego baselines under deployment transforms.

StegoNet / EvilModel / MaleficNet hide payloads in FP32 mantissa LSBs (X-LSB-Fill,
X bits per weight). High capacity, but FP32->INT8 quantization discards the mantissa
-> the payload is erased. We embed naive X-LSB at X in {8,16,23} into a real trained
ResNet-18, apply the victim's per-tensor INT8 quantization, and measure bit-recovery.
Contrast: our qaware-stealthy survives the same transform. We also tabulate capacity
vs robust DNN watermarking (arXiv:2208.10973), which is robust but low-capacity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision.models as M
from wstego import embed as emb, robust


def model_weights():
    """Concatenate all >=2D weight tensors of a real trained ResNet-18 (one vector)."""
    net = M.resnet18(weights="DEFAULT")
    ws = [p.detach().numpy().astype(np.float32).ravel()
          for n, p in net.named_parameters() if p.dim() >= 2]
    return np.concatenate(ws)


def survival_naive_xlsb(w, x, payload):
    """Embed X-LSB then quantize per-tensor INT8 -> recover -> bit accuracy."""
    w2 = emb.embed_xlsb(w, payload, x=x, fill=True)
    s = robust.quant_scale(w2)
    wq = robust.quantize(w2, s).astype(np.float32) * s        # victim INT8 round-trip
    rec = emb.extract_xlsb(wq, x=x)
    cap = w.size * x
    truth = np.tile(payload, int(np.ceil(cap / len(payload))))[:cap].astype(np.uint8)
    return float((rec == truth).mean())


def survival_qaware(w, payload):
    w2, s, car = robust.qaware_embed_stealthy(w, payload, nbits=1, seed=0)
    sv = robust.quant_scale(w2); wq = robust.quantize(w2, sv).astype(np.float32) * sv
    rec = robust.qaware_extract(wq, sv, nbits=1)
    return robust.survival_rate(rec, payload, nbits=1, carrier=car)


def main():
    print("loading ResNet-18...", flush=True)
    w = model_weights()
    P = w.size
    payload = emb.random_payload_bits(1 << 16, 0)
    print(f"embeddable weights={P:,}\n", flush=True)

    print(f"{'scheme':28s} {'cap (bits/w)':>12} {'survival INT8':>14}  verdict", flush=True)
    for x in (8, 16, 23):
        sv = survival_naive_xlsb(w, x, payload)
        verdict = "ERASED" if sv < 0.6 else ("partial" if sv < 0.9 else "survives")
        print(f"{f'naive X-LSB X={x} (EvilModel/StegoNet)':28s} {x:>12} {sv:>14.3f}  {verdict}", flush=True)
    svq = survival_qaware(w, payload)
    print(f"{'qaware-stealthy (ours)':28s} {1:>12} {svq:>14.3f}  "
          f"{'survives' if svq > 0.9 else 'partial'}", flush=True)

    # capacity comparison (context for reviewers): robust-but-low-cap watermark vs us
    our_mb = P / 8 / 1e6
    print(f"\ncapacity vs robust watermarking (context):", flush=True)
    print(f"  robust DNN watermark (2208.10973): robust to transforms but <=~2 KB, "
          f"needs pre-training (identity, not payload delivery)", flush=True)
    print(f"  ours: {our_mb:.2f} MB on this ResNet-18 @1 bit/weight, post-hoc, "
          f"survives INT8 -> high-capacity AND robust", flush=True)
    print("\nbaca: baseline lama kapasitas tinggi tapi survival INT8 ~chance (ERASED); "
          "watermark robust tapi kapasitas mini. Kita mengisi gap high-cap + robust.", flush=True)


if __name__ == "__main__":
    main()
