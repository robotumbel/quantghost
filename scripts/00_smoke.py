"""Fast correctness smoke test (no downloads). Validates the core pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from wstego import embed, gf_image, detector

print("=== 1. embedding round-trip + perturbation ===")
rng = np.random.default_rng(0)
# emulate trained weights: small-magnitude floats with full mantissa entropy
w = (rng.standard_normal(200_000).astype(np.float32) * 0.05)
X = 8
payload = embed.random_payload_bits(10_000, seed=1)
w_st = embed.embed_xlsb(w, payload, x=X, fill=True)
rec = embed.extract_xlsb(w_st, x=X)
# the embedded field is the tiled payload; check it matches what we wrote
cap = w.size * X
tiled = np.tile(payload, int(np.ceil(cap / len(payload))))[:cap]
ok = np.array_equal(rec, tiled.astype(np.uint8))
print(f"  round-trip exact: {ok}")
print("  perturbation:", {k: round(v, 8) for k, v in
                            embed.perturbation_stats(w, w_st).items()})

print("=== 2. GF image tiling ===")
tiles_clean = gf_image.weights_to_tiles(w, tile=128, max_tiles=4)
tiles_stego = gf_image.weights_to_tiles(w_st, tile=128, max_tiles=4)
print(f"  clean tiles {tiles_clean.shape} dtype={tiles_clean.dtype}")
print(f"  stego tiles {tiles_stego.shape}")
print(f"  tiles differ (stego != clean): {not np.array_equal(tiles_clean, tiles_stego)}")

print("=== 3. low-byte marginal histogram (why naive stats fail) ===")
lb_clean = gf_image.low_byte_plane(w, tile=128, max_tiles=4).ravel()
lb_stego = gf_image.low_byte_plane(w_st, tile=128, max_tiles=4).ravel()
def entropy(a):
    h = np.bincount(a, minlength=256) / a.size
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())
print(f"  low-byte entropy  clean={entropy(lb_clean):.3f}/8  stego={entropy(lb_stego):.3f}/8")

print("=== 4. SRNet forward on GPU ===")
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = detector.make_detector(width=0.5, device=dev)
x = torch.from_numpy(tiles_clean[:2].astype(np.float32)[:, None] / 255.0).to(dev)
with torch.no_grad():
    y = net(x)
nparams = sum(p.numel() for p in net.parameters())
print(f"  device={dev}  input={tuple(x.shape)}  output={tuple(y.shape)}  params={nparams:,}")
print("ALL OK" if ok else "ROUND-TRIP FAILED")
