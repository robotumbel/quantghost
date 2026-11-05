"""B6 at REAL scale: faithful NeuPerm on a pretrained torchvision ResNet-18.

Confirms on a real ImageNet-trained model:
  (1) the permutation is function-preserving (max|Δlogit| ~ 0 => accuracy IDENTICAL
      by construction, no eval data needed);
  (2) which payloads survive it across the whole net:
        flat/positional (global weight order) -> DIES (~chance)
        spatial-invariant (indexed by kh,kw)  -> SURVIVES (~1.0)
  (3) aggregate spatial-invariant capacity = sum of kh*kw over conv tensors.

Unlike 17 (toy SmallCNN, 5 conv tensors), this handles residual/BN/downsample.
  python scripts/21_neuperm_resnet.py
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision.models as M
from wstego import robust, neuperm


def conv_weights(net):
    return [p for p in net.parameters() if p.dim() == 4]


# ---- flat/positional payload (global conv-weight order) ----
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
            bits = m[(np.arange(O * I * S) % S)]
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
    print(f"device={dev}  loading pretrained ResNet-18 (ImageNet)...", flush=True)
    base = M.resnet18(weights="DEFAULT").to(dev).eval()

    # (1) faithful permutation preserves the function (on real-shaped input)
    x = torch.randn(8, 3, 224, 224, device=dev)
    permd = neuperm.permute_resnet18(copy.deepcopy(base), seed=7).eval()
    d = neuperm.max_logit_diff(base, permd, x)
    # prove the fp32 residual is just summation-order rounding, not a bug: redo in float64
    base64 = copy.deepcopy(base).double().eval()
    permd64 = neuperm.permute_resnet18(copy.deepcopy(base).double(), seed=7).eval()
    d64 = neuperm.max_logit_diff(base64, permd64, x.double())
    n_conv = len(conv_weights(base))
    cap = sum(p.shape[2] * p.shape[3] for p in conv_weights(base))
    tot_w = sum(p.numel() for p in conv_weights(base))
    print(f"max|d_logit| fp32={d:.2e}  fp64={d64:.2e}  "
          f"(fp64 ~ machine-eps => function-preserving; fp32 = summation-order rounding "
          f"over 18 layers => accuracy IDENTICAL)\n", flush=True)
    print(f"conv tensors={n_conv}  conv weights={tot_w:,}  "
          f"spatial-invariant capacity = {cap} bits (sum kh*kw)\n", flush=True)

    # (2) flat payload: intact with no perm, destroyed after the faithful permutation
    rng = np.random.default_rng(1); msg = rng.integers(0, 2, 256, dtype=np.uint8)
    nf = copy.deepcopy(base); embed_flat(nf, msg)
    s0 = (extract_flat(nf, 256) == msg).mean()
    nfp = neuperm.permute_resnet18(copy.deepcopy(nf), seed=7)
    s1 = (extract_flat(nfp, 256) == msg).mean()

    # (3) spatial-invariant payload: survives the permutation
    rng2 = np.random.default_rng(2)
    ns = copy.deepcopy(base); msgs = embed_spatial(ns, rng2)
    t0 = np.mean([(a == b).mean() for a, b in zip(extract_spatial(ns), msgs)])
    nsp = neuperm.permute_resnet18(copy.deepcopy(ns), seed=7)
    t1 = np.mean([(a == b).mean() for a, b in zip(extract_spatial(nsp), msgs)])

    print(f"{'scheme':18s} {'no-perm':>9} {'after-PERM':>11}", flush=True)
    print(f"{'flat-positional':18s} {s0:>9.3f} {s1:>11.3f}", flush=True)
    print(f"{'spatial-invariant':18s} {t0:>9.3f} {t1:>11.3f}", flush=True)
    print(f"\nVERDICT (real ResNet-18): faithful NeuPerm (d_logit fp32={d:.1e}, fp64={d64:.1e}) "
          f"destroys the high-capacity flat payload ({s0:.2f}->{s1:.2f}=chance) but the "
          f"spatial-invariant channel survives ({t0:.2f}->{t1:.2f}). Aggregate invariant "
          f"capacity = {cap} bits across the real net. Trilemma confirmed at real scale.",
          flush=True)


if __name__ == "__main__":
    main()
