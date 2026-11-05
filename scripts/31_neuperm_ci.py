"""Confidence intervals for the NeuPerm claim: instead of a single permutation, apply
MANY independent random faithful permutations to one embedded ResNet-18 and report the
DISTRIBUTION of payload recovery. Shows the positional channel collapses to chance with a
tight CI (not a lucky single seed) while the spatial-invariant channel stays at 1.000 every
time. Embed once (expensive over 11.2M conv weights); permutation is cheap, so we sweep
many seeds.

  python scripts/31_neuperm_ci.py --n 30
"""
import sys, os, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision.models as M
from wstego import robust, neuperm


def conv_weights(net):
    return [p for p in net.parameters() if p.dim() == 4]


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
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); a = ap.parse_args()
    np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  loading pretrained ResNet-18 ...", flush=True)
    base = M.resnet18(weights="DEFAULT").to(dev).eval()

    rng = np.random.default_rng(1); msg = rng.integers(0, 2, 256, dtype=np.uint8)
    nf = copy.deepcopy(base); embed_flat(nf, msg)
    flat0 = (extract_flat(nf, 256) == msg).mean()
    ns = copy.deepcopy(base); msgs = embed_spatial(ns, np.random.default_rng(2))
    spat0 = float(np.mean([(x == m).mean() for x, m in zip(extract_spatial(ns), msgs)]))

    flat, spat = [], []
    for k in range(a.n):
        seed = 100 + k
        nfp = neuperm.permute_resnet18(copy.deepcopy(nf), seed=seed)
        flat.append(float((extract_flat(nfp, 256) == msg).mean()))
        nsp = neuperm.permute_resnet18(copy.deepcopy(ns), seed=seed)
        spat.append(float(np.mean([(x == m).mean() for x, m in zip(extract_spatial(nsp), msgs)])))
        if (k + 1) % 5 == 0:
            print(f"  ...{k+1}/{a.n}", flush=True)

    flat = np.array(flat); spat = np.array(spat)
    np.savez("results/NEUPERM_ci.npz", flat=flat, spat=spat, flat0=flat0, spat0=spat0)
    print(f"\nNeuPerm over {a.n} independent random permutations (real ResNet-18):", flush=True)
    print(f"{'channel':20s} {'no-perm':>8} {'mean':>8} {'std':>7} {'min':>7} {'max':>7}", flush=True)
    print(f"{'flat-positional':20s} {flat0:>8.3f} {flat.mean():>8.3f} {flat.std():>7.3f} {flat.min():>7.3f} {flat.max():>7.3f}", flush=True)
    print(f"{'spatial-invariant':20s} {spat0:>8.3f} {spat.mean():>8.3f} {spat.std():>7.3f} {spat.min():>7.3f} {spat.max():>7.3f}", flush=True)
    # 95% CI (normal approx) for the positional channel
    lo = flat.mean() - 1.96 * flat.std() / np.sqrt(a.n)
    hi = flat.mean() + 1.96 * flat.std() / np.sqrt(a.n)
    print(f"\npositional 95% CI = [{lo:.4f}, {hi:.4f}] (chance 0.5); "
          f"spatial-invariant min over all seeds = {spat.min():.3f}", flush=True)


if __name__ == "__main__":
    main()
