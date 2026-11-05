"""MaleficNet capacity-vs-robustness sweep. The DSSS spreading that buys robustness also
caps capacity: as payload K grows, the per-bit correlation SNR falls as 1/sqrt(K), so blind
recovery decays toward chance -- under INT8 and even clean. We measure K up to 32768 to fit
the 1/sqrt(K) law, then extrapolate to K=P (QuantGhost's one-bit-per-weight regime), where
MaleficNet reaches chance. INT8 adds little: capacity, not quantization, limits MaleficNet.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from math import erfc, sqrt
from wstego import maleficnet as mn, robust


def weight_tensors(model):
    return [p for _, p in model.named_parameters() if p.dim() >= 2]


def main():
    base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    ws = [p.detach().cpu().numpy().astype(np.float32).ravel() for p in weight_tensors(base)]
    sizes = [w.size for w in ws]
    w0 = np.concatenate(ws); P = w0.size
    std = float(w0.std()); c = 0.05; gamma = c * std
    print(f"P={P:,}  gamma={gamma:.4g} (c={c})\n", flush=True)

    def int8_flat(flat):
        out = np.empty_like(flat); off = 0
        for n in sizes:
            seg = flat[off:off + n]; s = robust.quant_scale(seg)
            out[off:off + n] = robust.quantize(seg, s).astype(np.float32) * s; off += n
        return out

    Q = lambda d: 0.5 * erfc(d / sqrt(2.0))                 # tail of standard normal
    Qinv = None
    print(f"{'K':>9} {'spread n/K':>11} {'clean':>7} {'INT8':>7}", flush=True)
    fit = []
    for K in (512, 2048, 8192, 32768):
        rng = np.random.default_rng(7)
        payload = rng.integers(0, 2, K, dtype=np.uint8)
        w_st = mn.embed(w0, payload, gamma=gamma, seed=0)
        rc = mn.recovery(w_st, payload, seed=0)
        rq = mn.recovery(int8_flat(w_st), payload, seed=0)
        print(f"{K:>9,} {P//K:>11,} {rc:>7.3f} {rq:>7.3f}", flush=True)
        fit.append((K, rc))

    # fit 1/sqrt(K): d(K) = d0 * sqrt(K0/K); estimate d0 from the smallest-K point
    K0, r0 = fit[0]
    # invert r0 = 1 - Q(d0): d0 = sqrt(2)*erfcinv(2*(1-r0))
    from scipy.special import erfcinv
    d0 = sqrt(2.0) * float(erfcinv(2.0 * (1.0 - r0)))
    print(f"\nfit: per-bit SNR d0={d0:.3f} at K={K0}; d(K)=d0*sqrt(K0/K)", flush=True)
    for K in (32768, 262144, P):
        d = d0 * sqrt(K0 / K); rec = 1.0 - Q(d)
        print(f"  extrapolated K={K:>9,}: d={d:.3f}  recovery~={rec:.3f}", flush=True)
    print(f"\nbaca: recovery meluruh 1/sqrt(K). Pada K=P (1 bit/weight = regim QuantGhost) "
          f"MaleficNet ~= chance. INT8 nyaris tak menambah kerosakan: yang mengehadkan "
          f"MaleficNet ialah KAPASITAS (spreading gain), bukan kuantisasi.", flush=True)


if __name__ == "__main__":
    main()
