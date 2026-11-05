"""B2+B3: survival / accuracy / stealth across quantization SCHEMES.

Sweeps granularity {per-tensor, per-channel} x bit-width {INT8 (qmax=127),
INT4 (qmax=7)} for the qaware-stealthy embedding on a CIFAR-10 SmallCNN. For each
scheme reports:
  - survival : payload bit-recovery after the victim re-quantizes with that scheme
  - INT acc  : test accuracy after embed + quantize (vs a clean-quant reference)
  - mant.ent : low-mantissa-byte entropy of the embedded FP32 weights (stealth;
               ~8.0 = natural / evades Model X-Ray-class byte-distribution detectors)
The point: survival is ~1.0 BY CONSTRUCTION for every scheme; INT4 widens bins so
accuracy drops more, and we quantify the stealth/accuracy trade-off.
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from wstego import embed as emb, robust
torch.backends.cudnn.benchmark = True


class SmallCNN(nn.Module):
    def __init__(s, nc=10):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        s.f = nn.Sequential(blk(3, 32), blk(32, 32), nn.MaxPool2d(2),
                            blk(32, 64), blk(64, 64), nn.MaxPool2d(2),
                            blk(64, 128), nn.AdaptiveAvgPool2d(1))
        s.h = nn.Linear(128, 10)
    def forward(s, x): return s.h(s.f(x).flatten(1))


def cifar(dev):
    tr = tv.datasets.CIFAR10("data/cifar", train=True, download=True)
    te = tv.datasets.CIFAR10("data/cifar", train=False, download=True)
    f = lambda d: (torch.from_numpy(d.data).to(dev).permute(0, 3, 1, 2).contiguous(),
                   torch.tensor(d.targets, device=dev))
    return f(tr), f(te)


def acc(net, Xte, yte, dev):
    net.eval(); c = 0
    with torch.no_grad():
        for i in range(0, len(Xte), 1000):
            xb = Xte[i:i + 1000].float() / 255.
            c += (net(xb).argmax(1) == yte[i:i + 1000]).sum().item()
    return c / len(Xte)


def train(net, Xtr, ytr, dev, epochs=6, bs=512, sub=10000):
    sel = torch.randperm(len(Xtr), device=dev)[:sub]; Xtr, ytr = Xtr[sel], ytr[sel]
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=5e-4)
    for _ in range(epochs):
        p = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), bs):
            idx = p[i:i + bs]; xb = Xtr[idx].float() / 255.
            opt.zero_grad(); F.cross_entropy(net(xb), ytr[idx]).backward(); opt.step()


def weight_tensors(net):
    return [p for p in net.parameters() if p.dim() >= 2]


def embed_net(net, payload, gran, qmax, seed=0):
    """Embed qaware-stealthy into every weight tensor with the given scheme.
    Returns (survival_after_requant, mantissa_entropy_of_embedded_fp32)."""
    num = den = 0; ent_w = []
    with torch.no_grad():
        for p in weight_tensors(net):
            shp = p.shape
            if gran == "tensor":
                w = p.detach().cpu().numpy().astype(np.float32).ravel()
                w2, s, car = robust.qaware_embed_stealthy(w, payload, nbits=1, qmax=qmax, seed=seed)
                p.copy_(torch.from_numpy(w2.reshape(shp)).to(p.device))
                sv = robust.quant_scale(w2, qmax)
                wq = robust.quantize(w2, sv, qmax).astype(np.float32) * sv
                rec = robust.qaware_extract(wq, sv, nbits=1, qmax=qmax)
                surv = robust.survival_rate(rec, payload, nbits=1, carrier=car)
                cnt = int(car.sum())
            else:  # per-channel
                w2d = p.detach().cpu().numpy().astype(np.float32).reshape(shp[0], -1)
                w2, sc, car = robust.qaware_embed_stealthy_pc(w2d, payload, nbits=1, qmax=qmax, seed=seed)
                p.copy_(torch.from_numpy(w2.reshape(shp)).to(p.device))
                scv = robust.quant_scale_pc(w2, qmax)
                wq = (robust.quantize_pc(w2, scv, qmax).astype(np.float32) * scv[:, None])
                rec = robust.qaware_extract_pc(wq, scv, nbits=1, qmax=qmax)
                surv = robust.survival_rate(rec, payload, nbits=1, carrier=car.reshape(-1))
                cnt = int(car.sum())
            num += surv * cnt; den += cnt
            ent_w.append(p.detach().cpu().numpy().astype(np.float32).ravel())
    ent = robust.mantissa_lowbyte_entropy(np.concatenate(ent_w))
    return num / den, ent


def quantize_net(net, gran, qmax):
    """Victim re-quantizes every weight tensor with the given scheme."""
    with torch.no_grad():
        for p in weight_tensors(net):
            shp = p.shape
            if gran == "tensor":
                w = p.detach().cpu().numpy().astype(np.float32)
                s = robust.quant_scale(w.ravel(), qmax)
                wq = robust.quantize(w, s, qmax).astype(np.float32) * s
            else:
                w2d = p.detach().cpu().numpy().astype(np.float32).reshape(shp[0], -1)
                sc = robust.quant_scale_pc(w2d, qmax)
                wq = (robust.quantize_pc(w2d, sc, qmax).astype(np.float32) * sc[:, None]).reshape(shp)
            p.copy_(torch.from_numpy(wq.reshape(shp)).to(p.device))


def main():
    np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (Xtr, ytr), (Xte, yte) = cifar(dev)
    base = SmallCNN().to(dev); train(base, Xtr, ytr, dev)
    a0 = acc(base, Xte, yte, dev)
    ent0 = robust.mantissa_lowbyte_entropy(
        np.concatenate([p.detach().cpu().numpy().astype(np.float32).ravel() for p in weight_tensors(base)]))
    print(f"baseline FP32 acc={a0:.3f}  mant.entropy={ent0:.3f} (natural)\n", flush=True)
    payload = emb.random_payload_bits(1 << 16, 0)

    print(f"{'scheme':22s} {'survival':>9} {'clean acc':>10} {'embed acc':>10} {'mant.ent':>9}", flush=True)
    for gran in ("tensor", "channel"):
        for qmax, tag in ((127, "INT8"), (7, "INT4")):
            # clean-quant reference (accuracy cost of quantization alone)
            netc = copy.deepcopy(base); quantize_net(netc, gran, qmax); ac = acc(netc, Xte, yte, dev)
            # embed + victim quantize
            nete = copy.deepcopy(base)
            surv, ent = embed_net(nete, payload, gran, qmax)
            quantize_net(nete, gran, qmax); ae = acc(nete, Xte, yte, dev)
            print(f"{f'per-{gran} {tag}':22s} {surv:>9.3f} {ac:>10.3f} {ae:>10.3f} {ent:>9.3f}", flush=True)
    print("\nbaca: survival~1.0 di SEMUA skema (by construction). per-channel biasanya "
          "akurasi lebih baik; INT4 bin lebih lebar -> akurasi turun, entropi tetap ~natural.", flush=True)


if __name__ == "__main__":
    main()
