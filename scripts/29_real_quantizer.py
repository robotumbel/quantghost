"""Validate QuantGhost survival through PyTorch's PRODUCTION INT8 kernel
(torch.ao.quantization + torch.quantize_per_tensor/per_channel, fbgemm), not our
analytical eq:quant. Key subtlety found by probing: PyTorch symmetric qint8 uses
scale = max|w| / 127.5 (half the full 255 range), NOT max|w| / 127. So we test two
attackers:
  (1) convention-NAIVE  : embeds on the max/127 grid (our paper's default);
  (2) convention-AWARE  : embeds on the victim's actual max/127.5 grid.
Victim then quantizes with the REAL torch kernel (observer computes the scale from the
embedded weights), and we read the int8 codes back and measure bit survival, per-tensor
and per-channel, on a real CIFAR-10 ResNet-56. Redundancy + majority vote gives the
end-to-end byte-exact figure.

  python scripts/29_real_quantizer.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

QMIN, QMAX = -127, 127


def set_lsb(q, bit):
    return (q & ~1) | bit.astype(q.dtype)


def embed_per_tensor(w, bits, div, rng, noise=0.45):
    """Place each weight on the s=max/div grid, write payload in code LSB, pin the
    max-magnitude weight (defines the scale) so the victim observer recovers the same s."""
    s = np.abs(w).max() / div
    q = np.clip(np.round(w / s).astype(np.int64), QMIN, QMAX)
    imax = int(np.argmax(np.abs(w)))         # the true max-magnitude weight = M
    qe = set_lsb(q, bits[:w.size].astype(np.int64))
    qe = np.clip(qe, QMIN, QMAX)
    u = rng.uniform(-noise, noise, size=w.size).astype(np.float64); u[imax] = 0.0
    we = ((qe + u) * s).astype(np.float32)
    we[imax] = w[imax]                       # keep M exactly so victim observer recovers s
    return we, s


def victim_quantize_per_tensor(w):
    t = torch.from_numpy(w).float()
    obs = MinMaxObserver(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
    obs(t); s, zp = obs.calculate_qparams()
    q = torch.quantize_per_tensor(t, float(s), int(zp), torch.qint8)
    return q.int_repr().numpy().astype(np.int64)


def victim_quantize_per_channel(w):  # ch axis 0
    t = torch.from_numpy(w).float()
    obs = PerChannelMinMaxObserver(ch_axis=0, qscheme=torch.per_channel_symmetric, dtype=torch.qint8)
    obs(t); s, zp = obs.calculate_qparams()
    q = torch.quantize_per_channel(t, s.double(), zp.long(), 0, torch.qint8)
    return q.int_repr().numpy().astype(np.int64)


def embed_per_channel(w, bits, div, rng, noise=0.45):
    out = np.empty_like(w, np.float32); off = 0; flat_bits = bits
    for c in range(w.shape[0]):
        wc = w[c].ravel()
        we, _ = embed_per_tensor(wc, flat_bits[off:off + wc.size], div, rng, noise)
        out[c] = we.reshape(w[c].shape); off += wc.size
    return out


def majority(rec_bits, n_payload, R):
    v = rec_bits[:n_payload * R].reshape(n_payload, R)
    return (v.mean(1) >= 0.5).astype(np.int64)


def main():
    import torchvision  # noqa
    torch.hub.set_dir("data/torchhub")
    print("loading real CIFAR-10 ResNet-56 ...", flush=True)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56",
                           pretrained=True, trust_repo=True).eval()
    tensors = [p.detach().cpu().numpy().astype(np.float32)
               for _, p in model.named_parameters() if p.dim() >= 2]
    P = sum(t.size for t in tensors)
    print(f"weight tensors={len(tensors)}  params={P:,}\n", flush=True)
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, P, dtype=np.int64)

    print(f"{'mode':36s} {'raw survival':>13}", flush=True)
    for div, lbl in [(127.0, "naive grid (max/127)"), (127.5, "convention-aware (max/127.5)")]:
        # ---- per-tensor ----
        good = tot = 0; off = 0
        for t in tensors:
            w = t.ravel()
            we, _ = embed_per_tensor(w, payload[off:off + w.size], div, rng)
            codes = victim_quantize_per_tensor(we.reshape(t.shape)).ravel()
            rec = (codes & 1)
            good += (rec == payload[off:off + w.size]).sum(); tot += w.size; off += w.size
        print(f"  per-tensor, {lbl:24s} {good/tot:>13.4f}", flush=True)
        # ---- per-channel ----
        good = tot = 0; off = 0
        for t in tensors:
            we = embed_per_channel(t, payload[off:off + t.size], div, rng)
            codes = victim_quantize_per_channel(we).ravel()
            rec = (codes & 1)
            good += (rec == payload[off:off + t.size]).sum(); tot += t.size; off += t.size
        print(f"  per-channel,{lbl:24s} {good/tot:>13.4f}", flush=True)

    # ---- end-to-end byte-exact via redundancy + majority vote (convention-aware) ----
    print("\nEnd-to-end (convention-aware, per-tensor, redundancy + majority vote):", flush=True)
    msg = np.unpackbits(np.frombuffer(b"QuantGhost->TIFS: survives real torch INT8 kernel.", np.uint8)).astype(np.int64)
    for R in (1, 5, 21):
        nslot = P // 1  # one bit per weight
        rep = np.tile(msg, (nslot // len(msg)) + 1)[:nslot]
        good = 0; off = 0; recbits = np.empty(P, np.int64)
        for t in tensors:
            w = t.ravel()
            we, _ = embed_per_tensor(w, rep[off:off + w.size], 127.5, rng)
            codes = victim_quantize_per_tensor(we.reshape(t.shape)).ravel()
            recbits[off:off + w.size] = codes & 1; off += w.size
        # majority across the R repeats of each message bit (interleaved tile)
        rec_msg = []
        for b in range(len(msg)):
            idx = np.arange(b, P, len(msg))[:R]
            rec_msg.append(int(round(recbits[idx].mean())))
        acc = np.mean(np.array(rec_msg) == msg)
        print(f"  R={R:>2}: message-bit recovery = {acc:.4f} ({'byte-exact' if acc==1 else 'errors'})", flush=True)


if __name__ == "__main__":
    main()
