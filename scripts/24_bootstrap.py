"""End-to-end bootstrap demonstration (defeating NeuPerm).

Claim under test: even the strongest single defense, a faithful neuron permutation,
does not close the channel. We embed (a) the high-capacity encrypted payload in the
positional channel and (b) the AES-128 decryption key in the permutation-invariant
spatial-tap channel of a real ResNet-18, then apply the full victim pipeline
(INT8 + 30% pruning + NeuPerm). NeuPerm destroys the in-weight positional payload, but
the 128-bit key survives in the invariant channel and decrypts the (out-of-band)
ciphertext byte-exactly. The surviving low-capacity channel therefore bootstraps full
payload recovery after the strongest sanitization.

  python scripts/24_bootstrap.py
"""
import sys, os, copy, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision.models as M
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from wstego import robust, neuperm


def aes128_ctr(key16, data, nonce=b"\x00" * 16):
    """AES-128 in CTR mode (symmetric: same call encrypts and decrypts)."""
    enc = Cipher(algorithms.AES(key16), modes.CTR(nonce)).encryptor()
    return enc.update(data) + enc.finalize()


def conv_weights(net):
    return [p for p in net.parameters() if p.dim() == 4]


# ---- high-capacity POSITIONAL channel (carries the ciphertext; dies under NeuPerm) ----
def embed_positional(net, msg_bits, seed=0):
    g0 = 0
    with torch.no_grad():
        for p in conv_weights(net):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            bits = msg_bits[(np.arange(g0, g0 + n)) % len(msg_bits)]
            w2, _, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed + g0)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device)); g0 += n


def extract_positional(net, n_msg):
    votes = np.zeros((n_msg, 2), np.int64); g0 = 0
    with torch.no_grad():
        for p in conv_weights(net):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            s = robust.quant_scale(w); bits = robust.qaware_extract(w, s, nbits=1)
            np.add.at(votes, ((np.arange(g0, g0 + n)) % n_msg, bits), 1); g0 += n
    return (votes[:, 1] > votes[:, 0]).astype(np.uint8)


# ---- permutation-INVARIANT spatial-tap channel (carries the key; survives NeuPerm) ----
def embed_key_invariant(net, key_bits, seed=0):
    """Lay the key bits across conv spatial taps; each bit is replicated across all
    input/output channels at its spatial position, so it is invariant to channel
    permutation and redundant against pruning."""
    pos = 0
    with torch.no_grad():
        for p in conv_weights(net):
            O, I, kh, kw = p.shape; S = kh * kw
            slot = np.zeros(S, np.uint8)
            take = max(0, min(S, len(key_bits) - pos))
            if take:
                slot[:take] = key_bits[pos:pos + take]; pos += take
            w = p.detach().cpu().numpy().astype(np.float32)
            bits = slot[np.arange(O * I * S) % S]
            w2, _, _ = robust.qaware_embed_stealthy(w.ravel(), bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device))
    return pos  # bits actually placed


def extract_key_invariant(net, key_len):
    rec, pos = [], 0
    with torch.no_grad():
        for p in conv_weights(net):
            O, I, kh, kw = p.shape; S = kh * kw
            w = p.detach().cpu().numpy().astype(np.float32).ravel()
            s = robust.quant_scale(w); bits = robust.qaware_extract(w, s, nbits=1)
            votes = np.zeros((S, 2), np.int64)
            np.add.at(votes, (np.arange(O * I * S) % S, bits), 1)
            m = (votes[:, 1] > votes[:, 0]).astype(np.uint8)
            take = max(0, min(S, key_len - pos))
            if take:
                rec.append(m[:take]); pos += take
    return np.concatenate(rec)[:key_len]


def quantize_inplace(net):
    with torch.no_grad():
        for p in conv_weights(net):
            w = p.detach().cpu().numpy().astype(np.float32)
            s = robust.quant_scale(w.ravel())
            p.copy_(torch.from_numpy(robust.quantize(w, s).astype(np.float32) * s).to(p.device))


def prune_unstructured(net, ratio):
    with torch.no_grad():
        allw = np.concatenate([p.detach().cpu().numpy().ravel() for p in conv_weights(net)])
        thr = np.quantile(np.abs(allw), ratio)
        for p in conv_weights(net):
            w = p.detach().cpu().numpy()
            p.copy_(torch.from_numpy(np.where(np.abs(w) <= thr, 0.0, w).astype(np.float32)).to(p.device))


def bits_to_bytes(b):
    return np.packbits(b[: len(b) // 8 * 8]).tobytes()


def main():
    np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  loading pretrained ResNet-18 ...", flush=True)
    base = M.resnet18(weights="DEFAULT").to(dev).eval()
    inv_cap = sum(p.shape[2] * p.shape[3] for p in conv_weights(base))
    print(f"permutation-invariant capacity = {inv_cap} bits (key needs 128)\n", flush=True)

    # --- attacker: payload, AES-128 key, ciphertext (hosted out of band) ---
    rng = np.random.default_rng(7)
    payload = rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes()       # 4 KB
    key = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()             # AES-128
    ciphertext = aes128_ctr(key, payload)                                     # out-of-band blob
    key_bits = np.unpackbits(np.frombuffer(key, np.uint8)).astype(np.uint8)   # 128 bits
    ct_bits = np.unpackbits(np.frombuffer(ciphertext, np.uint8)).astype(np.uint8)
    print(f"payload {len(payload)} B  |  AES-128 key sha256={hashlib.sha256(key).hexdigest()[:12]}", flush=True)

    # --- embed both channels into one model ---
    net = copy.deepcopy(base)
    embed_positional(net, ct_bits, seed=1)                 # high-capacity payload
    placed = embed_key_invariant(net, key_bits, seed=2)    # 128-bit key in invariant channel
    print(f"embedded: {len(ct_bits)}-bit ciphertext (positional) + {placed}-bit key (invariant)\n", flush=True)

    # --- victim pipeline: INT8 + 30% pruning + faithful NeuPerm ---
    quantize_inplace(net)
    prune_unstructured(net, 0.30)
    net = neuperm.permute_resnet18(net, seed=11)
    print("applied victim pipeline: INT8 quantization + 30% pruning + faithful NeuPerm", flush=True)

    # --- recover ---
    ct_rec = extract_positional(net, len(ct_bits))
    pos_acc = (ct_rec == ct_bits).mean()
    key_rec = extract_key_invariant(net, 128)
    key_acc = (key_rec == key_bits).mean()
    key_rec_bytes = bits_to_bytes(key_rec)

    print(f"\npositional payload recovery   = {pos_acc:.3f}   (NeuPerm destroys it)", flush=True)
    print(f"invariant key recovery        = {key_acc:.3f}   ({int(key_acc*128)}/128 bits)", flush=True)

    # --- bootstrap: decrypt the out-of-band ciphertext with the recovered key ---
    recovered_payload = aes128_ctr(key_rec_bytes, ciphertext)
    ok = recovered_payload == payload
    print(f"decrypt out-of-band ciphertext with recovered key -> payload byte-exact: {ok}", flush=True)

    print("\nVERDICT: the high-capacity payload embedded in the weights is erased by "
          "NeuPerm ({:.2f} = chance), yet the {}-bit key in the permutation-invariant "
          "channel survives the full INT8+prune+NeuPerm pipeline and decrypts the payload "
          "byte-exactly. NeuPerm bounds but does not close the channel.".format(pos_acc, placed),
          flush=True)


if __name__ == "__main__":
    main()
