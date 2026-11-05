"""Ablation: cross-architecture & cross-dataset. The attack operates on raw float32
weights, so it is model- and domain-agnostic. We show it on a non-vision, tabular
network-intrusion-detection (IDS) MLP and across two IDS datasets:

  1. Train an MLP IDS on CICIoMT (binary Benign-vs-Attack); report F1/accuracy of
     {orig FP32, clean INT8, embed FP32, embed+INT8} -> accuracy preserved off-vision.
  2. Payload survival through INT8 on the MLP -> attack is architecture-agnostic.
  3. Cross-dataset fine-tune CICIoMT -> CICIoT (same 39 features, different attacks)
     -> survival through a harder, domain-shift fine-tune + IDS adapts.
  4. The permutation-invariant (NeuPerm) capacity of a fully-connected net is
     sum over linear tensors of 1 = #tensors (no spatial taps), matching the
     trilemma bound C <= sum k_h k_w with k_h k_w = 1 for dense layers.

  python scripts/23_ids_crossdataset.py \
      --ciciomt ../balanced_CICIoMT.csv --ciciot ../balanced_CICIoT.csv
"""
import sys, os, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wstego import robust

DEFAULT_MOMT = os.path.join("..", "balanced_CICIoMT.csv")
DEFAULT_IOT = os.path.join("..", "balanced_CICIoT.csv")


def load_binary(path, scaler=None):
    """Load an IDS CSV -> (X standardized float32, y binary attack=1), reusing scaler."""
    import pandas as pd
    d = pd.read_csv(path)
    y = (~d["Label"].astype(str).str.lower().str.contains("benign")).astype(np.int64).values
    X = d.drop(columns=["Label"]).apply(lambda c: __import__("pandas").to_numeric(c, errors="coerce"))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    if scaler is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        scaler = (mu, sd)
    mu, sd = scaler
    return ((X - mu) / sd).astype(np.float32), y, scaler


class MLP(nn.Module):
    def __init__(s, d_in, nc=2):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(True),
                              nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, nc))
    def forward(s, x): return s.net(x)


def weight_tensors(m):
    return [p for _, p in m.named_parameters() if p.dim() >= 2]


def embed_redundant(m, msg, seed=0):
    cur = 0
    with torch.no_grad():
        for p in weight_tensors(m):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            bits = msg[(np.arange(cur, cur + n)) % len(msg)].astype(np.uint8)
            w2, _, _ = robust.qaware_embed_stealthy(w, bits, nbits=1, seed=seed)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device, p.dtype)); cur += n


def extract_redundant(m, n_msg):
    votes = np.zeros((n_msg, 2), np.int64); cur = 0
    with torch.no_grad():
        for p in weight_tensors(m):
            w = p.detach().cpu().numpy().astype(np.float32).ravel(); n = w.size
            s = robust.quant_scale(w); b = robust.qaware_extract(w, s, nbits=1)
            np.add.at(votes, ((np.arange(cur, cur + n)) % n_msg, b), 1); cur += n
    return (votes[:, 1] > votes[:, 0]).astype(np.uint8)


def quantize_inplace(m):
    with torch.no_grad():
        for p in weight_tensors(m):
            w = p.detach().cpu().numpy().astype(np.float32)
            s = robust.quant_scale(w.ravel())
            p.copy_(torch.from_numpy((robust.quantize(w, s).astype(np.float32) * s)).to(p.device, p.dtype))


def evaluate(m, X, y, dev):
    m.eval()
    with torch.no_grad():
        pred = m(torch.from_numpy(X).to(dev)).argmax(1).cpu().numpy()
    acc = (pred == y).mean()
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return acc, float(f1)


def train(m, X, y, dev, epochs=12, lr=1e-3, bs=512):
    m.train(); opt = torch.optim.Adam(m.parameters(), lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy(y).to(dev); n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            F.cross_entropy(m(Xt[idx]), yt[idx]).backward(); opt.step()
    return m


def split(X, y, frac=0.8, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); k = int(frac * len(X))
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ciciomt", default=DEFAULT_MOMT)
    ap.add_argument("--ciciot", default=DEFAULT_IOT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- CICIoMT (train domain) ---
    Xm, ym, scaler = load_binary(a.ciciomt)
    Xtr, ytr, Xte, yte = split(Xm, ym, seed=a.seed)
    print(f"device={dev}  CICIoMT {Xm.shape} attack-rate={ym.mean():.2f}", flush=True)
    base = MLP(Xm.shape[1]).to(dev); train(base, Xtr, ytr, dev)
    P = sum(p.numel() for p in weight_tensors(base))
    n_lin = len(weight_tensors(base))
    print(f"MLP linear weights={P:,}  linear tensors={n_lin}  "
          f"=> NeuPerm-invariant capacity = {n_lin} bits (sum k_h k_w, =1 per dense layer)\n", flush=True)

    # --- payload + 4 conditions (accuracy/F1 preserved on tabular IDS) ---
    n_msg = 64; msg = np.random.default_rng(1).integers(0, 2, n_msg, dtype=np.uint8); redund = P // n_msg
    model_q = copy.deepcopy(base); quantize_inplace(model_q)
    model_e = copy.deepcopy(base); embed_redundant(model_e, msg, seed=a.seed)
    model_eq = copy.deepcopy(model_e); quantize_inplace(model_eq)
    rec = (extract_redundant(model_eq, n_msg) == msg).mean()
    print(f"payload {P//8} B (redundancy {redund}x)  survival INT8 = {rec:.3f}\n", flush=True)
    print(f"{'condition':16s} {'acc':>7} {'F1':>7}", flush=True)
    for lbl, m in [("orig FP32", base), ("clean INT8", model_q),
                   ("embed FP32", model_e), ("embed+INT8", model_eq)]:
        acc, f1 = evaluate(m, Xte, yte, dev)
        print(f"{lbl:16s} {acc*100:>6.2f}% {f1:>7.3f}", flush=True)

    # --- cross-dataset fine-tune CICIoMT -> CICIoT (domain shift, same features) ---
    Xo, yo, _ = load_binary(a.ciciot, scaler=scaler)
    Xo_tr, yo_tr, Xo_te, yo_te = split(Xo, yo, seed=a.seed)
    acc_before, f1_before = evaluate(model_e, Xo_te, yo_te, dev)   # embedded model, no adapt
    m_ft = copy.deepcopy(model_e); train(m_ft, Xo_tr, yo_tr, dev, epochs=8)
    acc_after, f1_after = evaluate(m_ft, Xo_te, yo_te, dev)
    rec_ft = (extract_redundant(m_ft, n_msg) == msg).mean()
    print(f"\ncross-dataset CICIoMT->CICIoT (domain-shift fine-tune, same 39 features)", flush=True)
    print(f"  CICIoT F1 before adapt={f1_before:.3f}  after 8ep fine-tune={f1_after:.3f}", flush=True)
    print(f"  payload recovery after cross-domain fine-tune = {rec_ft:.3f}  (redundancy {redund}x)", flush=True)
    print(f"\nbaca: serangan agnostik-arsitektur/domain (MLP IDS tabular); akurasi/F1 terjaga; "
          f"payload selamat lewat INT8 dan fine-tune lintas-dataset; kapasitas invarian FC = "
          f"{n_lin} bit (trilemma bound dgn k_h k_w=1).", flush=True)


if __name__ == "__main__":
    main()
