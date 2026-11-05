"""OSL detector: few-shot metric-learning steganalysis (Model X-Ray style).

The plain binary SRNet classifier overfits to training models and fails to
generalize (cross-model AUC ~0.5). Model X-Ray instead learns an EMBEDDING with
metric learning, then classifies by nearest class-centroid / 1-NN. That is what
generalizes the benign-vs-stego decision to unseen models. This module
reproduces it in torch on the GF tile representation.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector import Type1, Type2, Type3, Type4


class EmbeddingNet(nn.Module):
    """SRNet trunk -> L2-normalized embedding vector."""
    def __init__(self, in_ch=1, emb=64, width=0.5):
        super().__init__()
        w = lambda c: max(1, int(round(c * width)))
        self.stem = nn.Sequential(Type1(in_ch, w(64)), Type1(w(64), w(16)))
        self.t2 = nn.Sequential(*[Type2(w(16)) for _ in range(5)])
        self.t3 = nn.Sequential(Type3(w(16), w(16)), Type3(w(16), w(64)),
                                Type3(w(64), w(128)), Type3(w(128), w(256)))
        self.t4 = Type4(w(256), w(512))
        self.proj = nn.Linear(w(512), emb)

    def forward(self, x):
        x = self.t4(self.t3(self.t2(self.stem(x))))
        z = self.proj(x)
        return F.normalize(z, dim=1)


def online_triplet_loss(emb, labels, margin=0.3):
    """Batch-all triplet loss on L2-normalized embeddings (squared-euclid dist)."""
    d = torch.cdist(emb, emb) ** 2
    lab = labels.unsqueeze(0)
    pos = (lab == lab.t()) & ~torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    neg = lab != lab.t()
    losses = []
    for i in range(len(labels)):
        if pos[i].any() and neg[i].any():
            ap = d[i][pos[i]].unsqueeze(1)          # all positives
            an = d[i][neg[i]].unsqueeze(0)          # all negatives
            l = F.relu(ap - an + margin)
            losses.append(l[l > 0])
    if not losses:
        return emb.sum() * 0.0
    alll = torch.cat(losses)
    return alll.mean() if len(alll) else emb.sum() * 0.0


@torch.no_grad()
def embed_all(net, X, dev, bs=64):
    net.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs].astype(np.float32)[:, None] / 255.).to(dev)
        out.append(net(xb).cpu().numpy())
    return np.concatenate(out)


def centroid_1nn_eval(net, Xtr, ytr, Xte, yte, dev):
    """Build benign/stego centroids from TRAIN embeddings; classify TEST tiles.

    Returns (centroid_acc, nn_acc) on the held-out (cross-model) test tiles.
    """
    Ztr = embed_all(net, Xtr, dev)
    Zte = embed_all(net, Xte, dev)
    c0 = Ztr[ytr == 0].mean(0); c1 = Ztr[ytr == 1].mean(0)
    c0 /= np.linalg.norm(c0) + 1e-9; c1 /= np.linalg.norm(c1) + 1e-9
    # centroid: nearest of the two class means
    d0 = ((Zte - c0) ** 2).sum(1); d1 = ((Zte - c1) ** 2).sum(1)
    pred_c = (d1 < d0).astype(int)
    cent_acc = (pred_c == yte).mean()
    # 1-NN against all train embeddings
    D = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(2)
    pred_n = ytr[D.argmin(1)]
    nn_acc = (pred_n == yte).mean()
    return float(cent_acc), float(nn_acc)
