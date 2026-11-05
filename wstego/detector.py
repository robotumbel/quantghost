"""SRNet steganalysis CNN (Boroumand, Chen & Fridrich, IEEE TIFS 2019).

A faithful-but-compact PyTorch implementation. SRNet is the backbone Model X-Ray
uses to read the GF byte-image. It deliberately avoids fixed high-pass filters so
it can learn the residual that LSB embedding leaves; the first segment keeps full
resolution (no pooling) to preserve the stego signal.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Type1(nn.Module):
    """Conv-BN-ReLU (full resolution)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.c(x)))


class Type2(nn.Module):
    """Residual block (full resolution)."""
    def __init__(self, c):
        super().__init__()
        self.t1 = Type1(c, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return x + self.bn2(self.c2(self.t1(x)))


class Type3(nn.Module):
    """Residual block with stride-2 average pooling (downsampling)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.t1 = Type1(cin, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sc = nn.Conv2d(cin, cout, 1, stride=2, bias=False)
        self.scbn = nn.BatchNorm2d(cout)

    def forward(self, x):
        main = self.pool(self.bn2(self.c2(self.t1(x))))
        short = self.scbn(self.sc(x))
        return main + short


class Type4(nn.Module):
    """Final conv + global average pool."""
    def __init__(self, cin, cout):
        super().__init__()
        self.t1 = Type1(cin, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.bn2(self.c2(self.t1(x)))
        return x.mean(dim=(2, 3))  # global average pool -> (N, cout)


class SRNet(nn.Module):
    """Standard SRNet. `width` scales channels for speed on small GPUs."""
    def __init__(self, in_ch: int = 1, n_classes: int = 2, width: float = 1.0):
        super().__init__()
        w = lambda c: max(1, int(round(c * width)))
        self.stem = nn.Sequential(Type1(in_ch, w(64)), Type1(w(64), w(16)))
        self.t2 = nn.Sequential(*[Type2(w(16)) for _ in range(5)])
        self.t3 = nn.Sequential(
            Type3(w(16), w(16)), Type3(w(16), w(64)),
            Type3(w(64), w(128)), Type3(w(128), w(256)),
        )
        self.t4 = Type4(w(256), w(512))
        self.fc = nn.Linear(w(512), n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        return self.fc(x)


def make_detector(width: float = 0.5, device: str = "cuda") -> SRNet:
    """A width-0.5 SRNet fits comfortably in ~6 GB at 256x256, batch 16-32."""
    m = SRNet(in_ch=1, n_classes=2, width=width)
    return m.to(device)
