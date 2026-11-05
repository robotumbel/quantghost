"""Benign trained-weight source: torchvision pretrained models.

Using real pretrained models gives diverse *natural* trained-weight
distributions for free (no zoo training), which is exactly the benign class the
detector must distinguish from stego. We flatten every conv/linear weight tensor
into one float32 vector per model.
"""
from __future__ import annotations
import numpy as np
import torch

# (name, constructor-kwargs) — small/medium nets that download quickly.
DEFAULT_ZOO = [
    "resnet18", "resnet34", "mobilenet_v2", "mobilenet_v3_small",
    "shufflenet_v2_x1_0", "squeezenet1_1", "densenet121", "efficientnet_b0",
    "regnet_y_400mf", "mnasnet1_0",
]


def load_model_weights(name: str) -> np.ndarray:
    """Download a pretrained torchvision model and return its flat float32 weights."""
    import torchvision.models as M
    ctor = getattr(M, name)
    try:
        model = ctor(weights="DEFAULT")
    except Exception:
        model = ctor(pretrained=True)  # older torchvision
    parts = []
    for p in model.parameters():
        if p.dim() >= 2:  # conv / linear weight matrices (skip biases/BN scalars)
            parts.append(p.detach().cpu().numpy().astype(np.float32).ravel())
    w = np.concatenate(parts)
    return w


def load_zoo(names=None, verbose=True) -> dict[str, np.ndarray]:
    names = names or DEFAULT_ZOO
    out = {}
    for n in names:
        try:
            w = load_model_weights(n)
            out[n] = w
            if verbose:
                print(f"  {n:24s} params={w.size:>10,d}  bytes={w.nbytes:>12,d}")
        except Exception as e:
            if verbose:
                print(f"  {n:24s} SKIP ({e.__class__.__name__}: {e})")
    return out
