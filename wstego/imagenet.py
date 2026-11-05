"""ImageNet-val loader + top-1/top-5 eval (accuracy-preservation tests, B1).

Standard torchvision eval transform (Resize 256 -> CenterCrop 224 -> normalize).
The val directory must be arranged as ImageFolder: one subfolder per synset
(n01440764/, n01443537/, ...). Sorted-alphabetical synset order matches the
torchvision pretrained class index, so top-1 reproduces the published ~76% for
ResNet-50 when the data is correct.

Point it at the data with the IMAGENET_VAL env var or the `root` argument. If no
valid directory is found, val_loader() returns None so callers can skip accuracy
gracefully (the rest of the pipeline still runs).
"""
from __future__ import annotations
import os
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_val_dir(root: str | None = None) -> str | None:
    """Resolve the val directory from arg or IMAGENET_VAL env; None if missing."""
    cand = root or os.environ.get("IMAGENET_VAL")
    return cand if (cand and os.path.isdir(cand)) else None


def val_loader(root: str | None = None, batch_size: int = 64, workers: int = 4,
               resize: int = 256, crop: int = 224):
    """Return a torchvision ImageNet-val DataLoader, or None if no data found."""
    d = find_val_dir(root)
    if d is None:
        return None
    import torchvision as tv
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(resize), T.CenterCrop(crop), T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = tv.datasets.ImageFolder(d, transform=tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


@torch.no_grad()
def top1(model, loader, device, max_batches: int | None = None):
    """Evaluate top-1/top-5 over the loader. Returns (top1, top5, n_images).

    max_batches limits the number of batches for a quick estimate (None = full).
    """
    model.eval().to(device)
    c1 = c5 = n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        _, pred = out.topk(5, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(y.view(1, -1))
        c1 += correct[:1].reshape(-1).float().sum().item()
        c5 += correct[:5].reshape(-1).float().sum().item()
        n += y.size(0)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    return (c1 / n, c5 / n, n)
