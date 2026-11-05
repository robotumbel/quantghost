"""B1 helper: build an ImageNet-val ImageFolder that matches torchvision's
pretrained class index, so `14_mb_payload.py --val <out>` reports a faithful
ResNet-50 top-1 (~76%).

torchvision's pretrained models index classes by *sorted synset order*. We name the
output folders by zero-padded label index (0000..0999); ImageFolder sorts them into
numeric == label order == torchvision's index, so top-1 is correct without needing
the wnid table.

Three acquisition paths (pick one):

  # A) HuggingFace imagenet-1k (gated: accept terms at hf.co/datasets/imagenet-1k,
  #    then `huggingface-cli login` or set HF_TOKEN). 50k images, ~6 GB.
  python scripts/prepare_imagenet_val.py --source hf --out D:\\data\\imagenet\\val

  # B) Official tars from image-net.org (needs the val tar + devkit in --tar-dir):
  #    ILSVRC2012_img_val.tar  +  ILSVRC2012_devkit_t12.tar.gz
  python scripts/prepare_imagenet_val.py --source tar --tar-dir D:\\data\\imagenet_raw \\
         --out D:\\data\\imagenet\\val

  # C) Validate / use a folder you already arranged as ImageFolder:
  python scripts/prepare_imagenet_val.py --source check --out D:\\data\\imagenet\\val

After it prints "READY", run:
  python scripts/14_mb_payload.py --val D:\\data\\imagenet\\val --max_batches 50
(omit --max_batches for the full 50k-image number).
"""
from __future__ import annotations
import argparse
import os
import sys


def via_hf(out: str):
    """Stream HF imagenet-1k validation into out/<label:04d>/<idx>.jpeg."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("need `pip install datasets`")
    print("loading imagenet-1k validation from HuggingFace (gated; needs token)...", flush=True)
    ds = load_dataset("imagenet-1k", split="validation")  # raises if not authorized
    n = len(ds)
    print(f"{n} images; writing ImageFolder to {out}", flush=True)
    for i, ex in enumerate(ds):
        lbl = int(ex["label"])
        d = os.path.join(out, f"{lbl:04d}")
        os.makedirs(d, exist_ok=True)
        ex["image"].convert("RGB").save(os.path.join(d, f"{i:06d}.jpeg"), quality=95)
        if i % 2000 == 0:
            print(f"  {i}/{n}", flush=True)
    print("HF extraction done.", flush=True)


def via_tar(tar_dir: str, out: str):
    """Use torchvision.datasets.ImageNet to unpack the official tars (it arranges val
    into the correct synset layout matching the pretrained index), then symlink/copy
    into our index-named folders is unnecessary -- torchvision's own val/ layout is
    already correct, so we just point the loader there."""
    import torchvision as tv
    val_tar = os.path.join(tar_dir, "ILSVRC2012_img_val.tar")
    devkit = os.path.join(tar_dir, "ILSVRC2012_devkit_t12.tar.gz")
    for f in (val_tar, devkit):
        if not os.path.isfile(f):
            sys.exit(f"missing {f} (download from image-net.org)")
    print(f"unpacking official tars in {tar_dir} via torchvision.datasets.ImageNet ...", flush=True)
    tv.datasets.ImageNet(tar_dir, split="val")  # arranges tar_dir/val/<synset>/...
    arranged = os.path.join(tar_dir, "val")
    print(f"torchvision arranged val at: {arranged}", flush=True)
    print(f"NOTE: pass THIS path to --val (it is already ImageFolder + correct order):"
          f"\n  python scripts/14_mb_payload.py --val {arranged}", flush=True)


def check(out: str) -> bool:
    if not os.path.isdir(out):
        print(f"NOT READY: {out} does not exist", flush=True)
        return False
    classes = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    n_imgs = sum(len(files) for _, _, files in os.walk(out))
    print(f"{out}: {len(classes)} class folders, {n_imgs} files", flush=True)
    if len(classes) != 1000:
        print(f"WARN: expected 1000 class folders, found {len(classes)} "
              f"(top-1 will be wrong unless this is intentional)", flush=True)
    ok = len(classes) == 1000 and n_imgs >= 1000
    print("READY" if ok else "NOT READY (need 1000 synset folders w/ images)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--source", choices=["hf", "tar", "check"], required=True)
    ap.add_argument("--out", required=True, help="ImageFolder output dir")
    ap.add_argument("--tar-dir", default=None, help="dir holding the official ImageNet tars (--source tar)")
    a = ap.parse_args()
    if a.source == "hf":
        os.makedirs(a.out, exist_ok=True)
        via_hf(a.out)
        check(a.out)
    elif a.source == "tar":
        if not a.tar_dir:
            sys.exit("--source tar needs --tar-dir")
        via_tar(a.tar_dir, a.out)
    else:
        check(a.out)


if __name__ == "__main__":
    main()
