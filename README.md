# QuantGhost — Deployment-Robust, Stealthy Weight Steganography

Reproducibility package for the paper *"Surviving Sanitization with QuantGhost: A
Capacity–Robustness–Stealth Trilemma for Weight Steganography"* (under submission, IEEE
TIFS).

QuantGhost is a quantization-aware weight-embedding **attack**: it writes each payload bit
into the INT8 code that deployment quantization will produce, so the payload survives
INT8/INT4 by construction while keeping the mantissa distribution natural (defeating
byte-image steganalysis). The paper proves a **capacity–robustness–stealth trilemma** —
no weight-embedding scheme can be simultaneously high-capacity, permutation-invariant, and
stealthy-quantization-robust — and shows that the strongest sanitizer (neuron permutation)
bounds but does not close the channel. The framing is **red-team for defense**: we also
propose a detector and a defense-in-depth recipe. Payloads are random bytes or an AES key;
no real malware is used.

> **Ethics.** This is defensive security research. The benign model populations and the
> Model X-Ray detector are public; payloads are encrypted random bytes (and, in the
> bootstrap demo, an AES-128 key). Nothing here packages or distributes malware.

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # or conda
pip install -r requirements.txt                  # install torch per pytorch.org for your CUDA
```

Tested on Windows 11 + Python 3.12 + RTX 4050 (CUDA 12.8) and on CPU. A GPU is optional
(used only to speed up the real-model scripts); everything runs on CPU.

## Data

| Dataset | How to obtain | Used by |
|---|---|---|
| CIFAR-10 | auto-downloaded by torchvision on first run | accuracy, pruning, ViT-acc |
| Pretrained ResNet-18/50, ViT-B/16 | auto-downloaded by torchvision | capacity, NeuPerm, ViT |
| Pretrained CIFAR ResNet-56 | auto-pulled from `torch.hub` (chenyaofo/pytorch-cifar-models) | accuracy, pruning, production-quantizer |
| CIFAR carrier zoo (`data/zoo_cifar.npz`, 60 models) | included | residual detector |
| GHRP STL-10 zoo (`data/ghrp_stl10_sub.npz`, 3000 models) | subsample of Schürholt et al. *Model Zoos* (NeurIPS 2022); 3000-model cache included | diverse-zoo detector, Model X-Ray stealth |
| ImageNet-val | **not included** (gated, 6 GB); the ResNet-50 capacity number is the only result needing it and is corroborated by the CIFAR/ViT accuracy results | (optional) |

## Reproducing each paper result

Run from the repo root, e.g. `python scripts/29_real_quantizer.py`. Each writes a log to
`results/`.

| Paper element | Script | Output |
|---|---|---|
| Survival through INT8/INT4 (Tab. "Survival through quantization") | `08_robust_survival.py`, `15_quant_schemes.py` | `A4_survival_quant.txt`, `B2B3_quant_schemes.txt` |
| **Survival vs bit-width INT8→INT3** (Fig.) | `28_analysis_figs.py` → `make_analysis_figs.py` | `ANL_survbits.npz`, `paper/fig_survival_bits.pdf` |
| **Production INT8 kernel** (Tab. "production quantizer") | `29_real_quantizer.py` | `REALQ_torch_kernel.txt` |
| Accuracy on real CIFAR ResNet-56 (Tab.) | `20_cifar_accuracy.py` | `B1b_cifar_accuracy.txt` |
| Capacity, ResNet-50 3.2 MB payload | `14_mb_payload.py` | `B1_mb_payload_accuracy.txt` |
| **ViT-B/16 transformer** (Tab.) | `26_vit.py [--acc-hf <hf-model>]` | `VIT_transformer.txt` |
| Baselines under INT8 (Tab.) | `18_baselines.py` | `B7_baselines.txt` |
| Stealth vs Model X-Ray + byte detectors (Tab./Fig.) | `13_osl_stealth.py`, `12_stealth_eval.py` | `C3c_osl_ghrp_strong.txt`, `C4_detectors.txt` |
| Pruning & fine-tuning, real ResNet-56 (Tab./Fig.) | `22_prune_finetune_resnet.py` | `B4B5_prune_finetune_resnet.txt` |
| NeuPerm at real scale (Tab.) | `21_neuperm_resnet.py` | `B6b_neuperm_resnet.txt` |
| **NeuPerm CI over 30 permutations** | `31_neuperm_ci.py --n 30` | `NEUPERM_ci.txt`, `NEUPERM_ci.npz` |
| End-to-end bootstrap defeating NeuPerm (§) | `24_bootstrap.py` | `BOOT_bootstrap.txt` |
| Cross-architecture / cross-dataset ablation (Tab.) | `23_ids_crossdataset.py` | `ABL_ids_crossdataset.txt` |
| Sub-quantum residual detector + ROC/sweep (Alg./Tab./Fig.) | `25_detector.py`, `27_detector_eval.py` → `make_detector_figs.py` | `DET_eval.txt`, `DET_roc.npz`, `DET_sweep.npz` |
| **Detector on diverse GHRP zoo** (Tab.) | `30_detector_ghrp.py` | `DET_ghrp.txt`, `DET_ghrp_roc.npz` |

Bold rows are the additions made during the TIFS-strengthening revision.

## Code layout

```
wstego/            core library
  robust.py        quantization-aware embedding/extraction, scale-pinning, redundancy
  neuperm.py       faithful function-preserving neuron permutation (ResNet-18)
  ecc.py           error-correcting redundancy / majority vote
  detector.py, gf_image.py, osl.py   steganalysis baselines (Model X-Ray reproduction)
  zoo.py, imagenet.py                benign model / data sources
scripts/           numbered, one experiment each (see table above)
  make_*.py        regenerate the paper figures from results/*.npz
results/           text logs and .npz arrays produced by the scripts
data/              carrier zoos and cached datasets
```

The paper LaTeX source, the compiled PDF, and author photos are **not** part of this
artifact deposit (kept separately, subject to publisher copyright). Figures can be
regenerated from `results/*.npz` via `scripts/make_*.py`.

## License

Code released under the MIT License (see `LICENSE`). If you reuse the GHRP/Model-Zoos
weights, cite Schürholt et al. (2022); if you reuse the Model X-Ray detector, cite Gilkarov
& Dubin. Change the license here if your institution requires otherwise.

## Citing

A BibTeX entry will be added on acceptance. Until then, please cite the arXiv/preprint
version of the paper.
