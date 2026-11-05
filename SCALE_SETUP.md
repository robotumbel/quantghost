# Scale-up setup (WSL2 / cloud) for the large experiments

The local Windows path is fine for pilots, but ImageNet-scale models and the real
TF-based Model X-Ray detector need Linux GPU. Good news: **WSL2 Ubuntu is already
installed on this machine** (`wsl -l -v` shows Ubuntu, v2). WSL2 gives:
- TensorFlow GPU support (unavailable on native Windows since TF 2.10) → run the
  *real* Model X-Ray detector, not just our reproduction.
- Much lower kernel-launch overhead than Windows WDDM (the ~25× slowdown).

## A. WSL2 (recommended, already installed)
```bash
wsl -d Ubuntu                       # from PowerShell; enter Ubuntu
# inside Ubuntu:
nvidia-smi                          # should show the RTX 4050 (WSL2 GPU passthrough)
cd /mnt/d/Code-PhD/Projet_2026/q1_weightstego   # project is visible under /mnt/d
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/00_smoke.py          # verify pipeline runs on GPU under WSL2
```
If `nvidia-smi` fails inside WSL2: update the Windows NVIDIA driver (it provides
the WSL CUDA stack; no separate CUDA install needed inside Ubuntu).

### Running the REAL Model X-Ray detector (for the "evade the released detector" claim)
```bash
cd /mnt/d/Code-PhD/Projet_2026/ModelXRay
python -m venv .venv-mxr && source .venv-mxr/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/plots/plot_all.py    # sanity (plot-only path)
# then point its X-LSB attack at OUR qaware-stealthy weights and run exp1 OSL.
```

## B. Cloud alternative (if local GPU too small for ImageNet)
RTX 4050 has 6 GB — fine for ResNet-50 inference/embedding, tight for training.
For ImageNet-scale: a single A10/A100 hour on RunPod/Lambda/Colab-Pro suffices for
the evaluation runs (we only need inference accuracy + embedding, not training).
Sync the `q1_weightstego/` folder; results are small CSVs.

## What to run at scale (priority)
1. **Real detectors**: run Model X-Ray OSL + MalConv on our stealthy embeddings →
   ROC/AUC (replaces our TinyCNN reproduction). → Table `tab:stealth`.
2. **Real models + accuracy**: ResNet-50 (ImageNet val top-1) and a small LLM;
   embed a real multi-MB encrypted payload; verify exact extraction + accuracy.
3. **Full pipeline**: quantize (per-tensor + per-channel + INT4) / prune (to 90%) /
   fine-tune (transfer) → survival curves. → Table `tab:survive`, Fig F3.
4. **Real NeuPerm**: faithful neuron permutation on full models; aggregate
   spatial-invariant capacity. → trilemma figure.
5. **Baselines**: EvilModel/StegoNet under quantization (expect survival 0);
   watermarking (2208.10973) capacity comparison.
6. **ECC**: swap repetition for LDPC/fountain (e.g. `pip install galois`) →
   better capacity at equal robustness.
