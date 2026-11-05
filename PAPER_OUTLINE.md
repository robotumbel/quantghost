# Paper Outline — Deployment-Robust Weight Steganography

> Working scaffold. Framing = **red-team for defense** (responsible disclosure to
> Model X-Ray & NeuPerm authors; we also propose/analyze defenses). All numbers
> below are from our pilot pipeline (`q1_weightstego/`) and must be re-run at
> scale on real models before submission.

## Candidate titles
1. *Surviving Sanitization: Deployment-Robust Steganography in Neural-Network Weights*
2. *Hiding Payloads That Survive Quantization, Pruning, Fine-tuning, and Neuron Permutation*
3. *The Capacity–Robustness–Stealth Trilemma of Neural-Network Weight Steganography*

## Target venues (Q1)
- Journals: IEEE TIFS, IEEE TDSC, Computers & Security, IEEE TNNLS (security track)
- Conferences: USENIX Security, NDSS, ACM CCS, IEEE S&P (higher bar)

---

## Abstract (draft)
Neural networks are increasingly distributed as pre-trained artifacts and then
adapted and compressed before deployment. Prior weight-steganography attacks
(StegoNet, EvilModel, MaleficNet) hide large payloads in float32 mantissa bits,
but we show these payloads are **erased by routine deployment transforms** —
INT8 quantization, pruning, and fine-tuning — which are also the basis of recent
defenses (weight-disarm, NeuPerm). We ask whether a payload can survive the full
deployment pipeline while remaining stealthy and accuracy-preserving. We present
a **quantization-aware, distribution-preserving** embedding that places each
weight at a random point inside the INT8 bin encoding its payload bit: the
payload survives quantization **by construction** while the float32 mantissa
keeps its natural distribution, evading byte-distribution steganalysis
(Model X-Ray). Combined with redundant coding, it survives pruning and
fine-tuning. We further establish a fundamental **capacity–robustness–stealth
trilemma**: high-capacity positional payloads cannot be permutation-invariant, so
NeuPerm caps capacity — yet a low-capacity **channel-permutation-invariant**
channel (embedded in the un-permuted spatial dimensions) survives even NeuPerm,
enough to bootstrap a larger payload. Our results show deployment transforms and
NeuPerm provide **incomplete** sanitization, and we discuss what a complete
defense requires.

---

## Contributions
1. **Threat characterization**: first to show prior weight-stego payloads are
   fully erased by INT8 quantization (0% survival) — quantifying why
   quantization "works" as a defense.
2. **Deployment-robust + stealthy embedding**: quantization-aware bin placement
   → 100% survival through quantization with natural-looking mantissa
   (entropy 7.999 ≈ benign) that evades Model X-Ray-class detectors; +redundancy
   → survives pruning (100%) and fine-tuning (tunable).
3. **Capacity–robustness–stealth trilemma**: formal + empirical — at most two of
   {high-capacity, permutation-invariant, stealthy-quant-robust}.
4. **NeuPerm analysis + partial bypass**: NeuPerm kills high-capacity payloads but
   a spatial-invariant low-capacity channel survives (100%), sufficient for a
   bootstrap. Deployment sanitization is incomplete.
5. **Defense discussion**: what detection/sanitization must do to close the gap.

---

## Threat model
- **Attacker** distributes a pre-trained FP32 model (model hub / supply chain) with
  a hidden payload; goal = payload reaches the *deployed* model and is recoverable.
- **Victim/defender** may apply any subset of: PTQ to INT8, magnitude pruning,
  light fine-tuning / transfer learning, neuron permutation (NeuPerm), and run a
  steganalysis detector (Model X-Ray OSL, MalConv, entropy/threshold).
- **Attacker knowledge**: knows the quantization scheme is standard per-tensor
  symmetric INT8 (realistic; it's the default). Does NOT control the victim's data.
- **Success** = payload bit-recovery after the transform chain, at a stated
  capacity, with model accuracy preserved and the model passing detectors.
- **Ethics**: payloads in experiments are random bytes / benign binaries; no live
  malware. Responsible disclosure to affected detector/defense authors.

---

## Paper structure
1. **Introduction** — supply-chain threat; gap = survival through deployment.
2. **Background & Related Work**
   - Weight stego (StegoNet '20, EvilModel '21/'22, MaleficNet) — high-cap, NOT robust.
   - Detection (Model X-Ray '24, MalConv, entropy thresholds).
   - Defenses/sanitization (weight-disarm, NeuPerm '25, quantization).
   - Robust DNN watermarking (2208.10973) — robust+stealth but LOW-cap (≤2KB),
     requires pre-training. We differ: high-cap, post-hoc, offensive.
3. **Motivation: payloads die in deployment** — naive X-LSB 0% survival vs X (our Fig).
4. **Method**
   - 4.1 Quantization-aware bin embedding (survival by construction; scale pinning).
   - 4.2 Distribution-preserving stealth (random sub-quantum placement).
   - 4.3 Redundant coding for pruning/fine-tuning (repetition→ECC).
   - 4.4 Channel-invariant low-capacity channel (spatial dimensions).
5. **The Capacity–Robustness–Stealth Trilemma** — argument + empirical frontier.
6. **Evaluation** (see plan below).
7. **Defenses** — what detection/sanitization must do; limits of NeuPerm.
8. **Ethics & Responsible Disclosure**.
9. **Conclusion**.

---

## Results: have vs. needed
| Claim | Pilot result (have) | Needed for submission |
|-------|---------------------|------------------------|
| Naive payload dies in quant | 0% survival, all X | Confirm on ResNet-50/LLM |
| Quant-aware survives quant | 100% | Real models, multiple quant schemes (per-channel, INT4) |
| Stealth (mantissa natural) | entropy 7.999 | done — see stealth rows below |
| Stealth vs real OSL (homogeneous zoo) | ours 0.747 DETECTED — **artifact** of near-identical benign models | — |
| **Stealth vs real OSL (DIVERSE GHRP zoo, 20+20 support)** | **ours 0.50 = chance (EVADED); naive 0.81 (DETECTED) → AIRTIGHT** (`C3c`) | ✓ done; optional: faithful exp1 protocol |
| Stealth vs threshold detectors (C4) | evades MalConv/histogram-KL/byte-autocorr (~0.5); caught by byte_entropy (1.0) & weight_value_dist (0.84 per-tensor) | per-tensor helps value-dist; byte_entropy still catches |
| Survives pruning | **real ResNet-56: unstruct 30% rec 1.0 @ acc 93.8%; payload only dies (0.54) at 70–90% where acc also collapses (78%→20%); structured prune destroys model (10% acc) at all ratios** (`22`) | ✓ done — no clean operating point |
| Survives fine-tuning | **real ResNet-56, 3ep CIFAR: recovery 1.000 at BOTH 3326× and 207× redundancy, top-1 94.2% — light same-task fine-tune does NOT sanitize** (`22`) | ✓ done |
| Trilemma | **formalized: Def/Lemma/Prop/Thm/Cor in paper.tex §trilemma — bound C ≤ Σ kh·kw; ResNet-18 gap factor 5.7e4** | ✓ done |
| NeuPerm: high-cap dies | **real ResNet-18: flat 1.0→0.51 (chance)** (`21`) | ✓ done |
| NeuPerm: spatial-invar survives | **real ResNet-18: 1.0→1.0; 196-bit aggregate capacity, 20 conv tensors** (`21`) | ✓ done |
| NeuPerm function-preserving | **ResNet-18 Δlogit fp32=1.8e-3, fp64=1.1e-14 (machine-eps) → accuracy identical by construction** (`21`) | ✓ done (handles residual/BN/downsample) |
| Accuracy preserved | **real CIFAR-10 ResNet-56: −0.19pp (orig→embed+INT8), 106KB @ 99.997% recovery** (`20`) | ✓ done (ImageNet/ResNet-50 decided-skip: redundant, code ready if needed) |
| Capacity | **3.2 MB on ResNet-50, 99.9999% bit-recovery after INT8** (`14`) | + LLM-scale, byte-exact w/ ECC |
| Baselines | — | EvilModel/StegoNet survival=0; watermark cap comparison |
| **Cross-arch/cross-dataset ablation** | **MLP-IDS CICIoMT: F1 0.989 preserved, survival 1.0, FC invariant-cap 3 bits; cross-dataset fine-tune CICIoMT→CICIoT breaks payload (0.469@208×)** (`23`) | ✓ done — agnostic + OOD fine-tune is stronger sanitizer |

---

## Key figures (planned)
- F1: survival vs X (naive) — the motivating "everything dies".
- F2: method schematic (bin placement + redundancy + spatial channel).
- F3: survival through {quant, prune, fine-tune} vs redundancy/capacity.
- F4: detector ROC — ours evades, naive detected.
- F5: trilemma frontier (capacity vs permutation-robustness).
- F6: accuracy vs capacity.

## Immediate next experiments (priority order)
1. **Empirical stealth**: run our reproduced OSL on stealthy-embedded models →
   confirm AUC≈0.5 (vs ~1.0 for structured naive). *(have the detector code)*
2. **Scale**: ResNet-50 + ImageNet val accuracy; embed real MB payload; full pipeline.
3. **Real NeuPerm**: implement faithful neuron permutation on a full model.
4. **Baselines**: EvilModel/StegoNet under quantization (→0); watermark capacity.
5. **ECC**: replace repetition with LDPC/fountain for capacity efficiency.

> Compute note: steps 2–3 need WSL2/Linux or cloud GPU (Windows WDDM makes the
> local 4050 ~25× slower; fine for pilots, not for ImageNet-scale).

