# Zenodo deposit — ready-to-paste metadata

Copy each field into the matching box on the Zenodo "New upload" form.

---

**Upload type / Resource type:** Software

**Title:**
QuantGhost: Deployment-Robust, Stealthy Weight Steganography — Code and Data

**Version:** v1.0.0

**Language:** English

**License:** MIT License

---

**Creators** (Name — Affiliation — ORCID):
- Winanto, Eko Arip — Universiti Teknologi Malaysia; Universitas Dinamika Bangsa — 0000-0003-3274-1885
- Idris, Mohd Yazid — Universiti Teknologi Malaysia — <ORCID: fill in>
- Kurniabudi — Universitas Dinamika Bangsa — <ORCID: fill in>

---

**Description:**

Reproducibility artifact for the paper "Surviving Sanitization with QuantGhost: A
Capacity–Robustness–Stealth Trilemma for Weight Steganography."

QuantGhost is a quantization-aware weight-steganography attack: it writes each payload bit
into the INT8 code that deployment quantization produces, so the payload survives INT8/INT4
quantization by construction while keeping the floating-point mantissa distribution natural,
defeating byte-image steganalysis. The work proves a capacity–robustness–stealth trilemma —
no weight-embedding scheme can be simultaneously high-capacity, permutation-invariant, and
stealthy-quantization-robust — and shows that the strongest published sanitizer (faithful
neuron permutation) bounds but does not close the channel, since a small permutation-invariant
channel still suffices to bootstrap full payload recovery. The package also includes a
proposed sub-quantum residual detector and a defense-in-depth analysis.

This archive contains the full code base (the quantization-aware embedding and extraction
library, the faithful function-preserving neuron-permutation implementation, the detector,
and steganalysis baselines), the numbered experiment scripts with a one-to-one mapping from
every paper table and figure to the script that produces it, the result logs and arrays, and
the two carrier model zoos (a 60-model CIFAR zoo and a 3000-model subsample of the public
Model Zoos / GHRP STL-10 collection). Pretrained models and CIFAR-10 are fetched
automatically at runtime; experiments run on commodity hardware and a GPU is optional.

This is defensive security research (red-team for defense). All payloads are encrypted random
bytes or an AES-128 key; no live malware is embedded, executed, or distributed.

---

**Keywords:** (add one per line on Zenodo)
weight steganography
stegomalware
neural network security
model supply chain
quantization
INT8 deployment
steganalysis
neuron permutation
machine learning security
reproducibility

---

**Related/alternate identifiers** (add after you have them):
- "is supplement to" → <paper DOI or arXiv ID, once available>
- "is derived from" → Schürholt et al., Model Zoos (the GHRP STL-10 benign population)

**Notes (optional):**
Built and tested on Windows 11, Python 3.12, PyTorch 2.10 (CUDA 12.8) and on CPU.
See README.md inside the archive for per-result reproduction instructions.
