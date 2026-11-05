# JISA Submission Metadata — Editorial Manager

**Journal:** Journal of Information Security and Applications (Elsevier)

---

## Article title
QuantGhost: Deployment-Robust Weight Steganography and a Capacity–Robustness–Stealth Trilemma

## Article type
Research Paper (Full Length Article)

## Subject / classification
Information security; information hiding — steganography and steganalysis; machine-learning / deep-learning security; model supply-chain security.

---

## Abstract (240 words)
Pre-trained networks circulate as shared artifacts, so a tampered checkpoint on a popular hub reaches every downstream user. Prior weight-steganography attacks hide megabytes in float32 mantissas, yet their reported capacity rests on an untested assumption: that the model is deployed exactly as published. It rarely is. Before reaching an edge or medical (IoMT) device a model is quantized to INT8, pruned and fine-tuned, and we confirm on real networks that these steps erase mantissa-bit payloads, while a faithful MaleficNet survives them only at watermark-scale capacity. We present QuantGhost, a quantization-aware embedding that writes each payload bit into the INT8 code quantization produces, so it survives by construction, and places each weight at a random point in its code bin to keep the mantissa distribution benign; redundant coding carries the payload through pruning and fine-tuning. QuantGhost revises the picture: a 3.2 MB payload survives INT8 on ResNet-50 (99.9999%) and the production PyTorch kernel byte-exactly, at a cost of 0.19 accuracy points on ResNet-56, and evades the Model X-Ray detector on a diverse population (0.50 vs. 0.81 for prior art). A capacity-robustness-stealth trilemma we prove bounds any stealthy, quantization-robust, permutation-invariant channel to the sum of spatial taps, so a faithful neuron permutation erases the high-capacity payload but a 196-bit invariant channel survives to carry an out-of-band key; a proposed sub-quantum residual detector flags the basic attack (AUC 0.98–1.0) but an adaptive attacker reaches chance. Deployment sanitization alone is therefore incomplete, and the trilemma gives defenders a concrete target.

## Keywords
neural network steganography; stegomalware; model supply chain; quantization; steganalysis; deep learning security

## Highlights
- Naive mantissa-bit weight-steganography payloads are fully erased by INT8 quantization on real models; a faithful MaleficNet (spread-spectrum) survives INT8, pruning and fine-tuning but only at watermark-scale capacity.
- QuantGhost, a quantization-aware distribution-preserving embedding, reaches one bit per weight surviving INT8, pruning and fine-tuning at a cost of 0.19 percentage points of accuracy on a real ResNet-56.
- It evades the released Model X-Ray detector on a diverse model zoo (0.50, chance) where the prior attack is detected (0.81).
- We prove a capacity-robustness-stealth trilemma: for the neuron-permutation group, any stealthy, quantization-robust, permutation-invariant channel is bounded by the sum of spatial taps, a value our construction attains.
- A faithful neuron permutation destroys the high-capacity payload, but a 196-bit invariant key channel survives and bootstraps recovery, so permutation-based sanitization is incomplete.

---

## Authors (in order)

**1. Eko Arip Winanto**
- Affiliation a: Faculty of Computing, Universiti Teknologi Malaysia, Johor Bahru 81310, Malaysia
- Affiliation b: Department of Computer Engineering, Universitas Dinamika Bangsa, Jambi 36138, Indonesia
- Email: winanto@graduate.utm.my
- ORCID: 0000-0003-3274-1885

**2. Mohd Yazid Idris**
- Affiliation a: Faculty of Computing, Universiti Teknologi Malaysia, Johor Bahru 81310, Malaysia
- Affiliation c: MaGIC-X (Media and Games Innovation Centre of Excellence), Universiti Teknologi Malaysia, Johor Bahru 81310, Malaysia
- ORCID: 0000-0001-7702-6610

**3. Kurniabudi** — *Corresponding author*
- Affiliation d: Master of Information System, Universitas Dinamika Bangsa, Jambi 36138, Indonesia
- Email: kurniabudi@unama.ac.id
- ORCID: 0000-0001-9073-8858

**4. Febby Tri Ramadhanti**
- Affiliation e: Distance Learning Master's Program in Technology and Vocational Education, Universitas Negeri Yogyakarta, Yogyakarta 55281, Indonesia

**Submitting author:** Eko Arip Winanto (winanto@graduate.utm.my), on behalf of all authors.

---

## CRediT author statement (proposed — please verify/adjust)
- **Eko Arip Winanto:** Conceptualization, Methodology, Software, Investigation, Data curation, Formal analysis, Writing – original draft.
- **Mohd Yazid Idris:** Supervision, Methodology, Resources, Validation, Writing – review & editing.
- **Kurniabudi:** Supervision, Validation, Project administration, Writing – review & editing.
- **Febby Tri Ramadhanti:** Visualization, Writing – review & editing.

## Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This work was funded by Universitas Dinamika Bangsa, Jambi, Indonesia, and supported by MaGIC-X, Universiti Teknologi Malaysia. (No grant number — add if any.)

## Data availability statement
The code and data supporting this study are openly available at https://github.com/robotumbel/quantghost.

## Ethics / responsible disclosure
Offensive research conducted to strengthen defenses. All payloads are encrypted random bytes or an AES key; no live malware is embedded, executed, or distributed. Findings will be disclosed to the authors of the affected detector and defense (Model X-Ray and NeuPerm) prior to publication.

---

## To fill in yourself
- **Suggested reviewers (optional):** 3–5 names + emails + affiliations (needs your judgment; avoid the authors of the attacked works — Model X-Ray / NeuPerm / MaleficNet — as they are affected parties).
- **Opposed reviewers (optional):** any to exclude.
- **Grant/award numbers:** if the funding has any.
- **Declaration-of-generative-AI statement:** if AI tools were used in writing, Elsevier requires a short disclosure (e.g., "During the preparation of this work the author(s) used [tool] in order to [reason]; after using it the author(s) reviewed and edited the content and take full responsibility for the content of the publication.").
