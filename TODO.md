# Q1 Roadmap — Evasive Weight-Steganography (alur proyek)

Status: ✅ selesai · 🔄 sedang jalan · ⬜ belum · ❓ perlu keputusan

---

## FASE 0 — Setup & uji kelayakan ✅
- [x] Pilih pipeline PyTorch murni (Windows-friendly, GPU lokal)
- [x] Modul inti: `embed.py` (X-LSB), `gf_image.py` (GF byte-image + fourpart),
      `detector.py` (SRNet), `osl.py` (few-shot metric learning), `zoo.py`
- [x] Smoke-test: round-trip embedding exact, perturbasi 3e-5 (akurasi terjaga)

## FASE 1 — Reproduksi & pahami baseline Model X-Ray ✅
- [x] Latih zoo CNN homogen (carrier) — `03_train_zoo.py`
- [x] Temuan kunci: byte mantissa-low FP32 SUDAH seragam → embedding ACAK tak terdeteksi
- [x] Konfirmasi dari docs mereka: hasil headline pakai **payload malware terstruktur**
- [x] Grounding: pipeline kita reproduksi kurva deteksi (X8→1.0) — `07_grounding.py`
- [x] Evasi dasar terbukti: payload dienkripsi → deteksi jatuh 1.0→0.50

## FASE 2 — Tentukan sudut kontribusi Q1 ✅
- [x] Sadar: "evasi via enkripsi" pada FP32 = efektif tapi TRIVIAL → dibuang
- [x] Cek INT8: terstruktur di nilai, tapi LSB nyaris acak → pivot lemah, dibuang
- [x] **Cek Angle D (robust thru quantization): GAP TERBUKTI** — payload X-LSB
      100% terhapus kuantisasi di semua X; satu-satunya yg selamat (X tinggi)
      hancurkan akurasi. → **ARAH Q1 TERPILIH: ANGLE D**
- [ ] ⬜ Cek literatur: pastikan "HIGH-CAPACITY quantization-robust weight stego"
      belum dikerjakan (watermarking robust ADA, tapi low-capacity identity —
      payload MB-skala beda). Hindari dead-end "already done".

## FASE 3 — Bangun serangan robust (inti kebaruan: ANGLE D) 🔄
- [x] Literatur dicek: gap = high-capacity + robust + defeat-defenses (novelty moderat)
- [x] **Quantization-aware embedding survive 100%** (scale-pinning): `robust.py` +
      `08` → naif 0.50, qaware **1.000 survival** lewat INT8, akurasi ~−1%.
- [x] **Tegangan survival-vs-stealth DIPECAHKAN** (`qaware_embed_stealthy`): taruh
      bobot di titik ACAK dalam bin INT8 yg benar → survival 1.0 + mantissa-entropy
      7.999 (=natural, lolos Model X-Ray) + grid-frac 0.0002. INTI KONTRIBUSI.
- [x] **Survival lewat FINE-TUNING & PRUNING via redundansi+majority-vote**
      (`09_finetune_prune.py`): quantize & prune-30% = 1.000 di semua redundansi;
      fine-tune = transform tersulit, tapi redundansi tinggi menang (137x→0.80,
      2193x→1.00). Frontier kapasitas-vs-robustness terdemonstrasi.
- [x] **NeuPerm dianalisis** (`10_neuperm.py`): positional hancur (1.0→0.50);
      sorted-canon gagal → temuan TRILEMMA {high-cap, perm-invariant, stealthy-robust}
      pilih max 2. DOF perm-invariant = histogram (~256 bin) = low-cap.
- [x] **Channel-perm RESOLVED** (`11_channel_perm.py`): flat hancur (0.47);
      spatial-invariant SURVIVE NeuPerm (1.000) tapi low-cap (9 bit/tensor).
      => cerita lengkap: high-cap survive pipeline tapi mati NeuPerm; low-cap
      invariant survive SEMUA. ARC KONSEPTUAL PAPER SELESAI.

## ARC KONSEPTUAL SELESAI ✅ — sisa = rigor empiris (~60%, butuh WSL2/cloud)
## FASE 4 — Evaluasi menyeluruh ⬜ (lihat di bawah)
- [ ] ECC efisien (LDPC/fountain) ganti repetition naif → kapasitas lebih baik
- [ ] Konfirmasi stealth EMPIRIS lawan OSL Model X-Ray (bukan cuma entropy)
- [ ] Formalkan model ancaman + metrik lengkap
- [ ] Skala ke MB di model nyata (resnet/LLM) + jaga akurasi FP32

## FASE 4 — Evaluasi menyeluruh ⬜
- [ ] Survival juga melewati: fine-tuning, pruning, distillation (bukan cuma quant)
- [ ] Stealth vs detektor: Model X-Ray (OSL), threshold/entropi, MalConv-lite
- [ ] Beragam arsitektur & skema kuantisasi (per-tensor/per-channel, INT8/INT4)
- [ ] Ablation: kapasitas vs survival vs degradasi akurasi vs deteksi

## FASE 5 — Pertahanan (paruh blue-team, perkuat paper) ⬜
- [ ] Usulkan deteksi/mitigasi payload robust (mis. quantization sbg sanitasi —
      tunjukkan kapan gagal)
- [ ] Trade-off & batas

## FASE 6 — Penulisan & submit ⬜
- [ ] Pilih jurnal/konferensi Q1 (target keamanan/ML)
- [ ] Tulis paper + framing etis (red-team + responsible disclosure ke penulis Model X-Ray)
- [ ] Reproducibility package

---

### Catatan penting
- Disiplin: **selalu uji kelayakan + verifikasi literatur sebelum komit menulis**
  (sudah menyelamatkan kita 2× dari premis salah).
- GPU lambat ~25× di Windows (overhead WDDM) → eksperimen berat sebaiknya nanti di
  Linux/WSL2 atau `torch.compile`.
- Semua angka HARUS real (tanpa fabrikasi), konsisten dengan prinsip Anda.
