# RUNBOOK — menjalankan pipeline penuh (WSL2/Linux GPU)

Tujuan: dari "smoke test" → **full run** yang menghasilkan SEMUA angka di
`PAPER_OUTLINE.md`. Penting: hampir semua script hanya `print` ke layar (tidak
menulis CSV), jadi setiap langkah direkam ke `results/` dengan `tee`.

---

## ⭐ JALANKAN SEMUA (A → C) — 2 perintah

Master-runner sudah disediakan. Bagian A+B di Windows (GPU lokal), Bagian C di WSL2
(env TF 2.15 dibangun otomatis via uv, tanpa sudo). Semua hasil → `results/`.

```powershell
# Bagian A (pilot) + B (skala) — Windows PowerShell, Python dgn torch+CUDA
cd D:\Code-PhD\Projet_2026\q1_weightstego
.\run_all.ps1
#   (opsional, untuk angka akurasi ImageNet B1):
#   $env:IMAGENET_VAL="D:\data\imagenet\val"; .\run_all.ps1
```
```powershell
# Bagian C — detektor Model X-Ray ASLI (env TF dibangun sekali, lalu C1+C3+C4)
wsl -d Ubuntu -e bash -lc "bash /mnt/d/Code-PhD/Projet_2026/ModelXRay/run_all_C.sh"
```

Output kunci: `results/A*_*.txt`, `results/B*_*.txt` (A+B) dan
`results/C3_osl_real.txt`, `results/C4_detectors.txt` (C). Rincian per-langkah +
penjelasan ada di bawah.

---

Asumsi: sudah di WSL2 Ubuntu, `.venv` aktif, `pip install -r requirements.txt`
sudah jalan, dan `python scripts/00_smoke.py` lulus.

```bash
cd /mnt/d/Code-PhD/Projet_2026/q1_weightstego
source .venv/bin/activate
nvidia-smi          # pastikan RTX 4050 kebaca (kalau gagal: update driver NVIDIA Windows)
mkdir -p results    # tempat menyimpan semua log angka
```

Install ekstra yang dipakai langkah skala (sekali saja):
```bash
pip install timm scikit-learn pillow   # galois nanti kalau pakai ECC (langkah lanjutan)
```

---

## BAGIAN A — Pipeline pilot LENGKAP (semua di WSL2 lokal, ~1–2 jam)

Jalankan berurutan. Setiap baris merekam output ke `results/`.

### A0. Smoke (sudah lulus, ulang untuk pastikan GPU)
```bash
python scripts/00_smoke.py | tee results/00_smoke.txt
```

### A1. Latih zoo carrier (prasyarat untuk 04, 12, 13)
`data/zoo_cifar.npz` sudah ada (30 model). Untuk angka paper, latih ulang lebih besar:
```bash
python scripts/03_train_zoo.py --n_models 60 --epochs 10 | tee results/03_zoo.txt
```
Output: menulis ulang `data/zoo_cifar.npz`. (Kalau mau cepat dulu, lewati — file lama dipakai.)

### A2. Reproduksi detektor OSL: kurva deteksi-vs-X  → motivasi & Fig baseline
```bash
python scripts/04_osl_baseline.py | tee results/04_osl_baseline.txt
# (opsional, lebih kuat) python scripts/04_osl_baseline.py --per_model 60 --steps 500 | tee results/04_osl_baseline.txt
```
Harapan: deteksi rendah X<8, melonjak ~0.9 di X=8 (meniru `fsl_osl.csv` mereka).

### A3. Survival lewat kuantisasi  → Tabel "Survives quant"
```bash
python scripts/08_robust_survival.py | tee results/08_survival.txt
```
Harapan: naive ~0.5 (hancur), qaware ~1.0 (selamat), akurasi INT8 ~−1%.

### A4. Survival lewat fine-tune & prune  → Tabel "Survives prune/fine-tune" + F3
```bash
python scripts/09_finetune_prune.py | tee results/09_finetune_prune.txt
```

### A5. NeuPerm (trilemma, bagian 1)
```bash
python scripts/10_neuperm.py | tee results/10_neuperm.txt        # download resnet18 sekali
```

### A6. Channel-perm NeuPerm faithful (trilemma, bagian 2)
```bash
python scripts/11_channel_perm.py | tee results/11_channel_perm.txt
```

### A7. Stealth vs detektor CNN sederhana  → kontrol klaim #2
```bash
python scripts/12_stealth_eval.py | tee results/12_stealth.txt
```

### A8. Stealth vs metode OSL ASLI (reproduksi)  → Fig F4 / Tabel stealth
```bash
python scripts/13_osl_stealth.py | tee results/13_osl_stealth.txt
```
Harapan: naive terstruktur DETECTED (~1.0); qaware-stealthy evaded (~0.5).

### A9. Kapasitas MB di model nyata (ResNet-50)  → Tabel kapasitas
```bash
python scripts/14_mb_payload.py | tee results/14_mb_payload.txt   # download resnet50 sekali
```
Harapan: ~3.2 MB payload, bit-recovery ~99.9999% setelah INT8.

> **Setelah Bagian A selesai**, semua angka di `PAPER_OUTLINE.md` (kolom "have")
> ter-reproduksi & terekam di `results/`. Pilot pipeline = SELESAI.

---

## BAGIAN B — Naik skala "model nyata" (untuk angka yang tahan reviewer Q1)

Bagian A pakai CNN toy + CIFAR. Reviewer Q1 minta model nyata. Yang perlu di-upgrade:

1. **Akurasi ResNet-50 ImageNet** sebelum vs sesudah embedding+INT8.
   - `14_mb_payload.py` sudah embed ke ResNet-50 tapi belum mengukur akurasi.
   - Perlu: tambah evaluasi top-1 di ImageNet val (butuh dataset val ~6 GB).
   - GPU 6 GB cukup untuk *inference* (bukan training). Kalau lambat → sewa A10/A100 1 jam.
2. **Skema kuantisasi lebih banyak**: per-channel & INT4 (sekarang per-tensor INT8).
3. **Pruning sampai 90% + structured pruning** (sekarang 30% unstructured).
4. **NeuPerm asli di model penuh** (sekarang per-tensor / 1 juta bobot).

Ini perlu sedikit penambahan kode (bukan fitur baru, hanya scaling argumen + loader
ImageNet). Kerjakan setelah Bagian A bersih.

---

## BAGIAN C — Detektor Model X-Ray ASLI (klaim "evade the *released* detector")

`plot_all.py` BUKAN langkah akhir — itu cuma sanity check repo mereka.

### C1. Sanity: pastikan repo mereka jalan (plot-only, detik, no GPU)
```bash
cd /mnt/d/Code-PhD/Projet_2026/ModelXRay
python -m venv .venv-mxr && source .venv-mxr/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/plots/plot_all.py        # regen figur paper MEREKA dari CSV cache → sanity OK
```
Kalau ini menghasilkan figur tanpa error → repo siap. **Tapi ini belum menyentuh
bobot Anda.**

### C2. (BELUM ADA — perlu ditulis) Arahkan detektor mereka ke bobot Anda
Langkah inti yang membuktikan klaim "evade released detector":
1. Export bobot qaware-stealthy Anda ke format yang loader mereka baca
   (`model_xray/` punya loader bobot → cek `SETUP.md` env vars zoo).
2. Masukkan sebagai sampel "stego" ke pipeline OSL exp1 mereka, benign sebagai negatif.
3. Jalankan deteksi few-shot mereka → harusnya AUC≈0.5 (vs ~1.0 untuk naive X-LSB).

Ini glue code (~1 file) yang belum dibuat. Begitu jadi, jalankan end-to-end exp1:
```bash
# contoh kerangka (sesuaikan ke API mereka):
# python scripts/experiments/run_exp1_scz_oml.py --quick   # smoke dulu
```

---

## Ringkasan status setelah runbook ini
- Bagian A selesai → reproduksi penuh angka pilot (cukup untuk MULAI menulis paper).
- Bagian B & C → angka skala-nyata + detektor asli (wajib sebelum submit Q1).
- Tulisan paper (Intro/Related/Method/Trilemma) bisa jalan PARALEL dengan B & C.
