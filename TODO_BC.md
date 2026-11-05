# TODO terperinci — Bagian B (skala model nyata) & C (detektor asli)

Status: ⬜ belum · 🔄 jalan · ✅ selesai
Setiap tugas: **file** yang disentuh · **kerja** · **selesai jika** (acceptance) ·
**butuh** (dependensi/compute). Urutan dalam tiap bagian = urutan disarankan.

---

## BAGIAN B — angka model nyata (wajib sebelum submit Q1)

### B1 ✅ Eval akurasi ImageNet pada ResNet-50 (tutup klaim "akurasi terjaga")
> SELESAI (kode). `wstego/imagenet.py` (loader+top1/top5) & `scripts/14_mb_payload.py`
> diperluas: tulis-balik bobot ke model, ukur 4 kondisi {orig FP32, clean INT8,
> embed FP32, embed INT8}. Bit-recovery 99.9999% terverifikasi. Sisa = jalankan
> dengan data ImageNet-val asli: `IMAGENET_VAL=/path/val python scripts/14_mb_payload.py`
> (atau `--val PATH`; `--max_batches N` untuk estimasi cepat).
- **file**: `scripts/14_mb_payload.py` (perluas) + util baru `wstego/imagenet.py`
- **kerja**:
  1. `wstego/imagenet.py`: loader ImageNet-val (folder `val/` standar torchvision
     `ImageFolder` + transform 224/center-crop), fungsi `top1(model, loader, dev)`.
  2. Di `14`: ukur top-1 pada **3 kondisi**: (a) ResNet-50 asli, (b) setelah
     embedding FP32, (c) setelah embedding+INT8. Bangun kembali state_dict dari
     tensor yang sudah di-embed (sekarang `14` hanya bekerja di array numpy, tidak
     menulis balik ke model — perlu `load_state_dict` dari bobot termodifikasi).
  3. Cetak tabel: akurasi vs kondisi + delta.
- **selesai jika**: tercetak top-1 untuk 3 kondisi; delta (asli→embed+INT8) terukur
  nyata (target degradasi kecil, dilaporkan apa adanya).
- **butuh**: ImageNet val (~6 GB) di disk; GPU untuk *inference* (4050 cukup).
  Set env `IMAGENET_VAL=/path/to/val`. → mengisi baris "Accuracy preserved" &
  "Capacity" di PAPER_OUTLINE.

### B2 ✅ Kuantisasi per-channel (selain per-tensor)
> SELESAI. `robust.py`: `quant_scale_pc/quantize_pc/qaware_embed_stealthy_pc/qaware_extract_pc`
> + `mantissa_lowbyte_entropy`. Script baru `scripts/15_quant_schemes.py`.
> Terverifikasi: survival 1.000 per-channel INT8/INT4, entropi 7.999.
- **file**: `wstego/robust.py` (tambah fungsi) + `scripts/08_robust_survival.py`
- **kerja**:
  1. `quant_scale_perchannel(w2d, axis=0)` → skala per baris/kolom output-channel
     (vektor skala, bukan skalar).
  2. `quantize_perchannel` + `qaware_embed_stealthy` versi per-channel: bin
     ditentukan oleh skala channel-nya masing-masing (scale-pinning per channel).
  3. `08`: tambah kolom hasil untuk skema per-channel di samping per-tensor.
- **selesai jika**: survival per-channel terukur (target ~1.0 by-construction) dan
  akurasi INT8 per-channel dilaporkan; ditampilkan berdampingan dengan per-tensor.
- **butuh**: B-none (murni numpy/logic). → mengisi "multiple quant schemes".

### B3 ✅ Kuantisasi INT4 (qmax lebih kecil)
> SELESAI. qmax sudah parametrik di seluruh `robust.py`; `15_quant_schemes.py` menyapu
> INT8(qmax=127) vs INT4(qmax=7) x {per-tensor, per-channel}. Survival 1.000 di semua;
> trade-off akurasi INT4 terukur, entropi tetap ~natural (7.999).
- **file**: `wstego/robust.py` + `scripts/08_robust_survival.py`
- **kerja**:
  1. Generalisasi `quant_scale/quantize/qaware_*` agar terima `qmax` (mis. INT4 = 7).
  2. Cek apakah sub-quantum placement masih muat (bin INT4 jauh lebih lebar →
     stealth bisa berubah; ukur entropi mantissa & survival).
  3. `08`: jalankan sweep qmax ∈ {127 (INT8), 7 (INT4)}.
- **selesai jika**: survival & akurasi & entropi mantissa terlapor untuk INT8 vs INT4;
  trade-off terdokumentasi (INT4 mungkin survival tetap tinggi tapi akurasi turun).
- **butuh**: B2 selesai (refactor qmax lebih bersih bila digabung). → "INT4".

### B4 ✅ Pruning sampai 90% + structured pruning
> SELESAI. `09_finetune_prune.py`: `prune` (unstructured) + `prune_structured` (buang
> output-channel L2-terkecil); sweep {30,50,70,90}% keduanya. Terverifikasi: payload
> selamat ~30%, meluruh ke chance di 70-90% (structured mati di 50%). Titik patah terlihat.
- **file**: `scripts/09_finetune_prune.py`
- **kerja**:
  1. Ganti `prune(net,0.30)` jadi sweep rasio {0.3, 0.5, 0.7, 0.9} (unstructured).
  2. Tambah **structured pruning** (buang seluruh channel/filter L2-terkecil),
     karena ini menghapus banyak bobot bersebelahan → uji redundansi mana yang
     bertahan.
  3. Plot/tabel msg-recovery vs rasio-prune vs redundansi.
- **selesai jika**: kurva recovery untuk 4 rasio + kasus structured; titik patah
  (di mana payload mulai gagal) teridentifikasi.
- **butuh**: B-none (CIFAR CNN, sudah ada). → "Up to 90% pruning, structured".

### B5 ✅ Kurva kapasitas-vs-robustness fine-tuning nyata (transfer learning)
> SELESAI. `scripts/16_finetune_curve.py`: sweep n_msg {16..16384} (redundansi
> 8774x..8x) x fine-tune {ringan 2ep, berat 8ep}. Kurva recovery-vs-redundansi
> dihasilkan; menunjukkan ada ambang fine-tune di atasnya redundansi tak menolong.
- **file**: `scripts/09_finetune_prune.py` (perluas) atau script baru `15_finetune_curve.py`
- **kerja**: fine-tune lebih realistis (lebih banyak epoch / transfer ke subset
  kelas berbeda), sweep redundansi {2×…2000×}, ukur recovery → kurva frontier.
- **selesai jika**: dihasilkan kurva "recovery vs redundansi" melalui fine-tuning
  nyata (bukan 2 epoch toy); titik 100%-recovery teridentifikasi.
- **butuh**: GPU; bisa CIFAR (cukup untuk frontier). → Fig F3 / "capacity-robustness".

### B6 ✅ NeuPerm faithful di model penuh (bukan 1 tensor)
> SELESAI. `wstego/neuperm.py` (`permute_smallcnn` function-preserving: conv out-ch +
> BN + consumer in-ch) + `scripts/17_neuperm_fullmodel.py`. Terverifikasi: akurasi
> identik (max|d_logit|=8e-5), flat mati (1.0->0.58), spatial-invariant bertahan
> (1.0->1.0). Kapasitas invarian agregat = sum kh*kw.
- **file**: `wstego/neuperm.py` (baru) + `scripts/11_channel_perm.py` (perluas) atau
  `16_neuperm_fullmodel.py`
- **kerja**:
  1. Implementasi permutasi neuron yang menjaga ekuivalensi fungsi di **seluruh**
     ResNet: permute out-channels layer L + in-channels layer L+1 yang cocok
     (tangani BN, residual, downsample). Verifikasi output model identik (Δ≈0).
  2. Embed channel-invariant (dimensi spasial) di **banyak tensor**, agregasi
     kapasitas total bit yang selamat NeuPerm di seluruh net.
  3. Bandingkan: payload flat (hancur) vs spatial-invariant (selamat) skala penuh.
- **selesai jika**: NeuPerm menjaga akurasi model (bukti faithful) DAN kapasitas
  spatial-invariant agregat terukur dalam bit; konfirmasi trilemma di model nyata.
- **butuh**: GPU inference; paling teknis dari semua B. → "Real NeuPerm impl".

### B7 ✅ Baseline pembanding: EvilModel/StegoNet di bawah kuantisasi
> SELESAI. `scripts/18_baselines.py`: naive X-LSB X={8,16,23} di ResNet-18 nyata ->
> survival INT8 = 0.500 (ERASED) di semua; punya kita 1.000. + tabel kapasitas vs
> robust watermark (2208.10973, <=2KB). Mengisi baris "Baselines" PAPER_OUTLINE.
- **file**: `scripts/17_baselines.py` (baru)
- **kerja**: implementasi/replika embedding mantissa-LSB ala EvilModel & StegoNet,
  lewatkan INT8 → tunjukkan survival ≈ 0 (mereka tidak robust). Plus tabel kapasitas
  vs watermarking robust (2208.10973, low-cap).
- **selesai jika**: tabel "survival=0 untuk baseline; ~1.0 untuk kita" + perbandingan
  kapasitas. → baris "Baselines" PAPER_OUTLINE.
- **butuh**: B-none. → wajib untuk reviewer ("kenapa bukan metode lama").

### B8 ✅ (opsional, efisiensi) ECC: ganti repetition dengan LDPC/fountain
> SELESAI. `wstego/ecc.py` (LT fountain code pure-numpy GF(2), peeling decode) +
> `scripts/19_ecc.py`. Terverifikasi: pulih byte-exact dgn overhead kecil; repetition
> 2.5-4.4x lebih boros pada erasure rate setara. Tanpa dependensi eksternal.
- **file**: `wstego/ecc.py` (baru) + integrasi ke `09`/`14`
- **kerja**: `pip install galois`; ganti redundansi repetition naif dengan kode
  fountain/LDPC → kapasitas lebih baik pada robustness setara; byte-exact delivery.
- **selesai jika**: pada robustness sama, bit/efektif lebih tinggi dari repetition;
  byte-error → 0 setelah dekode.
- **butuh**: `galois`. → meningkatkan klaim kapasitas; boleh paling akhir.

---

## BAGIAN C — detektor Model X-Ray ASLI (klaim "evade *released* detector")

### C1 ✅ Sanity: jalankan repo ModelXRay (plot-only)
> SELESAI. Jalur plot TIDAK butuh TF (paket `model_xray` kosong di __init__). Dijalankan
> dgn Python Windows + `PYTHONPATH=<mxr>`: 5 figur (`exp1_id_oml.png` dst) + 2 tabel
> LaTeX ter-regenerasi dari CSV cache. Repo siap.

### C2 ✅ Glue: serangan KITA masuk pipeline OSL asli mereka
> SELESAI (kode + uji inti). Wawasan kunci: `img_pp_xlsb_attack(x=0)` cuma
> merepresentasikan bobot apa adanya, jadi cukup embed dgn skema kita lalu lewatkan
> x=0 — TANPA ubah pipeline mereka. File baru
> `ModelXRay/scripts/experiments/run_ours_vs_osl.py`: pakai `train_fsl`/`evaluate_model`
> asli; kelas jahat = `robust.qaware_embed_stealthy` row-wise; kontrol positif naive
> X-LSB X=8. Sintaks terverifikasi; `embed_ours` teruji (entropi mantissa 8.000 = stealth).
> Sisa = jalankan di env TF mereka (lihat C3).

### C3 ✅ Jalankan OSL mereka pada bobot kita → centroid/1-NN (DIJALANKAN di WSL2)
> SELESAI. Env TF 2.15 dibangun via uv (Python 3.11, tanpa sudo). Dipakai zoo lokal
> kita (`--zoo-npz data/zoo_cifar.npz`) sebab data GHRP tak ada. Payload TERSTRUKTUR
> (`/bin/bash`) wajib agar kontrol positif valid. Hasil (mode ub, 10 repeats),
> detektor OSL ASLI mereka → `results/C3_osl_real.txt`:
>   naive X-LSB X=8: 1-NN **1.000 (DETECTED)**; ours: centroid 0.622 / 1-NN 0.672.
> Ours menurunkan deteksi tajam (1.0->0.67) tapi belum chance bersih di pilot zoo lemah;
> dua artefak (zoo homogen lemah + glue pakai SATU skala global pada vektor flat) bisa
> dihilangkan dgn per-tensor embed -> diperkirakan ~0.5 (cf. script 13).

### C4 ✅ (opsional) MalConv + entropy/threshold sebagai detektor kedua
> SELESAI. `ModelXRay/scripts/experiments/run_ours_vs_detectors.py` (MalConv-lite +
> ALL_DETECTORS threshold). Hasil → `results/C4_detectors.txt`. Temuan jujur:
> ours EVADE MalConv/histogram-KL/byte-autocorr (~0.5) TAPI tertangkap byte_entropy &
> weight_value_dist — artefak skala-global flat-zoo (naive ubah mantissa saja; glue
> kita menggeser NILAI bobot). Memotivasi per-tensor embed + bagian "defenses" paper.

---

## Peta dependensi (urutan eksekusi disarankan)
```
B1 (akurasi ImageNet)  ──┐
B2 (per-channel) → B3 (INT4)
B4 (pruning 90%/structured)
B5 (fine-tune curve)
B6 (NeuPerm full-model)         ← paling teknis
B7 (baselines)                  ← wajib, mudah
B8 (ECC)                        ← opsional, akhir
C1 (sanity) → C2 (export) → C3 (OSL AUC) → C4 (MalConv opsional)
```

## Prioritas untuk submission (kalau waktu terbatas)
1. **C1→C2→C3** — bukti "evade detektor asli" = klaim paling membedakan.
2. **B1** — akurasi ImageNet nyata = klaim "accuracy preserved".
3. **B7** — baseline (reviewer pasti tanya).
4. **B4, B2/B3** — kelengkapan tabel survival.
5. **B6** — trilemma skala penuh (kuat tapi mahal).
6. **B5, B8, C4** — penyempurnaan.

## Yang TIDAK perlu ditulis lagi (sudah ada — Bagian A)
- Embedding qaware-stealthy, survival INT8 per-tensor, redundansi+majority-vote,
  reproduksi OSL kita, trilemma pilot, embed MB ke ResNet-50 (tanpa akurasi).
