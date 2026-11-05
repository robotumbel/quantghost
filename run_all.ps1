# run_all.ps1 — Bagian A (pilot) + Bagian B (skala) end-to-end.
# Jalankan dari Windows (Python dgn torch+CUDA). Tiap script direkam ke results\.
# Bagian C (detektor Model X-Ray asli) ada di ModelXRay\run_all_C.sh (WSL2/TF).
#
#   cd D:\Code-PhD\Projet_2026\q1_weightstego
#   .\run_all.ps1                      # semua A+B
#   $env:IMAGENET_VAL="D:\data\imagenet\val"; .\run_all.ps1   # + akurasi B1 nyata

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force results | Out-Null

# (file, args, label) — urutan menghormati dependensi (03 bikin zoo utk 04/12/13).
$steps = @(
  @("00_smoke.py",            "",                          "A0_smoke"),
  @("03_train_zoo.py",        "--n_models 60 --epochs 10", "A1_zoo"),
  @("04_osl_baseline.py",     "",                          "A2_osl_baseline"),
  @("07_grounding.py",        "",                          "A3_grounding"),
  @("08_robust_survival.py",  "",                          "A4_survival_quant"),
  @("09_finetune_prune.py",   "",                          "A5_finetune_prune"),
  @("10_neuperm.py",          "",                          "A6_neuperm"),
  @("11_channel_perm.py",     "",                          "A7_channel_perm"),
  @("12_stealth_eval.py",     "",                          "A8_stealth_cnn"),
  @("13_osl_stealth.py",      "",                          "A9_stealth_osl"),
  @("15_quant_schemes.py",    "",                          "B2B3_quant_schemes"),
  @("16_finetune_curve.py",   "",                          "B5_finetune_curve"),
  @("17_neuperm_fullmodel.py","",                          "B6_neuperm_full"),
  @("18_baselines.py",        "",                          "B7_baselines"),
  @("19_ecc.py",              "",                          "B8_ecc"),
  @("14_mb_payload.py",       "",                          "B1_mb_payload_accuracy")
)

$summary = @()
foreach ($s in $steps) {
  $file, $args, $label = $s
  $out = "results\$label.txt"
  Write-Host "`n===== [$label] python scripts\$file $args =====" -ForegroundColor Cyan
  $cmd = "scripts\$file"
  if ($args) { python $cmd $args.Split(" ") 2>&1 | Tee-Object -FilePath $out }
  else       { python $cmd            2>&1 | Tee-Object -FilePath $out }
  $ok = ($LASTEXITCODE -eq 0)
  $summary += [pscustomobject]@{ step = $label; ok = $ok; log = $out }
  if (-not $ok) { Write-Host "!! [$label] exit $LASTEXITCODE (lanjut)" -ForegroundColor Yellow }
}

Write-Host "`n=================== RINGKASAN A+B ===================" -ForegroundColor Green
$summary | Format-Table -AutoSize
Write-Host "Semua log di results\ . Lanjut Bagian C: WSL2 -> bash ModelXRay/run_all_C.sh"
