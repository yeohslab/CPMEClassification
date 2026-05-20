# Train + export pipeline
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location "$PSScriptRoot\..\src"

if (-not (Test-Path "..\splits\emotion.json")) {
    python prepare_dataset.py
}

Write-Host "=== Train emotion ==="
python train_emotion.py --epochs 3 --batch_size 128 --log_every 100

Write-Host "=== Train MBTI ==="
python train_mbti.py --epochs 10 --batch_size 24 --max_posts 16 --log_every 8

Write-Host "=== Evaluate ==="
python evaluate.py

Write-Host "=== Export ONNX ==="
python export_onnx.py

Write-Host "=== Done ==="
