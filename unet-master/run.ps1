
# U-Net Project - Install Dependencies and Run
# ==========================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "U-Net Project - Install and Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[1/4] Checking Python environment..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found! Please install Python first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

Write-Host "[2/4] Installing dependencies (TensorFlow, Keras, scikit-image)..." -ForegroundColor Yellow
Write-Host "   This may take several minutes, please wait..." -ForegroundColor Gray
pip install tensorflow scikit-image
if ($LASTEXITCODE -ne 0) {
    Write-Host "   WARNING: Dependencies installation may have issues, but continuing..." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "[3/4] Running U-Net training and prediction..." -ForegroundColor Yellow
python main.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to run! Please check the error messages above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

Write-Host "[4/4] Done!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Predictions saved in data\membrane\test\" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
