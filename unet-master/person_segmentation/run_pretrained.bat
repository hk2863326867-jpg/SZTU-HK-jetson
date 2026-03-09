@echo off
chcp 65001 >nul
echo ========================================
echo Person Segmentation - Pre-trained Model
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Running prediction with pre-trained method...
python predict_pretrained.py

echo.
echo ========================================
echo Check results in dataset/test/
echo Look for files ending with _pretrained.png
echo ========================================
echo.
pause
