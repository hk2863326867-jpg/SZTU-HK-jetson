
@echo off
chcp 65001 >nul
echo ========================================
echo Person Segmentation - Setup
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Creating sample data...
python download_sample_data.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to create sample data
    echo Please make sure you have numpy and scikit-image installed
    pause
    exit /b 1
)

echo.
echo [2/3] Training model...
python train_person.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed
    pause
    exit /b 1
)

echo.
echo [3/3] Making predictions...
python predict_person.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Prediction failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo ALL DONE!
echo ========================================
echo.
echo Check results in dataset/test/
echo.
pause
