
@echo off
chcp 65001 >nul
echo ========================================
echo U-Net Project - Install and Run
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python first
    pause
    exit /b 1
)
echo.

echo [2/4] Installing dependencies (TensorFlow, Keras, scikit-image)...
echo This may take several minutes, please wait...
pip install tensorflow scikit-image
if %errorlevel% neq 0 (
    echo WARNING: Dependencies installation may have issues, but continuing...
)
echo.

echo [3/4] Running U-Net training and prediction...
python main.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to run! Please check the error messages above
    pause
    exit /b 1
)
echo.

echo [4/4] Done!
echo ========================================
echo Predictions saved in data\membrane\test\
echo ========================================
echo.
pause
