@echo off
chcp 65001 >nul
title YOLOv8 Model Training

:: Change working directory to the project root
cd /d "%~dp0.."

echo ========================================================
echo Current Directory: %CD%
echo ========================================================
echo.

set /p run_pre="Run preprocessing first? (Y/N, default is N): "
if /I "%run_pre%"=="Y" (
    echo.
    echo ========================================================
    echo Running Preprocessing...
    echo ========================================================
    python src\preprocess.py --config config.yaml
)

echo.
echo ========================================================
echo Starting YOLOv8 Model Training...
echo ========================================================
python src\train.py --config config.yaml

echo.
echo ========================================================
echo Training Execution Finished.
echo ========================================================
pause
