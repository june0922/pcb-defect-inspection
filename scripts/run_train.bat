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
    if errorlevel 1 (
        echo [Error] Preprocessing failed.
        pause
        exit /b 1
    )
)

echo.
echo ========================================================
echo Current Configuration Parameters:
echo ========================================================
python scripts\show_config.py
echo.

:: --- Guard: check preprocessed data exists ---
:: NOTE: This path must match the 'processed' path resolved by config.yaml + get_paths()
if not exist "preprocessed_data\images\train" (
    echo [Warning] preprocessed_data\images\train not found.
    echo           Consider running preprocessing first (select Y at the next prompt).
    echo.
)

set /p run_train="Proceed with training using these parameters? (Y/N, default is N): "
if /I not "%run_train%"=="Y" (
    echo Training cancelled by user.
    pause
    exit /b
)

echo.

set TRAIN_CMD=python src\train.py --config config.yaml
if not exist "runs\train\weights\last.pt" goto start_train

echo ========================================================
echo [Notice] Found a previous training checkpoint (last.pt).
echo ========================================================
set /p do_resume="Do you want to resume training from this checkpoint? (Y/N, default is N): "
if /I "%do_resume%"=="Y" (
    set TRAIN_CMD=python src\train.py --config config.yaml --resume
)
echo.

:start_train
echo ========================================================
echo Starting YOLOv8 Model Training...
echo ========================================================
%TRAIN_CMD%
if errorlevel 1 (
    echo.
    echo [Error] Training encountered an error.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo Training Execution Finished.
echo ========================================================
pause
