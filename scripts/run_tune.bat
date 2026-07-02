@echo off
chcp 65001 >nul
title YOLOv8 Hyperparameter Tuning

:: Change working directory to the project root
cd /d "%~dp0.."

echo ========================================================
echo  YOLOv8 Hyperparameter Tuning
echo  Current Directory: %CD%
echo ========================================================
echo.

:: --- Preprocessing step (optional) ---
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
    echo [Error] preprocessed_data\images\train not found.
    echo         Please run preprocessing first.
    pause
    exit /b 1
)

:: --- Guard: check model weights exist ---
if not exist "weights\yolov8n.pt" (
    echo [Warning] weights\yolov8n.pt not found.
    echo           Make sure the base model weight file exists before tuning.
    echo.
)

:: --- Confirm tuning ---
set /p run_tune="Proceed with hyperparameter tuning? (Y/N, default is N): "
if /I not "%run_tune%"=="Y" (
    echo Tuning cancelled by user.
    pause
    exit /b 0
)

echo.

:: --- Guard: check previous tune results ---
if not exist "runs\tune" goto start_tune

echo ========================================================
echo [Notice] Previous tuning results found in runs\tune\.
echo   O = Overwrite (start tuning from scratch)
echo   N = Cancel (keep existing results)
echo ========================================================
set /p tune_action="Your choice (O/N, default is N): "

if /I "%tune_action%"=="O" (
    echo Overwriting previous tuning results...
    goto start_tune
)
echo Tuning cancelled by user.
pause
exit /b 0

:start_tune
echo.
echo ========================================================
echo Starting YOLOv8 Hyperparameter Tuning...
echo [Note] This process runs multiple short training cycles.
echo        Estimated time: iterations x epochs duration.
echo ========================================================
python src\tune.py --config config.yaml
if errorlevel 1 (
    echo.
    echo [Error] Hyperparameter tuning encountered an error.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo Hyperparameter Tuning Finished.
echo Results saved to: runs\tune\
echo Best hyperparameters: runs\tune\best_hyperparameters.yaml
echo ========================================================
pause
