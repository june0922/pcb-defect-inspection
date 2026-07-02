@echo off
chcp 65001 >nul
title YOLOv8 K-Fold Cross Validation

:: Change working directory to the project root
cd /d "%~dp0.."

echo ========================================================
echo  YOLOv8 K-Fold Cross Validation
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
if not exist "preprocessed_data\images\train" (
    echo [Error] preprocessed_data\images\train not found.
    echo         Please run preprocessing first.
    pause
    exit /b 1
)

:: --- Confirm K-Fold training ---
set /p run_kfold="Proceed with K-Fold cross validation? (Y/N, default is N): "
if /I not "%run_kfold%"=="Y" (
    echo K-Fold training cancelled by user.
    pause
    exit /b 0
)

echo.

:: --- Guard: check previous kfold results ---
if not exist "runs\kfold" goto start_kfold

echo ========================================================
echo [Notice] Previous K-Fold results found in runs\kfold\.
echo          Existing fold directories will be overwritten (exist_ok=True).
echo ========================================================
set /p confirm_overwrite="Continue and overwrite existing results? (Y/N, default is N): "
if /I not "%confirm_overwrite%"=="Y" (
    echo K-Fold training cancelled by user.
    pause
    exit /b 0
)
echo.

:start_kfold
echo ========================================================
echo Starting YOLOv8 K-Fold Cross Validation...
echo ========================================================
python src\train_kfold.py --config config.yaml
if errorlevel 1 (
    echo.
    echo [Error] K-Fold training encountered an error.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo K-Fold Training Execution Finished.
echo Results saved to: runs\kfold\
echo Weights saved to: weights\best_fold_1.pt ~ best_fold_N.pt
echo ========================================================
pause
