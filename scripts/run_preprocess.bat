@echo off
chcp 65001 >nul
title Data Preprocessing

:: Change working directory to the project root
cd /d "%~dp0.."

:: Activate virtual environment if it exists
if exist "%~dp0..\venv\Scripts\activate.bat" (
    call "%~dp0..\venv\Scripts\activate.bat"
)

echo ========================================================
echo Current Directory: %CD%
echo ========================================================
echo.

echo ========================================================
echo Current Configuration Parameters:
echo ========================================================
python scripts\show_config.py
echo.

set /p run_pre="Proceed with preprocessing? (Y/N, default is N): "
if /I not "%run_pre%"=="Y" (
    echo Preprocessing cancelled by user.
    pause
    exit /b 0
)

echo.
echo ========================================================
echo Running Preprocessing...
echo ========================================================
python src\preprocess.py --config config.yaml
if errorlevel 1 (
    echo.
    echo [Error] Preprocessing failed.
    pause
    exit /b 1
)

echo.
set /p run_merge="Proceed with Image Merge? (Y/N, default is N): "
if /I not "%run_merge%"=="Y" (
    echo Image Merge cancelled by user.
    goto skip_merge
)

echo.
echo ========================================================
echo Running Image Merge...
echo ========================================================
python src\merge_images.py
if errorlevel 1 (
    echo.
    echo [Error] Image Merge failed.
    pause
    exit /b 1
)

:skip_merge
echo.
echo ========================================================
echo Preprocessing Execution Finished.
echo ========================================================
pause
exit /b 0
