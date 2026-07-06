@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title DeepPCB Defect Review Station (Back App)

:: Activate virtual environment if it exists
if exist "%~dp0..\venv\Scripts\activate.bat" (
    call "%~dp0..\venv\Scripts\activate.bat"
)

echo ========================================
echo  DeepPCB Defect Review Station (Back App)
echo ========================================
echo.

echo [INFO] Starting review station (Back)...
echo.

python app_back/run.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
    exit /b 1
)

pause
exit /b 0
