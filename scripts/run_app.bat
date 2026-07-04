@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title DeepPCB Defect Review Station

echo ========================================
echo  DeepPCB Defect Review Station
echo ========================================
echo.

echo [INFO] Starting review station...
echo.

python app/run.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
    exit /b 1
)

pause
exit /b 0
