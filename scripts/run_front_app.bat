@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title DeepPCB Real-Time Inspection Monitor (Front App)

:: Activate virtual environment if it exists
if exist "%~dp0..\venv\Scripts\activate.bat" (
    call "%~dp0..\venv\Scripts\activate.bat"
)

echo ========================================
echo  DeepPCB Real-Time Inspection Monitor (Front App)
echo ========================================
echo.

echo [INFO] Starting inspection monitor (Front)...
echo.

python app_front/run.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
    exit /b 1
)

pause
exit /b 0
