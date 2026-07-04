@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Install Dependencies

echo ======== PRE-FLIGHT CHECKS ========
:: Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)
echo [INFO] Python is installed.

echo ======== VIRTUAL ENVIRONMENT ========
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment 'venv' already exists.
)

echo ======== INSTALLING DEPENDENCIES ========
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul

echo [INFO] Installing packages from requirements.txt...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo ======== SUCCESS ========
echo [INFO] All requirements have been installed successfully.
echo [INFO] To activate the environment manually in the future, run:
echo        venv\Scripts\activate
echo =========================
pause
exit /b 0
