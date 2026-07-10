@echo off
chcp 65001 >nul
cd /d "%~dp0.."

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [INFO] 가상환경이 없습니다. 현재 Python 환경을 사용합니다.
)

echo ============================================================
echo  DeepPCB Review Station (app_back)
echo ============================================================
echo.
python app_back\run.py
