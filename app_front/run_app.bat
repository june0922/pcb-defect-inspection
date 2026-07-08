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
echo  DeepPCB PCB 결함 검사 시스템
echo  (Review Station + Inspection Monitor 동시 실행)
echo ============================================================
echo.
echo [1/2] app_back ^(Review Station^) 시작 중...
echo       - 5개 YOLO 모델 로딩에 약 30초~1분 소요됩니다.
start "DeepPCB Review Station" python app_back\run.py

echo.
echo [대기] app_back 초기화를 위해 3초 대기 중...
timeout /t 3 /nobreak > nul

echo.
echo [2/2] app_front ^(Inspection Monitor^) 시작 중...
start "DeepPCB Inspection Monitor" python app_front\run.py

echo.
echo ============================================================
echo  두 프로그램이 시작되었습니다.
echo  - Review Station: REVIEW 타일 수신 대기 중 (모델 로딩 완료 후 폴링 시작)
echo  - Inspection Monitor: 폴더 선택 후 검사를 시작하세요 (File > Open Folder)
echo ============================================================
