@echo off
chcp 65001 >nul
echo ========================================================
echo [경고] 로컬 저장소를 원격 'main' 브랜치와 동일하게 초기화합니다.
echo 커밋되지 않은 모든 작업 내역과 추가된 파일이 영구적으로 삭제됩니다!
echo ========================================================
echo.
set /p confirm="정말로 초기화를 진행하시겠습니까? (Y/N): "

if /i "%confirm%" neq "Y" (
    echo.
    echo 작업이 취소되었습니다. 안전하게 종료합니다.
    pause
    exit /b
)

echo.
echo 원격 main 브랜치 기준으로 초기화 중...
git fetch origin
git reset --hard origin/main
git clean -fd
echo.
echo 초기화가 완료되었습니다!
pause
