@echo off
echo ========================================================
echo [WARNING] This will reset your local repository to match
echo the remote 'main' branch.
echo All uncommitted changes and untracked files will be
echo PERMANENTLY DELETED!
echo ========================================================
echo.
set /p confirm="Are you sure you want to proceed? (Y/N): "

if /i "%confirm%" neq "Y" (
    echo.
    echo Operation cancelled. Exiting safely.
    pause
    exit /b
)

echo.
echo Resetting to the latest origin/main...
git fetch origin
git reset --hard origin/main
:: 윈도우 파일 잠금으로 인한 y/n 무한 대기를 막기 위해 주요 폴더 먼저 강제 삭제 시도
if exist data rmdir /s /q data 2>nul
if exist dataset rmdir /s /q dataset 2>nul
if exist runs rmdir /s /q runs 2>nul
if exist weights rmdir /s /q weights 2>nul
:: 남은 파일들에 대해서 git clean 을 수행하며 입력을 강제로 무시함
git clean -fdx < nul
echo.
echo Reset complete!
pause
