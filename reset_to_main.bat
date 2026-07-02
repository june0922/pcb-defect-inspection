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
:: Force delete main directories to prevent y/n prompt freeze due to Windows file lock
if exist preprocessed_data rmdir /s /q preprocessed_data 2>nul
if exist dataset rmdir /s /q dataset 2>nul
if exist runs rmdir /s /q runs 2>nul
if exist weights rmdir /s /q weights 2>nul
:: Run git clean on remaining files and force ignore inputs
git clean -fdx < nul
echo.
echo Reset complete!
pause
