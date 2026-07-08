@echo off
chcp 65001 >nul
:: Change directory to the project root
cd /d "%~dp0.."
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
:: Pre-delete stale remote ref subdirectories to avoid Windows file-lock prompt during fetch
for /d %%D in (".git\refs\remotes\origin\*") do (
    if /i not "%%~nxD"=="HEAD" rmdir /s /q "%%D" 2>nul
)
:: Force delete untracked large directories to prevent y/n prompt freeze due to Windows file lock
:: This is done BEFORE git reset --hard so any accidentally deleted tracked files are restored.
if exist preprocessed_data rmdir /s /q preprocessed_data 2>nul
if exist dataset rmdir /s /q dataset 2>nul
if exist venv rmdir /s /q venv 2>nul
if exist src\__pycache__ rmdir /s /q src\__pycache__ 2>nul
if exist app_back\__pycache__ rmdir /s /q app_back\__pycache__ 2>nul
if exist web_test\preprocessed_data rmdir /s /q web_test\preprocessed_data 2>nul
if exist web_test\results rmdir /s /q web_test\results 2>nul
if exist web_test\runs rmdir /s /q web_test\runs 2>nul
if exist web_test\weights rmdir /s /q web_test\weights 2>nul

git fetch origin --prune < nul
git reset --hard origin/main

:: Run git clean on remaining untracked/ignored files
git clean -fdx < nul
echo.
echo Reset complete!
pause
