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
git fetch origin --prune < nul
git reset --hard origin/main
:: Force delete untracked large directories to prevent y/n prompt freeze due to Windows file lock
:: NOTE: Do NOT delete runs/, weights/, or preprocessed_data/ here — they are tracked in main branch.
if exist dataset rmdir /s /q dataset 2>nul
if exist runs\detect rmdir /s /q runs\detect 2>nul
if exist runs\train rmdir /s /q runs\train 2>nul
if exist runs\tune rmdir /s /q runs\tune 2>nul
if exist venv rmdir /s /q venv 2>nul
if exist src\__pycache__ rmdir /s /q src\__pycache__ 2>nul
:: Run git clean on remaining untracked/ignored files
git clean -fdx < nul
echo.
echo Reset complete!
pause
