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
git clean -fdx
echo.
echo Reset complete!
pause
