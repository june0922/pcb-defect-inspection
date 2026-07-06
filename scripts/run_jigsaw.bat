chcp 65001 >nul
@echo off
cd /d "%~dp0"
title DeepPCB Jigsaw Reconstruction

echo ========================================
echo Starting Jigsaw Reconstruction Pipeline
echo ========================================

if "%1"=="" (
    set TARGET_GROUP=group00041
    echo No group specified, defaulting to group00041
) else (
    set TARGET_GROUP=%1
    echo Target group: %1
)

echo.
echo Running Jigsaw Solver...
python ../src/jigsaw_solver.py --group %TARGET_GROUP%
if errorlevel 1 (
    echo [ERROR] Jigsaw Solver failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Jigsaw Reconstruction Completed successfully.
echo Please check the recovered_data directory.
echo ========================================
pause
exit /b 0
