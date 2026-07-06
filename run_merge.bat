@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Merge Images Tool
echo ========================================
echo Starting Image Merge...
echo ========================================
python src\merge_images.py
if errorlevel 1 (
    echo [ERROR] An error occurred while running the script.
    exit /b 1
)
echo ========================================
echo Process completed successfully.
pause
exit /b 0
