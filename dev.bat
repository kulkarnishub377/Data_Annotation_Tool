@echo off
title Data Annotation Studio (Live Dev Mode)
setlocal

cd /d "%~dp0"

echo ================================================================
echo   Data Annotation Studio - Live Developer Mode
echo   (No rebuild needed! Edits to HTML/CSS/JS update on refresh)
echo ================================================================

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" app.py --browser
) else (
    python app.py --browser
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server exited with code %ERRORLEVEL%.
    pause
)
