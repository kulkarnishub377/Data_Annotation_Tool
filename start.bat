@echo off
title Data Annotation Studio Launcher
setlocal

cd /d "%~dp0"

echo ================================================================
echo   Data Annotation Studio
echo ================================================================

if exist "venv\Scripts\python.exe" (
    echo [*] Starting Data Annotation Studio in Virtual Environment...
    "venv\Scripts\python.exe" app.py %*
) else (
    echo [*] Starting Data Annotation Studio with system Python...
    python app.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
