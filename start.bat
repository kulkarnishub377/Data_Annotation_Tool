@echo off
title Data Annotation Studio - close this window to stop
setlocal
cd /d "%~dp0"

rem How the UI opens is set in config.ini -> [App] LAUNCH_MODE (default: browser).
rem Closing this window stops the app.

if exist "dist\DataAnnotationStudio\DataAnnotationStudio.exe" (
    "dist\DataAnnotationStudio\DataAnnotationStudio.exe"
    exit /b %ERRORLEVEL%
)

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" app.py
) else (
    python app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Data Annotation Studio exited with code %ERRORLEVEL%.
    pause
)
