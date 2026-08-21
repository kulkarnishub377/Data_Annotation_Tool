@echo off
setlocal
cd /d "%~dp0"

if exist "dist\DataAnnotationStudio\DataAnnotationStudio.exe" (
    start "" "dist\DataAnnotationStudio\DataAnnotationStudio.exe"
    exit /b 0
)

if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" app.py
    exit /b 0
)

if exist "venv\Scripts\python.exe" (
    start "" "venv\Scripts\python.exe" app.py
    exit /b 0
)

start "" python app.py
exit /b 0
