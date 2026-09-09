@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else (
  echo [meeting-record] Runtime environment is not installed.
  echo Run setup.bat once, then launch start.bat again.
  echo.
  pause
  exit /b 1
)
if errorlevel 1 pause
