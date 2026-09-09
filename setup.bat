@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
) else (
  set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating an isolated Python environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :failed
)

echo [2/3] Installing recording and transcription dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :failed

echo [3/3] Checking the application...
".venv\Scripts\python.exe" -c "import sounddevice, numpy; import app; print('Setup check passed')"
if errorlevel 1 goto :failed

echo.
echo Setup completed. You can now launch start.bat.
pause
exit /b 0

:failed
echo.
echo Setup did not complete. Keep the error text shown above.
pause
exit /b 1
