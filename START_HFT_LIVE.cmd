@echo off
setlocal
title Start HFT Paper Trader
cd /d "%~dp0"

if not exist "state" mkdir "state"

set "PYTHON="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON set "PYTHON=%%P"
if not defined PYTHON (
  echo Python was not found.
  echo Install Python or add it to PATH, then try again.
  pause
  exit /b 1
)

powershell -NoProfile -Command ^
  "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'run\.py\s+dashboard' -and $_.CommandLine -match '--port\s+8000' }; if ($p) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo Starting local dashboard...
  powershell -NoProfile -WindowStyle Hidden -Command ^
    "Start-Process -FilePath '%PYTHON%' -ArgumentList @('-u','run.py','dashboard','--port','8000') -WorkingDirectory '%CD%' -RedirectStandardOutput '%CD%\state\dashboard.stdout.log' -RedirectStandardError '%CD%\state\dashboard.stderr.log' -WindowStyle Hidden"
) else (
  echo Local dashboard is already running.
)

powershell -NoProfile -Command ^
  "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'run\.py\s+live' -and $_.CommandLine -match '(--mode\s+hft|run\.py\s+live)' }; if ($p) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo Starting HFT paper bot...
  powershell -NoProfile -WindowStyle Hidden -Command ^
    "Start-Process -FilePath '%PYTHON%' -ArgumentList @('-u','run.py','live','--mode','hft','--cash','100','--publish-github') -WorkingDirectory '%CD%' -RedirectStandardOutput '%CD%\state\hft-live.stdout.log' -RedirectStandardError '%CD%\state\hft-live.stderr.log' -WindowStyle Hidden"
) else (
  echo HFT paper bot is already running.
)

echo Waiting for the dashboard...
powershell -NoProfile -Command "Start-Sleep -Seconds 2"

if /i not "%~1"=="--no-open" start "" "http://127.0.0.1:8000/"

echo Ready: http://127.0.0.1:8000/
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
endlocal
