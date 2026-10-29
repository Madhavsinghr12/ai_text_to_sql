@echo off
setlocal
cd /d "%~dp0"
title text2sql - ask your data in plain English
set PYTHONUTF8=1

if exist ".venv\.installed" goto run

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH", then double-click start.bat again.
  pause
  exit /b 1
)

echo First run: installing, this takes about a minute...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -e . || goto failed
echo ok> ".venv\.installed"

:run
".venv\Scripts\python.exe" -m text2sql %*
echo.
pause
exit /b 0

:failed
echo.
echo Setup failed. Check your internet connection and double-click start.bat again.
pause
exit /b 1
