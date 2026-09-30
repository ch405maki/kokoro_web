@echo off
REM Generate a WAV file from text.  Usage:  speak.bat "Your text here"
if "%~1"=="" (
  echo Usage: speak.bat "text to speak"  [--voice af_heart] [--speed 1.0]
  exit /b 1
)
cd /d "%~dp0"
".venv\Scripts\python.exe" -m app.cli %*
