@echo off
REM Start the Kokoro TTS API + web UI.
REM Usage:  run_server.bat  [port]  [host]
setlocal
cd /d "%~dp0"

set PORT=%1
if "%PORT%"=="" set PORT=8000
set HOST=%2
if "%HOST%"=="" set HOST=127.0.0.1

echo.
echo   Kokoro TTS
echo   ---------------
echo   Web UI : http://%HOST%:%PORT%/
echo   API    : http://%HOST%:%PORT%/docs
echo.
echo   First request downloads ~330 MB of weights. Ctrl+C to stop.
echo.

".venv\Scripts\python.exe" -m uvicorn app.server:app --host %HOST% --port %PORT%
