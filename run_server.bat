@echo off
REM Start the Kokoro TTS API + web UI.
REM Usage:  run_server.bat  [port]  [host]  [workers]
REM
REM workers is optional and defaults to 1. Each extra worker is another process
REM with its own copy of the model (~1.3 GB of RAM) that serves requests at the
REM same time instead of queueing behind the others. It is capped at the number
REM of CPUs, and KOKORO_WORKERS sets the same thing.
setlocal
cd /d "%~dp0"

set PORT=%1
if "%PORT%"=="" set PORT=8000
set HOST=%2
if "%HOST%"=="" set HOST=127.0.0.1
set WORKEROPT=
if not "%3"=="" set WORKEROPT=--workers %3

".venv\Scripts\python.exe" -m app.launcher --host %HOST% --port %PORT% %WORKEROPT%
