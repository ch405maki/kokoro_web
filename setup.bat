@echo off
REM Recreate the virtualenv from scratch and pre-download the model.
REM Safe to run more than once.
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
  echo [ERROR] uv not found. Install it with:  pip install uv   then re-run this script.
  exit /b 1
)

echo.
echo  [1/3] Creating virtualenv (Python 3.12)
uv python install 3.12
uv venv --python 3.12 .venv || exit /b 1

echo.
echo  [2/3] Installing dependencies ^(torch CPU build first^)
uv pip install --python .venv\Scripts\python.exe torch ^
  --index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match || exit /b 1

uv pip install --python .venv\Scripts\python.exe ^
  "kokoro>=0.9.4" "misaki[en]>=0.9.4" soundfile espeakng-loader ^
  fastapi "uvicorn[standard]" python-multipart numpy || exit /b 1

echo.
echo  [3/3] Pre-downloading Kokoro-82M weights ^(~330 MB, one time^)
".venv\Scripts\python.exe" -m app.cli --warmup || exit /b 1

echo.
echo  Done. Start the server with  run_server.bat
echo.
pause
