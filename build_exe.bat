@echo off
REM Build the standalone Kokoro TTS desktop app (onedir) with PyInstaller.
REM Result: dist\KokoroTTS\KokoroTTS.exe  -- double-click, no server, no browser.
setlocal
cd /d "%~dp0"

set PY=".venv\Scripts\python.exe"

if not exist %PY% (
  echo ERROR: .venv not found. Create it first ^(see README^).
  exit /b 1
)

echo [1/3] Checking PyInstaller...
%PY% -c "import PyInstaller" 2>nul
if errorlevel 1 (
  echo   PyInstaller is not installed. Install it into the venv with:
  echo.
  echo       uv pip install --python "%CD%\.venv\Scripts\python.exe" pyinstaller
  echo.
  echo   then run this script again.
  exit /b 1
)

echo [1b/3] Checking the spaCy English model (misaki.en needs it)...
%PY% -c "import en_core_web_sm" 2>nul
if errorlevel 1 (
  echo   Installing en_core_web_sm...
  %PY% -m spacy download en_core_web_sm
  if errorlevel 1 (
    echo ERROR: could not install en_core_web_sm ^(needs network^).
    exit /b 1
  )
)

echo [2/3] Fetching model weights (skipped if already present)...
if exist "bundled_weights\hf\hub" (
  echo   bundled_weights\hf already present - skipping.
) else (
  %PY% -m app.weights --target "bundled_weights\hf"
  if errorlevel 1 (
    echo ERROR: weight download failed. Run with network access, or copy an
    echo        existing HF cache into bundled_weights\hf manually.
    exit /b 1
  )
)

echo [3/3] Building with PyInstaller...
%PY% -m PyInstaller --noconfirm --clean kokoro_desktop.spec
if errorlevel 1 (
  echo ERROR: PyInstaller build failed.
  exit /b 1
)

echo.
echo Done. The app is at:  dist\KokoroTTS\KokoroTTS.exe
echo Ship the whole dist\KokoroTTS folder, not just the .exe.
endlocal
