@echo off
REM Build the self-contained Kokoro TTS web server into the Production folder.
REM Result: Production\Start Kokoro TTS.bat -> opens the UI in the browser.
setlocal
cd /d "%~dp0"

set PY=".venv\Scripts\python.exe"

if not exist %PY% (
  echo ERROR: .venv not found. Create it first ^(see README / setup.bat^).
  exit /b 1
)

echo [1/4] Checking PyInstaller...
%PY% -c "import PyInstaller" 2>nul
if errorlevel 1 (
  echo   PyInstaller is not installed. Install it into the venv with:
  echo.
  echo       uv pip install --python "%CD%\.venv\Scripts\python.exe" pyinstaller
  echo.
  exit /b 1
)

echo [2/4] Checking the spaCy English model ^(misaki.en needs it^)...
%PY% -c "import en_core_web_sm" 2>nul
if errorlevel 1 (
  echo   Installing en_core_web_sm...
  %PY% -m spacy download en_core_web_sm
  if errorlevel 1 (
    echo ERROR: could not install en_core_web_sm ^(needs network^).
    exit /b 1
  )
)

echo [3/4] Fetching model weights ^(skipped if already present^)...
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

echo [4/4] Building with PyInstaller ^(this takes several minutes^)...
%PY% -m PyInstaller --noconfirm --clean --distpath Production --workpath build\production kokoro_server.spec
if errorlevel 1 (
  echo ERROR: PyInstaller build failed.
  exit /b 1
)

echo.
echo Assembling the Production folder...
copy /y "config.json" "Production\KokoroServer\config.json" >nul
copy /y "deploy\production\Start Kokoro TTS.bat" "Production\Start Kokoro TTS.bat" >nul
copy /y "deploy\production\README.txt" "Production\README.txt" >nul
if not exist "Production\KokoroServer\output" mkdir "Production\KokoroServer\output"

echo.
echo Done. Start it with:  Production\Start Kokoro TTS.bat
echo Copy the WHOLE Production folder to another machine to run it there.
endlocal
