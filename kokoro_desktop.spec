# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Kokoro TTS desktop app (onedir, no console).

Build on Windows with ``build_exe.bat``; it fetches the weights first. The build
is deliberately *onedir*: a onefile build would unpack several gigabytes of
torch to a temp directory on every launch.

The packages below all ship data files (lexicons, espeak-ng binaries) or are
imported lazily, so PyInstaller's static analysis cannot see everything. They
are collected wholesale; if a package is not installed, it is skipped.
"""

import os

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    # Imported lazily by app.desktop.clean_legal_text; also needs config.json.
    "legal_preprocessor",
]

for package in (
    "kokoro",
    "misaki",
    "espeakng_loader",
    "phonemizer",
    "huggingface_hub",
    "soundfile",
    # phonemizer's segment handling pulls these in, and they ship data files
    # (JSON subtag tables) that PyInstaller's default hooks do not collect.
    "language_tags",
    "langcodes",
    "csvw",
    "segments",
    "num2words",
    # misaki.en calls spacy.load("en_core_web_sm"); the model is data, not a
    # dependency PyInstaller can infer, and must be present for offline use.
    "en_core_web_sm",
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# The preprocessor reads config.json beside the module; ship it in the bundle
# root (which is where a frozen module's __file__ resolves to).
datas += [("config.json", ".")]

# Pre-fetched model weights in Hugging Face cache layout. app.weights looks for
# them under weights/hf in the bundle and flips the process to offline mode.
weights_src = os.path.join("bundled_weights", "hf")
if os.path.isdir(weights_src):
    datas += [(weights_src, os.path.join("weights", "hf"))]

a = Analysis(
    ["desktop_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["app.server", "uvicorn", "fastapi", "starlette"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KokoroTTS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX compression is a common antivirus false-positive trigger.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="KokoroTTS",
)
