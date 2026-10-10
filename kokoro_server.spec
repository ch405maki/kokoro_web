# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Kokoro TTS web server (onedir, browser UI).

Build on Windows with ``build_server.bat``. The result is a self-contained
folder: the Python interpreter, torch, kokoro/misaki, FastAPI/uvicorn, the web
UI assets and the model weights all travel with the executable, so the target
machine needs no Python and no network.

The build is deliberately *onedir*: a onefile build would unpack gigabytes of
torch to a temp directory on every launch.

The heavy packages below ship data files (lexicons, espeak-ng binaries) or are
imported lazily, so PyInstaller's static analysis cannot see everything. They
are collected wholesale; if a package is not installed, it is skipped.
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = [
    # The application package and the root-level module it imports.
    "app",
    "app.server",
    "app.engine",
    "app.weights",
    # The multi-worker launcher and the string it hands to uvicorn; the worker
    # processes import "app.server" by name out of this bundle.
    "app.launcher",
    "legal_preprocessor",
]

# Packages that carry data files, native libraries or lazy imports.
for package in (
    "kokoro",
    "misaki",
    "espeakng_loader",
    "phonemizer",
    "huggingface_hub",
    "soundfile",
    "language_tags",
    "langcodes",
    "csvw",
    "segments",
    "num2words",
    "en_core_web_sm",
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# uvicorn resolves its loop/protocol/logger classes by string at runtime, so the
# static analysis misses them; the ASGI stack is pure Python apart from
# pydantic_core, which has its own hook.
for package in ("uvicorn", "fastapi", "starlette", "anyio", "pydantic",
                "click", "h11", "httptools", "websockets"):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

# The web UI is served from app/static via StaticFiles(directory=...), which
# resolves relative to the frozen module's __file__.
datas += [("app/static", "app/static")]

# Pre-fetched model weights in Hugging Face cache layout. app.weights looks for
# them under weights/hf in the bundle and flips the process to offline mode.
weights_src = os.path.join("bundled_weights", "hf")
if os.path.isdir(weights_src):
    datas += [(weights_src, os.path.join("weights", "hf"))]

a = Analysis(
    ["server_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # No desktop GUI, no plotting/IPython stack: keeps the bundle smaller.
    excludes=["tkinter", "app.desktop", "matplotlib", "IPython", "PyQt5",
              "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KokoroServer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX compression is a common antivirus false-positive trigger.
    console=True,  # a visible log window; closing it stops the server
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
    name="KokoroServer",
)
