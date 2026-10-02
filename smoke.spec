# -*- mode: python ; coding: utf-8 -*-
"""Console PyInstaller spec for tools/frozen_smoke.py.

Same package collection as the real desktop build, but console=True so the
result can be run and read, and without the model weights (the smoke test points
KOKORO_WEIGHTS_DIR at the source ``bundled_weights/hf`` instead).
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

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

a = Analysis(
    ["tools/frozen_smoke.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # tkinter must stay: the smoke entry imports app.desktop (which needs it).
    excludes=["app.server", "uvicorn", "fastapi", "starlette"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="FrozenSmoke",
    # Windowed on purpose: this mirrors the real desktop build, so windowed-only
    # failures (None std streams) show up here instead of in the shipped app.
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="FrozenSmoke")
