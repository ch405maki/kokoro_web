"""Frozen-executable entry point for the standalone Kokoro desktop app.

Lives at the repo root so PyInstaller treats it as the import root: ``import app``
resolves, and the repo-root ``legal_preprocessor`` module is importable. Running
it directly (``python desktop_app.py``) works the same as the frozen build.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # torch can spawn helper processes; the frozen bootloader must know this is
    # not a child before any of them start.
    multiprocessing.freeze_support()

    # Point Hugging Face at the bundled cache *before* anything imports
    # huggingface_hub or kokoro, so the app never reaches for the network.
    from app import weights

    weights.configure_offline_weights()

    from app.desktop import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
