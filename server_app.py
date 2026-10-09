"""Frozen-executable entry point for the Kokoro TTS web server.

Build it with ``build_server.bat``. The result is a self-contained ``onedir``
folder: dual-clicking ``KokoroServer.exe`` starts the FastAPI app on
``127.0.0.1:8000`` and opens the web UI in the default browser. No Python, no
``pip install`` and no network access are needed on the target machine because
the interpreter, every dependency and the model weights all travel in the
folder.

Ship the whole folder, not just the ``.exe``: the executable loads its runtime
from the ``_internal`` directory beside it.

Environment overrides (all optional):
    KOKORO_PORT    port to listen on            (default 8000)
    KOKORO_BIND    interface to bind            (default 127.0.0.1)
    KOKORO_OUTPUT_DIR  where save=true writes   (default <exe folder>/output)
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _app_dir() -> Path:
    """The folder holding the executable (frozen) or this script (source)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _open_browser_when_ready(url: str, health: str, timeout: float = 90.0) -> None:
    """Poll /health, then open the browser so the first paint is never an error.

    The model loads lazily, so /health can take a few seconds to answer on a
    cold start; waiting for a 200 means the user never sees "engine offline" in
    the status dot before the engine has had a chance to come up.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.4)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    # torch can spawn helper processes; the frozen bootloader must know this is
    # not a child before any of them start.
    multiprocessing.freeze_support()

    base = _app_dir()
    os.chdir(base)
    os.environ.setdefault("KOKORO_OUTPUT_DIR", str(base / "output"))

    # Point Hugging Face at the bundled cache *before* anything imports
    # huggingface_hub, kokoro or app.engine, so the app never reaches for the
    # network. No-op when the weights are absent (falls back to an online fetch).
    from app import weights

    weights.configure_offline_weights()

    host = os.getenv("KOKORO_BIND", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("KOKORO_PORT", "8000"))
    args = argv if argv is not None else sys.argv[1:]
    if args:
        port = int(args[0])

    from app.server import app
    import uvicorn

    local_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{local_host}:{port}/"
    health = f"http://127.0.0.1:{port}/health"

    threading.Thread(
        target=_open_browser_when_ready, args=(url, health), daemon=True
    ).start()

    print()
    print("  Kokoro TTS")
    print("  ----------")
    print(f"  Web UI : {url}")
    print(f"  API    : {url}docs")
    print("  Close this window (or press Ctrl+C) to stop.")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
