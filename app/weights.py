"""Locate and activate bundled Kokoro weights for offline, serverless use.

The HTTP server and the CLI download the ~330 MB model from Hugging Face into
``HF_HOME`` on first use. The standalone desktop build has no business making
that call at runtime, so the weights are fetched ahead of time (see
``build_exe.bat``) and shipped beside the executable. This module finds that
copy, points the Hugging Face cache at it, and flips the hub into offline mode
*before* ``kokoro`` or ``huggingface_hub`` are imported anywhere.

Layout, matching the Hugging Face cache convention ``HF_HOME/hub/...``::

    coined_weights/hf/hub/models--hexgrad--Kokoro-82M/...

When run as a script it downloads the weights into the target directory::

    python -m app.weights --target bundled_weights/hf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

__all__ = [
    "default_weights_dir",
    "configure_offline_weights",
    "fetch_weights",
    "DEFAULT_REPO",
]

DEFAULT_REPO = "hexgrad/Kokoro-82M"


def default_weights_dir() -> Path:
    """Where to look for bundled weights, in priority order.

    ``KOKORO_WEIGHTS_DIR`` wins so an operator can point a deployment at a
    shared copy. Inside a frozen executable the data is unpacked under
    ``sys._MEIPASS`` (PyInstaller) or sits beside the ``.exe``. From a source
    checkout it is the ``bundled_weights/hf`` folder the fetch step fills.
    """
    override = os.getenv("KOKORO_WEIGHTS_DIR")
    if override:
        return Path(override).expanduser()

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "weights" / "hf"

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "weights" / "hf"

    return Path(__file__).resolve().parent.parent / "bundled_weights" / "hf"


def configure_offline_weights(weights_dir: Path | str | None = None,
                              offline: bool = True) -> Path | None:
    """Point Hugging Face at the bundled cache and forbid network access.

    Returns the weights directory when a usable cache is present, otherwise
    ``None`` (the caller may then fall back to the normal online download).

    ``os.environ.setdefault`` is deliberate: an operator who exported ``HF_HOME``
    or set ``HF_HUB_OFFLINE=0`` keeps control. Call this before importing
    ``app.engine`` so the hub reads the cache path at import time.
    """
    target = Path(weights_dir) if weights_dir else default_weights_dir()
    if not (target / "hub").is_dir():
        return None

    os.environ.setdefault("HF_HOME", str(target))
    if offline:
        # The model, voices and tokenizer are all in the cache; any hub call
        # would only mean something is missing, so fail fast instead of hanging.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return target


def fetch_weights(target: Path | str | None = None, repo: str = DEFAULT_REPO) -> Path:
    """Download the full model repo into the cache at ``target``.

    Used by the build script; requires network access. Returns the snapshot
    directory Hugging Face resolved to.
    """
    destination = Path(target) if target else default_weights_dir()
    destination = destination.expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    # Must be set before huggingface_hub is imported so the cache lands here.
    os.environ["HF_HOME"] = str(destination)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo))


def _human_size(path: Path) -> str:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.weights",
        description="Fetch the Kokoro-82M weights for offline bundling.",
    )
    parser.add_argument("--target", type=Path, default=None,
                        help="Cache directory (default: bundled_weights/hf)")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"Hugging Face repo id (default: {DEFAULT_REPO})")
    args = parser.parse_args(argv)

    target = args.target or default_weights_dir()
    print(f"Fetching {args.repo} into {target} ...")
    snapshot = fetch_weights(target, args.repo)
    print(f"Downloaded to {snapshot}")
    print(f"Cache size: {_human_size(target)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual/build entry
    raise SystemExit(main())
