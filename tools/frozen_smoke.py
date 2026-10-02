"""Smoke test for the frozen bundle.

The GUI only imports ``torch`` and ``kokoro`` when you press Generate, so a
successful window launch does not prove they were bundled correctly. This entry
exercises the whole path (weights -> kokoro -> misaki/espeak G2P -> wav).

It is built **windowed** (``console=False``), like the real app, so it also
catches windowed-only failures such as ``sys.stderr`` being ``None`` (which
breaks kokoro's loguru setup). Because a windowed process has no console, the
result is written to ``SMOKE_OUT`` (default ``smoke_out.txt``).

Build with ``smoke.spec`` and run with ``KOKORO_WEIGHTS_DIR`` set.
"""

from __future__ import annotations

import os
import sys
import traceback


def main() -> int:
    outcome = "FAILED: no result"
    status = 1
    try:
        # Same first step as the real desktop entry point.
        from app.desktop import ensure_standard_streams

        ensure_standard_streams()

        from app import weights

        weights.configure_offline_weights()

        from app import engine as E

        chunks = E.get_engine().synthesize(
            "Hello world. This is a frozen build.", voice="af_heart"
        )
        wav, content_type = E.encode(chunks, fmt="wav")
        outcome = (
            f"OK chunks={len(chunks)} bytes={len(wav)} type={content_type} "
            f"device={E.detect_device()}"
        )
        status = 0
    except Exception as exc:  # noqa: BLE001 - a smoke test reports everything
        outcome = f"FAILED: {exc!r}\n{traceback.format_exc()}"

    destination = os.environ.get("SMOKE_OUT", "smoke_out.txt")
    try:
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(outcome + "\n")
    except OSError:
        pass
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # last resort: windowed process would otherwise die silently
        try:
            with open(os.environ.get("SMOKE_OUT", "smoke_out.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write("FAILED at startup:\n" + traceback.format_exc())
        except OSError:
            pass
        sys.exit(1)
