"""Throughput benchmark. Run once, prints a table for the README.

    .venv\\Scripts\\python.exe -m tests.bench
"""

from __future__ import annotations

import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import engine as E  # noqa: E402

CASES = [
    ("one sentence", "The quick brown fox jumps over the lazy dog."),
    ("one paragraph", (
        "Kokoro is a text to speech model with eighty two million parameters. "
        "It runs entirely on your own hardware, so no text ever leaves your machine. "
        "The weights are licensed under Apache, which means commercial use is fine."
    )),
    ("three paragraphs", "\n\n".join([
        "The sky above the port was the color of television, tuned to a dead channel.",
        "It was a sprawl voice and a sprawl joke, and nobody in the room looked up.",
        "These decisions were to have an enormous impact, not only because they were "
        "associated with Constantine, but because they mattered for centuries to come.",
    ])),
]


def main() -> int:
    eng = E.get_engine()
    t0 = time.perf_counter()
    eng.warm_up()
    cold = time.perf_counter() - t0
    print(f"device = {eng.device}\nmodel load (warm from disk) = {cold:.2f}s\n")

    header = f"{'case':<16} {'chars':>6} {'audio':>8} {'gen':>8} {'xRT':>7} {'chunks':>7}"
    print(header)
    print("-" * len(header))

    for label, text in CASES:
        # First pass pays one-off lazy init; second is the steady-state number.
        eng.synthesize(text, voice="af_heart", split_pattern=r"\n+")
        runtimes, total_audio, n_chunks = [], 0.0, 0
        for _ in range(3):
            t = time.perf_counter()
            chunks = eng.synthesize(text, voice="af_heart", split_pattern=r"\n+")
            runtimes.append(time.perf_counter() - t)
            total_audio = sum(c.duration for c in chunks)
            n_chunks = len(chunks)
        gen = statistics.median(runtimes)
        print(f"{label:<16} {len(text):>6} {total_audio:>7.2f}s {gen:>7.2f}s "
              f"{total_audio / gen:>6.1f}x {n_chunks:>7}")

    print()
    print("Voices:")
    for locale in ("a", "b"):
        names = [v["name"] for v in E.list_voices(locale)]
        print(f"  {locale}  ({len(names)})  {', '.join(names)}")
    print(f"\nTotal voices available: {len(E.list_voices())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
