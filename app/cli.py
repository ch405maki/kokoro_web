"""Command-line interface for one-shot Kokoro speech generation.

Examples
--------
    python -m app.cli "Hello from Kokoro."
    python -m app.cli --text-file script.txt --voice bf_emma --speed 0.9
    python -m app.cli --list-voices
    python -m app.cli "Long text..." --format mp3 --out narration.mp3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import engine as E


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kokoro",
        description="Generate speech with Kokoro-82M.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("text", nargs="*", help="Text to speak (or use --text-file / stdin)")
    p.add_argument("--text-file", "-t", type=Path, help="Read text from a file")
    p.add_argument("--stdin", action="store_true", help="Read text from stdin")
    p.add_argument("--voice", "-v", default=E.DEFAULT_VOICE, help="Voice id (default: %(default)s)")
    p.add_argument("--speed", "-s", type=float, default=1.0, help="0.5-2.0 (default: %(default)s)")
    p.add_argument("--format", dest="fmt", choices=["wav", "mp3"], default="wav",
                   help="Audio format (default: %(default)s)")
    p.add_argument("--out", "-o", type=Path, help="Output file (default: timestamped file in ./output)")
    p.add_argument("--bitrate", help="MP3 only, e.g. 96k (default: KOKORO_MP3_BITRATE)")
    p.add_argument("--gap", type=float, default=0.12, help="Silence between chunks, seconds")
    p.add_argument("--split-pattern", help="Custom regex chunker, e.g. r'\\n+'")
    p.add_argument("--no-join", action="store_true", help="Write one file per chunk instead of joining")
    p.add_argument("--list-voices", action="store_true",
                   help="Print available voices and exit")
    p.add_argument("--all", action="store_true",
                   help="With --list-voices, include voices whose language pack is missing")
    p.add_argument("--show-phonemes", action="store_true", help="Print graphemes/phonemes per chunk")
    p.add_argument("--play", action="store_true", help="Open the result in the default player")
    p.add_argument("--warmup", action="store_true", help="Download/load the model, then exit")
    return p


def resolve_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    return " ".join(args.text)


def main(argv: list[str] | None = None) -> int:
    # IPA phoneme output is not encodable in the Windows cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    # Nothing typed and nothing piped in: fall back to stdin rather than erroring.
    if not (args.text or args.text_file or args.stdin or args.list_voices or args.warmup):
        args.stdin = not sys.stdin.isatty()

    if args.list_voices:
        items = E.list_voices()
        if not args.all:
            items = [v for v in items if v["available"]]
        width = max(len(v["name"]) for v in items)
        print(f"{'VOICE'.ljust(width)}  {'LOCALE'.ljust(17)}  {'GENDER'.ljust(7)}  QUALITY")
        print("-" * (width + 42))
        for v in items:
            mark = "" if v["available"] else "   [needs misaki pack]"
            print(f"{v['name'].ljust(width)}  {str(v['locale']).ljust(17)}  "
                  f"{str(v['gender']).ljust(7)}  {v['quality']}{mark}")
        usable = sorted(E.available_lang_codes())
        print(f"\n{len(items)} voices listed. Usable languages: {', '.join(usable)}")
        if not args.all:
            print(f"({len(E.list_voices()) - len(items)} more hidden; pass --all to show them)")
        return 0

    eng = E.get_engine()
    if args.warmup:
        info = eng.warm_up()
        print(f"Model ready on {info['device']} in {info['load_seconds']}s ({info['voices']} voices).")
        return 0

    text = resolve_text(args)
    if not text.strip():
        print("error: no text provided", file=sys.stderr)
        return 2

    split_pattern = args.split_pattern
    if split_pattern is None and args.gap > 0:
        split_pattern = r"\n+"  # paragraph-aware chunking

    started = time.perf_counter()
    try:
        chunks = eng.synthesize(
            text=text, voice=args.voice, speed=args.speed, split_pattern=split_pattern
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    if args.show_phonemes:
        for i, c in enumerate(chunks):
            print(f"[{i}] ({c.duration}s)")
            print(f"    graphemes: {c.graphemes.strip()}")
            print(f"    phonemes : {c.phonemes.strip()}")

    E.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.no_join:
        for i, c in enumerate(chunks):
            path = args.out.with_name(f"{args.out.stem}-{i:03d}{args.out.suffix}") if args.out else E.OUTPUT_DIR / f"{args.voice}-{i:03d}.{args.fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(E.encode([c], args.fmt, gap=0.0, bitrate=args.bitrate)[0])
            written.append(path)
    else:
        target = args.out or E.OUTPUT_DIR / f"{args.voice}-{time.strftime('%Y%m%d-%H%M%S')}.{args.fmt}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(E.encode(chunks, args.fmt, gap=args.gap, bitrate=args.bitrate)[0])
        written.append(target)

    total = sum(c.duration for c in chunks)
    print(
        f"{len(chunks)} chunk(s), {total:.2f}s audio generated in {elapsed:.2f}s "
        f"({total / elapsed:.1f}x realtime, device={eng.device})"
    )
    for p in written:
        print(f"  -> {p}  ({p.stat().st_size / 1024:.0f} KB)")

    if args.play and written:
        import os

        os.startfile(written[0])  # noqa: S606 - Windows-only convenience
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
