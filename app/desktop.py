"""Standalone Tkinter desktop front end for the Kokoro engine.

Same engine as the CLI and the HTTP API, but with no server and no browser: the
widgets call :mod:`app.engine` directly. Tkinter is used because it is in the
standard library, so freezing this into an executable does not add a second GUI
toolkit on top of torch.

The file is split so the part worth testing is not tangled up in widgets:

* :class:`SynthesisController` - text cleaning, synthesis and encoding. It takes
  the engine module as a parameter, so tests can drive it with a stub.
* :class:`DesktopApp` - the window. All synthesis happens on a worker thread and
  results come back through a queue, because Tkinter must only be touched from
  the main thread.
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from . import weights  # noqa: F401  (imported here so the app and tests share it)

__all__ = [
    "AudioResult",
    "SynthesisController",
    "DesktopApp",
    "clean_legal_text",
    "title_case",
    "estimate_seconds",
    "format_duration",
    "main",
]

APP_TITLE = "Kokoro TTS"


def ensure_standard_streams() -> None:
    """Give ``sys.stdout``/``sys.stderr`` a sink when running windowed.

    A PyInstaller build with ``console=False`` leaves both as ``None``. That is
    normally harmless, but importing ``kokoro`` calls
    ``loguru.logger.add(sys.stderr, ...)``, and loguru raises
    ``TypeError: Cannot log to objects of type 'NoneType'`` for a ``None`` sink
    - so the model fails to load with a confusing error. Point them at the null
    device before any third-party import can see them. No-op when a real stream
    exists, so the CLI and API are unaffected.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
# Kokoro's own default silence between chunks; long enough to sound like a
# paragraph break, short enough not to feel like a stall.
DEFAULT_GAP = 0.12

# Measured on the reference machine: ordinary prose runs at roughly this many
# characters per second of Kokoro audio at speed 1.0. It only feeds the live
# estimate next to the character count; the real duration is reported after
# synthesis. Throughput is per-CPU, so this is a deliberately round number.
CHARS_PER_SECOND = 15.0

# Words that stay lowercase inside a title unless they open the text or start a
# sentence. Mirrors the browser UI's title-case pass so both front ends format
# the same text the same way.
SMALL_WORDS = frozenset({
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into",
    "nor", "of", "on", "onto", "or", "over", "per", "so", "the", "to", "up",
    "via", "with", "yet",
})

# A word is a letter or digit followed by letters, digits and apostrophes.
# ``[^\W_]`` is Unicode-aware without the third-party regex module; the
# apostrophe and the typographic right quote are added back explicitly so
# "don't" is a single word.
_WORD_RE = re.compile(r"[^\W_](?:[^\W_]|['\u2019])*")
_SENTENCE_END_RE = re.compile(r"[.!?:\n]\s*$")


def _case_word(raw: str, force_cap: bool = False) -> str:
    lower = raw.lower()
    if force_cap or lower not in SMALL_WORDS:
        out = lower[:1].upper() + lower[1:]
    else:
        out = lower
    # A rare casing pair can change length (e.g. "\u00df" -> "SS"); leave those
    # alone so the text keeps its exact length.
    return out if len(out) == len(raw) else raw


def title_case(text: str) -> str:
    """Normalise prose to title case, the same way the browser UI does.

    Every word is capitalised with the rest lowercased, so an all-caps ``SAMPLE``
    becomes ``Sample``, while short joining words stay lowercase unless they open
    the text or a sentence.
    """
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text
    out: list[str] = []
    prev = 0
    for index, match in enumerate(matches):
        between = text[prev:match.start()]
        force_cap = index == 0 or bool(_SENTENCE_END_RE.search(between))
        out.append(between)
        out.append(_case_word(match.group(0), force_cap))
        prev = match.end()
    out.append(text[prev:])
    return "".join(out)


def estimate_seconds(text: str, speed: float = 1.0) -> float:
    """Rough spoken length of ``text`` at ``speed``, for the live counter."""
    rate = CHARS_PER_SECOND * (speed if speed > 0 else 1.0)
    return len(text) / rate if rate else 0.0


def format_duration(seconds: float) -> str:
    """Human duration: ``1h 02m 03s``, ``2m 05s`` or ``7s``."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def clean_legal_text(text: str) -> str:
    """Run the legal preprocessor, imported lazily so it can be stubbed in tests."""
    from legal_preprocessor import preprocess_legal_text

    return preprocess_legal_text(text)


@dataclass
class AudioResult:
    """One finished synthesis pass."""

    audio: bytes          # in the requested format
    fmt: str              # "wav" or "mp3"
    playback_wav: bytes   # always WAV, for in-memory playback
    duration: float
    chunks: int
    text: str


class SynthesisController:
    """UI-free glue between the window and the engine.

    ``engine_module`` defaults to :mod:`app.engine`, but is injected in tests.
    It is looked up lazily so importing this module never pulls in torch.
    ``preprocessor`` likewise defaults to the legal-text cleaner.
    """

    def __init__(self, engine_module=None, preprocessor=None) -> None:
        self._engine_module = engine_module
        self._preprocess = preprocessor or clean_legal_text

    @property
    def engine(self):
        if self._engine_module is None:
            from . import engine as E

            self._engine_module = E
        return self._engine_module

    def voices(self, only_available: bool = True) -> list[dict]:
        items = list(self.engine.list_voices())
        if only_available:
            items = [v for v in items if v.get("available")]
        return items

    def default_voice(self) -> str:
        default = getattr(self.engine, "DEFAULT_VOICE", "af_heart")
        names = [str(v["name"]) for v in self.voices()]
        return default if default in names else (names[0] if names else default)

    def format_text(self, text: str) -> str:
        """Auto-Format: legal cleaning followed by title case.

        The browser UI runs the same two passes. Cleaning happens first so the
        preprocessor still sees the document's original casing - it matches
        abbreviations such as ``CA`` and ``G.R.`` on those exact tokens, and
        title-casing first would hand it ``Ca`` instead.
        """
        return title_case(self._preprocess(text))

    def synthesize(self, *, text: str, voice: str, speed: float = 1.0,
                   fmt: str = "wav", gap: float = DEFAULT_GAP,
                   clean: bool = True, bitrate: str | None = None) -> AudioResult:
        """Clean (optionally), synthesize and encode in one call.

        Raises ``ValueError`` for empty input after cleaning, which is what the
        window surfaces in the status bar rather than a traceback.
        """
        source = self._preprocess(text) if clean else text
        if not source or not source.strip():
            raise ValueError("nothing to speak")

        engine = self.engine
        chunks = engine.get_engine().synthesize(source, voice=voice, speed=speed)
        if not chunks:
            raise ValueError("the engine produced no audio")

        playback_wav, _ = engine.encode(chunks, fmt="wav", gap=gap)
        if fmt == "wav":
            audio = playback_wav
        else:
            audio, _ = engine.encode(chunks, fmt=fmt, gap=gap, bitrate=bitrate)

        duration = round(sum(float(getattr(c, "duration", 0.0)) for c in chunks), 3)
        return AudioResult(audio=audio, fmt=fmt, playback_wav=playback_wav,
                           duration=duration, chunks=len(chunks), text=source)

    def play(self, result: AudioResult) -> None:
        """Play a result in memory. Windows-only; otherwise it is a no-op."""
        try:
            import winsound
        except ImportError:
            return
        winsound.PlaySound(result.playback_wav, winsound.SND_MEMORY)


class DesktopApp:
    """The main window. Create it with an existing ``tk.Tk`` root."""

    def __init__(self, root: tk.Tk, controller: SynthesisController | None = None) -> None:
        self.root = root
        self.controller = controller or SynthesisController()
        self.queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.result: AudioResult | None = None
        self._busy = False
        # Guards the <<Modified>> handler against re-entering while we rewrite
        # the widget, and holds the id of the pending title-case debounce.
        self._formatting = False
        self._debounce: str | None = None

        root.title(APP_TITLE)
        root.geometry("760x620")
        root.minsize(560, 460)

        self._build()
        self._load_voices()
        self._poll()

    # ------------------------------------------------------------- construction

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="Text to speak").pack(side="left")
        self.count = ttk.Label(header, text="0 characters")
        self.count.pack(side="right")
        ttk.Button(header, text="Auto-Format", command=self.on_format).pack(
            side="right", padx=(0, 8))

        self.text = ScrolledText(outer, wrap="word", height=16, undo=True)
        self.text.pack(fill="both", expand=True, pady=(2, 8))
        self.text.insert("1.0", "In the matter of the respondents, the Court ruled.")
        self.text.edit_modified(False)
        self.text.bind("<<Modified>>", self._on_modified)

        options = ttk.LabelFrame(outer, text="Options", padding=8)
        options.pack(fill="x")

        ttk.Label(options, text="Voice").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.voice = ttk.Combobox(options, state="readonly", width=22, values=[])
        self.voice.grid(row=0, column=1, sticky="w")

        ttk.Label(options, text="Speed").grid(row=0, column=2, sticky="w", padx=(16, 6))
        self.speed = tk.DoubleVar(value=1.0)
        ttk.Scale(options, from_=0.5, to=2.0, variable=self.speed,
                  orient="horizontal", length=150).grid(row=0, column=3, sticky="w")
        self.speed_label = ttk.Label(options, text="1.00x")
        self.speed_label.grid(row=0, column=4, sticky="w", padx=(6, 0))
        self.speed.trace_add("write", self._on_speed)

        ttk.Label(options, text="Format").grid(row=1, column=0, sticky="w", pady=(8, 0))
        # MP3 matches the browser UI's default and is a fraction of the size of
        # WAV for a 24 kHz mono voice (see KOKORO_MP3_BITRATE).
        self.fmt = tk.StringVar(value="mp3")
        fmt_frame = ttk.Frame(options)
        fmt_frame.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Radiobutton(fmt_frame, text="WAV", value="wav",
                        variable=self.fmt).pack(side="left")
        ttk.Radiobutton(fmt_frame, text="MP3", value="mp3",
                        variable=self.fmt).pack(side="left", padx=(8, 0))

        self.clean = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Clean legal text before speaking",
                        variable=self.clean).grid(row=1, column=2, columnspan=3,
                                                  sticky="w", pady=(8, 0))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(10, 6))
        self.generate_btn = ttk.Button(buttons, text="Generate", command=self.on_generate)
        self.generate_btn.pack(side="left")
        self.play_btn = ttk.Button(buttons, text="Play", command=self.on_play, state="disabled")
        self.play_btn.pack(side="left", padx=(8, 0))
        self.save_btn = ttk.Button(buttons, text="Save as...", command=self.on_save, state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Clear", command=self.on_clear).pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x")
        self.status = ttk.Label(outer, text="Ready.", anchor="w")
        self.status.pack(fill="x", pady=(6, 0))
        self._update_count()

    def _load_voices(self) -> None:
        try:
            voices = self.controller.voices()
        except Exception as exc:  # keep the window usable even if enumeration fails
            voices = []
            self._set_status(f"Could not list voices: {exc}")
        names = [str(v["name"]) for v in voices]
        self.voice["values"] = names
        if names:
            chosen = self.controller.default_voice()
            self.voice.set(chosen if chosen in names else names[0])

    # ----------------------------------------------------------------- callbacks

    def _on_speed(self, *_: object) -> None:
        self.speed_label.configure(text=f"{self.speed.get():.2f}x")
        self._update_count()

    # ------------------------------------------------------- formatting state

    def _update_count(self) -> None:
        """Refresh the live character count and the speech-length estimate."""
        text = self.text.get("1.0", "end-1c")
        count = len(text)
        unit = "character" if count == 1 else "characters"
        estimate = estimate_seconds(text, self.speed.get())
        self.count.configure(text=f"{count:,} {unit}  \u00b7  ~{format_duration(estimate)}")

    def _insert_offset(self) -> int | None:
        """Caret position as an absolute character offset, so re-casing keeps it."""
        try:
            line, column = (int(part) for part in self.text.index("insert").split("."))
            offset = column
            for number in range(1, line):
                offset += len(self.text.get(f"{number}.0", f"{number}.end")) + 1
            return offset
        except Exception:
            return None

    def _replace_text(self, new_text: str) -> None:
        """Replace the whole box while leaving the caret where it was."""
        offset = self._insert_offset()
        self._formatting = True
        try:
            self.text.delete("1.0", "end")
            self.text.insert("1.0", new_text)
            if offset is not None:
                self.text.mark_set("insert", f"1.0+{min(offset, len(new_text))}c")
        finally:
            self._formatting = False

    def _on_modified(self, _event: object = None) -> None:
        """Text changed: refresh the counter and re-case shortly afterwards."""
        if not self.text.edit_modified():
            return
        self.text.edit_modified(False)
        if self._formatting:
            return
        self._update_count()
        if self._debounce is not None:
            self.root.after_cancel(self._debounce)
        self._debounce = self.root.after(250, self._apply_title_case)

    def _apply_title_case(self) -> None:
        """Debounced title-case pass, mirroring the browser UI's input handler."""
        self._debounce = None
        current = self.text.get("1.0", "end-1c")
        formatted = title_case(current)
        if formatted != current:
            self._replace_text(formatted)
            self._update_count()

    def _set_status(self, message: str) -> None:
        self.status.configure(text=message)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.generate_btn.configure(state=state)
        if not busy and self.result is not None:
            self.play_btn.configure(state="normal")
            self.save_btn.configure(state="normal")

    def on_format(self) -> None:
        """Run the browser UI's Auto-Format, in place and visibly."""
        text = self.text.get("1.0", "end-1c")
        if not text.strip():
            self._set_status("Nothing to format.")
            return
        try:
            formatted = self.controller.format_text(text)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Auto-Format failed: {exc}")
            return
        if formatted != text:
            self._replace_text(formatted)
            self._update_count()
            self._set_status("Auto-formatted: legal text cleaned and title-cased.")
        else:
            self._set_status("Text was already formatted.")

    def on_generate(self) -> None:
        if self._busy:
            return
        text = self.text.get("1.0", "end-1c")
        voice = self.voice.get()
        if not voice:
            messagebox.showwarning(APP_TITLE, "No voice selected.")
            return

        # Auto-Format here too, so the audio is always built from the formatted
        # text and the box shows exactly what is being spoken - the same contract
        # the web UI enforces before it posts. The formatting itself happens in
        # SynthesisController; clean=False below prevents a second pass. This
        # runs before the busy state is set, so a failure leaves the window
        # responsive.
        if self.clean.get():
            try:
                formatted = self.controller.format_text(text)
            except Exception as exc:
                messagebox.showerror(APP_TITLE, f"Auto-Format failed: {exc}")
                return
            if formatted != text:
                self._replace_text(formatted)
                self._update_count()
            text = formatted

        self.result = None
        self.play_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self._set_busy(True)
        self.progress.start(12)
        self._set_status("Synthesizing...")

        args = dict(text=text, voice=voice, speed=round(self.speed.get(), 2),
                    fmt=self.fmt.get(), clean=False)
        threading.Thread(target=self._run_synthesis, kwargs=args, daemon=True).start()

    def _run_synthesis(self, **kwargs: object) -> None:
        """Worker thread: never touches Tk, only the queue."""
        try:
            result = self.controller.synthesize(**kwargs)  # type: ignore[arg-type]
            self.queue.put(("done", result))
        except Exception as exc:  # surfaced in the status bar
            self.queue.put(("error", exc))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, kind: str, payload: object) -> None:
        self.progress.stop()
        if kind == "error":
            self._set_busy(False)
            self._set_status(f"Error: {payload}")
            messagebox.showerror(APP_TITLE, str(payload))
            return
        # Store the result first: _set_busy reads it to decide whether the
        # Play and Save buttons become clickable.
        assert isinstance(payload, AudioResult)
        self.result = payload
        self._set_busy(False)
        estimated = estimate_seconds(payload.text, self.speed.get())
        self._set_status(
            f"Done: {format_duration(payload.duration)} of audio "
            f"(estimated {format_duration(estimated)}), {payload.chunks} chunk(s), "
            f"{len(payload.audio) / 1024:.0f} KB {payload.fmt.upper()}."
        )

    def on_play(self) -> None:
        if self.result is not None:
            self.controller.play(self.result)

    def on_save(self) -> None:
        if self.result is None:
            return
        ext = self.result.fmt
        path = filedialog.asksaveasfilename(
            title="Save audio",
            defaultextension=f".{ext}",
            initialfile=f"kokoro.{ext}",
            filetypes=[(f"{ext.upper()} audio", f"*.{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_bytes(self.result.audio)
        self._set_status(f"Saved {path}")

    def on_clear(self) -> None:
        self.text.delete("1.0", "end")
        self.result = None
        self.play_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self._update_count()
        self._set_status("Cleared.")


def main(argv: list[str] | None = None) -> int:
    """Configure offline weights, open the window, run the Tk main loop."""
    # Must come before anything imports kokoro, which configures loguru with
    # sys.stderr at import time.
    ensure_standard_streams()
    found = weights.configure_offline_weights()
    root = tk.Tk()
    app = DesktopApp(root)
    if found is None:
        app._set_status(
            "No bundled weights found; the model will download on first Generate."
        )
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - GUI entry
    raise SystemExit(main())
