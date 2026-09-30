"""Shared Kokoro-82M inference layer.

Single source of truth for model loading + synthesis so the CLI, the REST API
and the web UI all behave identically.
"""

from __future__ import annotations

import importlib
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

# kokoro 0.9.4 still calls the pre-2.1 torch weight_norm API and constructs
# LSTM layers with a dropout arg newer torch warns about. Both still work.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("KOKORO_OUTPUT_DIR", ROOT / "output"))
MODEL_REPO = os.getenv("KOKORO_MODEL_REPO", "hexgrad/Kokoro-82M")

SAMPLE_RATE = 24000
DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "af_heart")
DEFAULT_LANG = os.getenv("KOKORO_DEFAULT_LANG", "a")

# ---------------------------------------------------------------- device setup

def _configure_threads() -> None:
    """Kokoro is a small model; thread thrash makes it slower, not faster."""
    n = int(os.getenv("KOKORO_THREADS", "0")) or (os.cpu_count() or 4)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


_configure_threads()


def detect_device() -> str:
    """cuda > mps > cpu, with a KOKORO_DEVICE override."""
    forced = os.getenv("KOKORO_DEVICE")
    if forced:
        return forced
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep of kokoro
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------- voices

# name -> (gender, english quality tier, approx training duration)
VOICE_CATALOG: dict[str, dict[str, object]] = {}


def _load_voice_catalog() -> None:
    if VOICE_CATALOG:
        return
    catalog: dict[str, dict[str, object]] = {}
    try:
        from huggingface_hub import HfApi

        names: set[str] = set()
        for f in HfApi().list_repo_files(MODEL_REPO):
            m = re.match(r"voices/([a-z]{1,2}_[a-z]+)\.pt$", f)
            if m:
                names.add(m.group(1))
    except Exception:
        names = set()

    # Fallback: the voice list published in the upstream README.
    if not names:
        names = {
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george",
            "bm_lewis",
        }

    excellent = {"heart", "bella", "nicole", "sarah", "sky", "adam",
                 "michael", "george", "emma", "isabella", "lewis"}
    available = available_lang_codes()
    for name in sorted(names):
        lang, who = name.split("_", 1)
        # Naming scheme: <locale><gender>_<character>, e.g. af_heart, bf_emma.
        gender = {"f": "female", "m": "male"}.get(lang[1]) if len(lang) > 1 else None
        catalog[name] = {
            "name": name,
            "lang_code": lang,
            "locale": {"a": "American English", "b": "British English",
                       "e": "Spanish", "f": "French", "h": "Hindi",
                       "i": "Italian", "j": "Japanese", "p": "Brazilian Portuguese",
                       "z": "Mandarin Chinese"}.get(lang[0], lang),
            "description": who.replace("_", " "),
            "gender": gender or "neutral",
            "quality": "excellent" if who in excellent else "good",
            "available": lang[0] in available,
        }
    VOICE_CATALOG.update(catalog)


def list_voices(lang: str | None = None) -> list[dict[str, object]]:
    """List voices, optionally narrowed to a locale.

    ``lang`` accepts either a locale prefix ('a', 'b', 'z') or a full code
    ('af', 'bf'), so ``?lang=b`` and ``?lang=bf`` both mean British English.
    """
    _load_voice_catalog()
    voices = list(VOICE_CATALOG.values())
    if lang:
        code = lang.lower()
        voices = [v for v in voices if str(v["lang_code"]).startswith(code)]
    return voices


def resolve_lang_code(voice: str) -> str:
    """kokoro requires lang_code to match the voice prefix."""
    prefix = voice.split("_", 1)[0]
    if prefix == "z":  # Mandarin
        return "z"
    if prefix == "j":  # Japanese
        return "j"
    return prefix[:1]


# How kokoro dispatches G2P per language (see kokoro/pipeline.py LANG_CODES):
#   a, b  -> misaki.en.G2P + espeak-ng fallback   (pip: misaki[en])
#   j     -> misaki.ja.JAG2P                     (pip: misaki[ja])
#   z     -> misaki.zh.ZHG2P                     (pip: misaki[zh])
#   e/f/h/i/p -> misaki.espeak.EspeakG2P         (no extra pip package)
# misaki.espeak configures phonemizer's espeak-ng paths at import time, so the
# espeak locales are usable whenever espeakng-loader is present.
_MISAKI_MODULE_FOR_PREFIX = {"a": "en", "b": "en", "j": "ja", "z": "zh"}
ESPEAK_PREFIXES = ("e", "f", "h", "i", "p")
KNOWN_PREFIXES = ("a", "b", *ESPEAK_PREFIXES, "j", "z")


@lru_cache(maxsize=1)
def available_lang_codes() -> set[str]:
    """Which locales can actually be spoken given the installed packages."""
    available: set[str] = set()

    for prefix, module in _MISAKI_MODULE_FOR_PREFIX.items():
        try:
            importlib.import_module(f"misaki.{module}")
            available.add(prefix)
        except Exception:
            continue

    # es, fr, hi, it, pt-br all go through espeak-ng and need no pip extra.
    try:
        importlib.import_module("misaki.espeak")
        available.update(ESPEAK_PREFIXES)
    except Exception:
        pass

    return available or {"a"}


# ------------------------------------------------------------------- pipeline

@dataclass
class Synthesized:
    graphemes: str
    phonemes: str
    audio: np.ndarray
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration = round(len(self.audio) / SAMPLE_RATE, 3)


class UnsupportedLanguage(ValueError):
    """Raised when the requested voice's G2P extras are not installed."""


class KokoroEngine:
    """Thread-safe wrapper around KPipeline."""

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}
        self._lock = threading.Lock()
        self._ready = False
        self.device = detect_device()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_seconds(self) -> float | None:
        return getattr(self, "_load_seconds", None)

    def _get_pipeline(self, lang_code: str):
        pipe = self._pipelines.get(lang_code)
        if pipe is not None:
            return pipe
        with self._lock:
            pipe = self._pipelines.get(lang_code)
            if pipe is not None:
                return pipe
            t0 = time.perf_counter()
            from kokoro import KPipeline

            pipe = KPipeline(lang_code=lang_code, repo_id=MODEL_REPO, device=self.device)
            self._pipelines[lang_code] = pipe
            self._ready = True
            if getattr(self, "_load_seconds", None) is None:
                self._load_seconds = round(time.perf_counter() - t0, 2)
            return pipe

    def warm_up(self) -> dict[str, object]:
        self._get_pipeline(DEFAULT_LANG)
        return {
            "device": self.device,
            "load_seconds": self.load_seconds,
            "voices": len(list_voices()),
        }

    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        split_pattern: str | None = None,
    ) -> list[Synthesized]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        lang_code = resolve_lang_code(voice)
        if lang_code not in available_lang_codes():
            raise UnsupportedLanguage(
                f"voice '{voice}' needs a language pack that is not installed. "
                f"Usable now: {', '.join(sorted(available_lang_codes()))}. "
                f"Add more with: uv pip install 'misaki[ja]'  (or misaki[zh])"
            )
        pipe = self._get_pipeline(lang_code)

        chunks: list[Synthesized] = []
        # The generator is not re-entrant, so hold the lock for the whole pass.
        with self._lock:
            kwargs = {"voice": voice, "speed": speed}
            if split_pattern:
                kwargs["split_pattern"] = split_pattern
            for gs, ps, audio in pipe(text, **kwargs):
                chunks.append(Synthesized(graphemes=gs, phonemes=ps, audio=audio))
        return chunks


@lru_cache(maxsize=1)
def get_engine() -> KokoroEngine:
    return KokoroEngine()


# ----------------------------------------------------------------- encoding

def chunks_to_wav(chunks: list[Synthesized], gap: float = 0.0) -> bytes:
    """Concatenate chunks into a single in-memory WAV file.

    ``gap`` inserts silence between chunks, which makes multi-paragraph input
    sound far more natural than a hard butt-joint.
    """
    import io

    import soundfile as sf

    arrays = [np.asarray(c.audio, dtype=np.float32) for c in chunks]
    if not arrays:
        raise ValueError("no audio produced")
    if len(arrays) == 1 or gap <= 0:
        merged = np.concatenate(arrays)
    else:
        silence = np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)
        merged = np.concatenate(
            [part for a in arrays for part in (a, silence)][:-1]
        )
    merged = np.clip(merged, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, merged, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def chunks_to_mp3(chunks: list[Synthesized], bitrate: str = "192k",
                  gap: float = 0.0) -> bytes:
    """Requires ffmpeg on PATH (used only when the caller asks for mp3)."""
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; request format=wav instead")

    wav = chunks_to_wav(chunks, gap=gap)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.mp3"
        src.write_bytes(wav)
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(src),
             "-codec:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", str(dst)],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:400])
        return dst.read_bytes()


CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}


def encode(chunks: list[Synthesized], fmt: str = "wav",
           gap: float = 0.0) -> tuple[bytes, str]:
    """Serialise synthesised chunks, inserting ``gap`` seconds of silence
    between them."""
    fmt = fmt.lower().lstrip(".")
    if fmt == "wav":
        return chunks_to_wav(chunks, gap=gap), CONTENT_TYPES["wav"]
    if fmt == "mp3":
        return chunks_to_mp3(chunks, gap=gap), CONTENT_TYPES["mp3"]
    raise ValueError(f"unsupported format: {fmt}")
