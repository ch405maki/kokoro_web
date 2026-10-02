"""Tests for the standalone desktop building blocks (no window, no torch).

Run with:

    .venv\\Scripts\\python.exe -m unittest tests.test_desktop -v

The GUI itself is not exercised here; only the controller and the weights
resolution, which is where the logic that can break silently lives.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import weights  # noqa: E402
from app.desktop import (  # noqa: E402
    AudioResult,
    SynthesisController,
    ensure_standard_streams,
    estimate_seconds,
    format_duration,
    title_case,
)


class _StubEngine:
    """A stand-in for app.engine that records the calls made to it."""

    def __init__(self, chunks=None) -> None:
        self.calls: list[tuple] = []
        self.chunks = chunks if chunks is not None else [
            types.SimpleNamespace(duration=1.25, audio=[0.0, 0.1]),
            types.SimpleNamespace(duration=0.75, audio=[0.2]),
        ]
        self.DEFAULT_VOICE = "af_heart"

    def list_voices(self, lang: str | None = None) -> list[dict]:
        return [
            {"name": "af_heart", "available": True},
            {"name": "zf_xiaobei", "available": False},
        ]

    def get_engine(self):
        def synthesize(text, voice, speed, split_pattern=None):
            self.calls.append(("synthesize", text, voice, speed))
            return self.chunks

        return types.SimpleNamespace(synthesize=synthesize)

    def encode(self, chunks, fmt="wav", gap=0.0, bitrate=None):
        self.calls.append(("encode", fmt))
        return f"<{fmt}>".encode(), f"audio/{fmt}"


class TestSynthesisController(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _StubEngine()
        self.cleaned: list[str] = []

        def preprocess(text: str) -> str:
            self.cleaned.append(text)
            return "CLEANED"

        self.controller = SynthesisController(engine_module=self.engine,
                                              preprocessor=preprocess)

    def test_unavailable_voices_are_hidden(self):
        self.assertEqual([v["name"] for v in self.controller.voices()], ["af_heart"])

    def test_default_voice_falls_back_to_the_first_when_missing(self):
        self.assertEqual(self.controller.default_voice(), "af_heart")

    def test_cleaning_runs_by_default_and_feeds_the_engine(self):
        result = self.controller.synthesize(text="raw text", voice="af_heart", speed=1.0)
        self.assertEqual(self.cleaned, ["raw text"])
        self.assertIsInstance(result, AudioResult)
        self.assertEqual(result.text, "CLEANED")
        self.assertIn(("synthesize", "CLEANED", "af_heart", 1.0), self.engine.calls)

    def test_cleaning_can_be_skipped(self):
        result = self.controller.synthesize(text="raw text", voice="af_heart",
                                             speed=1.0, clean=False)
        self.assertEqual(self.cleaned, [])
        self.assertEqual(result.text, "raw text")

    def test_wav_encodes_once_and_reuses_it_for_playback(self):
        result = self.controller.synthesize(text="x", voice="af_heart", speed=1.0,
                                             fmt="wav", clean=False)
        self.assertEqual(result.audio, result.playback_wav)
        self.assertEqual([c for c in self.engine.calls if c[0] == "encode"], [("encode", "wav")])

    def test_mp3_also_keeps_a_wav_copy_for_playback(self):
        result = self.controller.synthesize(text="x", voice="af_heart", speed=1.0,
                                             fmt="mp3", clean=False)
        self.assertEqual(result.fmt, "mp3")
        self.assertEqual(result.audio, b"<mp3>")
        self.assertEqual(result.playback_wav, b"<wav>")
        self.assertEqual([c for c in self.engine.calls if c[0] == "encode"],
                         [("encode", "wav"), ("encode", "mp3")])

    def test_duration_is_the_sum_of_the_chunks(self):
        result = self.controller.synthesize(text="x", voice="af_heart", speed=1.0,
                                             clean=False)
        self.assertEqual(result.duration, 2.0)
        self.assertEqual(result.chunks, 2)

    def test_empty_input_after_cleaning_is_rejected(self):
        controller = SynthesisController(engine_module=self.engine,
                                         preprocessor=lambda text: "   ")
        with self.assertRaises(ValueError):
            controller.synthesize(text="x", voice="af_heart", speed=1.0)
        self.assertEqual([c for c in self.engine.calls if c[0] == "synthesize"], [])

    def test_no_audio_is_rejected(self):
        engine = _StubEngine(chunks=[])
        controller = SynthesisController(engine_module=engine,
                                         preprocessor=lambda text: text)
        with self.assertRaises(ValueError):
            controller.synthesize(text="x", voice="af_heart", speed=1.0)

    def test_default_preprocessor_is_the_legal_cleaner(self):
        controller = SynthesisController(engine_module=self.engine)
        result = controller.synthesize(text="Court of Appeals (CA) ruled.",
                                       voice="af_heart", speed=1.0)
        self.assertEqual(result.text.strip(), "Court of Appeals ruled.")


class TestFormatting(unittest.TestCase):
    def test_title_case_capitalises_and_lowercases(self):
        self.assertEqual(title_case("SAMPLE TEXT"), "Sample Text")

    def test_small_words_stay_lowercase_inside_a_title(self):
        self.assertEqual(title_case("the court of appeals"), "The Court of Appeals")

    def test_sentence_starts_are_forced_capital(self):
        self.assertEqual(title_case("hello. world"), "Hello. World")
        self.assertEqual(title_case("the end. the beginning"), "The End. The Beginning")

    def test_apostrophes_are_preserved(self):
        self.assertEqual(title_case("don't stop"), "Don't Stop")

    def test_non_word_characters_are_left_alone(self):
        self.assertEqual(title_case("a/b  [j]  10%"), "A/B  [J]  10%")

    def test_estimate_scales_with_length_and_speed(self):
        self.assertEqual(estimate_seconds("x" * 150, 1.0), 10.0)
        self.assertEqual(estimate_seconds("x" * 150, 2.0), 5.0)

    def test_duration_is_human_readable(self):
        self.assertEqual(format_duration(7.4), "7s")
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_duration(3661), "1h 01m 01s")
        self.assertEqual(format_duration(59.6), "1m 00s")

    def test_controller_format_text_cleans_then_cases(self):
        controller = SynthesisController(engine_module=_StubEngine(),
                                         preprocessor=lambda text: "CLEANED")
        self.assertEqual(controller.format_text("raw text"), "Cleaned")

    def test_default_format_text_runs_the_legal_cleaner(self):
        controller = SynthesisController(engine_module=_StubEngine())
        self.assertEqual(controller.format_text("Court of Appeals (CA) ruled.").strip(),
                         "Court of Appeals Ruled.")


class TestStandardStreams(unittest.TestCase):
    """Windowed builds have None streams; kokoro's loguru setup rejects that."""

    def test_none_streams_get_a_sink(self):
        with mock.patch.object(sys, "stdout", None), mock.patch.object(sys, "stderr", None):
            ensure_standard_streams()
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            sys.stdout.close()
            sys.stderr.close()

    def test_real_streams_are_left_untouched(self):
        sentinel = object()
        with mock.patch.object(sys, "stdout", sentinel), mock.patch.object(sys, "stderr", sentinel):
            ensure_standard_streams()
            self.assertIs(sys.stdout, sentinel)
            self.assertIs(sys.stderr, sentinel)


class TestWeightsResolution(unittest.TestCase):
    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"KOKORO_WEIGHTS_DIR": "D:/w"}, clear=True):
            self.assertEqual(weights.default_weights_dir(), Path("D:/w"))

    def test_configure_activates_a_present_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "hub").mkdir()
            with mock.patch.dict(os.environ, {}, clear=True):
                found = weights.configure_offline_weights(target)
                self.assertEqual(found, target)
                self.assertEqual(os.environ["HF_HOME"], str(target))
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_missing_cache_returns_none_and_sets_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(weights.configure_offline_weights(Path(tmp)))
                self.assertNotIn("HF_HOME", os.environ)

    def test_existing_environment_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "hub").mkdir()
            preset = {"HF_HOME": "custom", "HF_HUB_OFFLINE": "0"}
            with mock.patch.dict(os.environ, preset, clear=True):
                weights.configure_offline_weights(target)
                self.assertEqual(os.environ["HF_HOME"], "custom")
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
