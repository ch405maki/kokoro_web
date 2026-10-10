"""Tests for the config.json rule-editing helpers.

Run with:

    .venv\\Scripts\\python.exe -m unittest tests.test_config_store -v

Every test redirects the config discovery to a temp copy so the real
config.json and the rules of other test modules are never touched.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import legal_preprocessor as lp  # noqa: E402
from legal_preprocessor import preprocess_legal_text  # noqa: E402


def clean(text: str) -> str:
    """Preprocess and strip the single trailing newline the pipeline adds."""
    return preprocess_legal_text(text).strip()


class ConfigStoreTestCase(unittest.TestCase):
    """Isolates load_config/save_config behind a temp config.json."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.json"
        patchers = [
            mock.patch.object(lp, "_CONFIG_CACHE", None),
            mock.patch.object(lp, "_CONFIG_MISSING_REPORTED", False),
            mock.patch.object(lp, "_config_candidates", return_value=[self.path]),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_config(self, data: dict) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def read_config(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


class TestAddAbbreviation(ConfigStoreTestCase):
    def test_new_abbreviation_persists_and_applies(self):
        self.write_config({"abbreviations": {}})
        result = lp.add_abbreviation("ACME", "Acme Corporation")
        self.assertEqual(self.read_config()["abbreviations"]["ACME"], "Acme Corporation")
        self.assertIn("ACME", result["abbreviations"])
        self.assertEqual(clean("the ACME ruling"), "the Acme Corporation ruling")

    def test_existing_key_is_replaced(self):
        self.write_config({"abbreviations": {"ACME": "Old Name"}})
        lp.add_abbreviation("ACME", "New Name")
        self.assertEqual(clean("the ACME policy"), "the New Name policy")

    def test_add_is_idempotent(self):
        self.write_config({"abbreviations": {"ACME": "Acme Corporation"}})
        before = self.read_config()
        lp.add_abbreviation("ACME", "Acme Corporation")
        self.assertEqual(self.read_config(), before)

    def test_empty_input_is_rejected(self):
        self.write_config({"abbreviations": {}})
        for abv, full in (("", "X"), ("X", "  "), ("a\tb", "X")):
            with self.assertRaises(ValueError):
                lp.add_abbreviation(abv, full)


class TestRemoveAbbreviation(ConfigStoreTestCase):
    def test_user_entry_is_deleted(self):
        self.write_config({"abbreviations": {"ABBY": "Abby Incorporated"}})
        lp.remove_abbreviation("ABBY")
        self.assertNotIn("ABBY", self.read_config()["abbreviations"])
        self.assertEqual(clean("the ABBY case"), "the ABBY case")

    def test_builtin_default_is_disabled_not_deleted(self):
        self.write_config({"abbreviations": {}})
        lp.remove_abbreviation("No.")
        stored = self.read_config()["abbreviations"]
        self.assertEqual(stored.get("No."), "")
        self.assertIn("No.", lp.DEFAULTS["abbreviations"])
        self.assertEqual(clean("No. 5 applies"), "No. 5 applies")

    def test_removing_absent_rule_is_a_noop(self):
        self.write_config({"abbreviations": {"KEEP": "Kept"}})
        before = self.read_config()
        result = lp.remove_abbreviation("NOWHERE")
        self.assertEqual(self.read_config(), before)
        self.assertIn("KEEP", result["abbreviations"])


class TestAddWatermark(ConfigStoreTestCase):
    def test_garble_persists_and_is_removed(self):
        self.write_config({"watermark_markers": []})
        lp.add_watermark("GARBX1")
        self.assertIn("GARBX1", self.read_config()["watermark_markers"])
        self.assertEqual(clean("the ruling GARBX1 stands"), "the ruling stands")

    def test_matches_case_insensitively(self):
        self.write_config({"watermark_markers": []})
        lp.add_watermark("GARBX1")
        self.assertEqual(clean("garbx1 and GARBX1 both go"), "and both go")

    def test_builtins_apply_when_rule_unlisted(self):
        garble = lp.DEFAULTS["watermark_markers"][0]
        self.write_config({})
        self.assertNotIn(garble, clean("X %s Y" % garble))

    def test_add_is_idempotent_when_already_present(self):
        garble = lp.DEFAULTS["watermark_markers"][0]
        self.write_config({"watermark_markers": [garble]})
        result = lp.add_watermark(garble)
        stored = self.read_config()["watermark_markers"]
        self.assertEqual(stored.count(garble), 1)
        self.assertIn(garble, result["watermark_markers"])

    def test_empty_word_is_rejected(self):
        self.write_config({"watermark_markers": []})
        with self.assertRaises(ValueError):
            lp.add_watermark("  ")


class TestRemoveWatermark(ConfigStoreTestCase):
    def test_user_garble_is_deleted(self):
        self.write_config({"watermark_markers": ["my.garble"]})
        lp.remove_watermark("my.garble")
        self.assertNotIn("my.garble", self.read_config()["watermark_markers"])
        self.assertEqual(clean("my.garble stays"), "my.garble stays")

    def test_builtin_garble_can_be_disabled(self):
        # With no watermark key in the file, the built-in markers are in
        # effect; removing one rewrites the list without it (list semantics
        # mean a config list replaces the defaults rather than merging).
        garble = lp.DEFAULTS["watermark_markers"][0]
        self.write_config({})
        before = clean("X %s Y" % garble)
        self.assertNotIn(garble, before)
        lp.remove_watermark(garble)
        stored = self.read_config()["watermark_markers"]
        self.assertNotIn(garble, stored)
        after = clean("X %s Y" % garble)
        self.assertIn(garble, after)


class TestPersistence(ConfigStoreTestCase):
    def test_comment_keys_survive_and_unicode_is_written_literal(self):
        garble = lp.DEFAULTS["watermark_markers"][0]
        self.write_config({"_comment": "do not touch", "watermark_markers": [garble]})
        result = lp.add_watermark("ZZGARBLE")
        stored = self.read_config()
        self.assertEqual(stored["_comment"], "do not touch")
        self.assertIn("ZZGARBLE", stored["watermark_markers"])
        raw = self.path.read_text(encoding="utf-8")
        self.assertIn(garble, raw)
        self.assertNotIn("\\u", raw)
        self.assertIn("\n\n  \"watermark_markers\"", raw)
        self.assertIn("ZZGARBLE", result["watermark_markers"])

    def test_new_config_created_when_none_exists(self):
        missing = Path(self.tmp.name) / "brand-new.json"
        with mock.patch.object(
            lp, "_config_candidates", return_value=[missing]
        ), mock.patch.object(
            lp, "_default_config_path", return_value=missing
        ):
            result = lp.add_abbreviation("MEPO", "Mandaluyong East Police Office")
            self.assertTrue(missing.is_file())
            self.assertEqual(
                result["abbreviations"]["MEPO"],
                "Mandaluyong East Police Office",
            )

    def test_malformed_existing_file_is_replaced(self):
        self.write_config({"abbreviations": "not-a-dict"})
        result = lp.add_abbreviation("AAA", "Triple A")
        self.assertEqual(result["abbreviations"]["AAA"], "Triple A")
        self.assertEqual(self.read_config()["abbreviations"], {"AAA": "Triple A"})


if __name__ == "__main__":
    unittest.main()