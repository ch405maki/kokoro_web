"""Tests for legal_preprocessor.

Run with:

    .venv\\Scripts\\python.exe -m unittest tests.test_preprocessor -v

The examples come straight from the preprocessing specification, plus
regressions for the bugs found while building it.
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
from legal_preprocessor import number_to_words, preprocess_legal_text  # noqa: E402


def clean(text: str) -> str:
    """Preprocess and strip the single trailing newline the pipeline adds."""
    return preprocess_legal_text(text).strip()


class TestFootnoteAndMarkerRemoval(unittest.TestCase):
    """Step A - attached footnote references and editorial noise."""

    def test_digits_fused_to_word_are_removed(self):
        self.assertEqual(clean("Certiorari1"), "Certiorari")
        self.assertEqual(clean("respondents.2"), "respondents.")

    def test_bracketed_numeric_references_are_removed(self):
        self.assertEqual(clean("See [1] and [33] and [2] here."), "See and and here.")

    def test_ocr_marker_variants_are_removed(self):
        self.assertEqual(clean("a misprint (awÞhi( here"), "a misprint here")
        self.assertEqual(clean("plain (awÞhi here"), "plain here")

    def test_sic_and_editorial_apparatus_are_removed(self):
        self.assertEqual(clean("the ruling [sic] stands"), "the ruling stands")
        self.assertEqual(clean("per [a.f.] the Court"), "per the Court")
        self.assertEqual(clean("[Emphasis in the original] judgment"), "judgment")
        self.assertEqual(clean("(Emphasis in the original) judgment"), "judgment")
        self.assertEqual(clean("(Emphasis supplied) ruling"), "ruling")
        self.assertEqual(clean("(Citations omitted) here"), "here")
        self.assertEqual(
            clean("(Emphasis supplied, citations omitted) here"), "here"
        )

    def test_singletons_go_but_merged_letters_survive(self):
        # The order matters: [j]ust is a single bracket plus a word, while
        # [j] [t] is two isolated brackets that must both disappear.
        self.assertEqual(clean("[j]ust [t]he [i]nternet"), "just the internet")
        self.assertEqual(clean("[s] [j] [t] alone"), "alone")

    def test_real_numbers_are_not_mistaken_for_footnotes(self):
        # Regression: the footnote rule used to delete the ",000.00" tail of an
        # amount, because those digits followed a comma.
        self.assertEqual(clean("COVID-19"), "COVID-19")
        self.assertEqual(clean("P1,234.56"), "One Thousand Two Hundred Thirty-Four Pesos and Fifty-Six Centavos")


class TestOcrFixes(unittest.TestCase):
    """Step B - digit-for-letter confusions from the scanner."""

    def test_spec_examples(self):
        self.assertEqual(
            clean("prope1iies of the parties"), "properties of the parties"
        )
        self.assertEqual(clean("th1s 1s the ruling"), "this is the ruling")

    def test_digits_inside_words_are_left_to_other_rules(self):
        # "1s" is only an abbreviation swap when it is a whole word.
        self.assertEqual(clean("R2D2 1s here"), "R2D is here")

    def test_hyphenated_and_spaced_numbers_survive(self):
        # The footnote rule only fires on digits fused to the end of a word, so
        # the hyphenated spellings that appear in real orders are untouched.
        self.assertEqual(clean("COVID-19 and SARS-COV-2"), "COVID-19 and SARS-COV-2")
        self.assertEqual(clean("COVID 19 cases"), "COVID 19 cases")

    def test_excess_whitespace_is_collapsed(self):
        self.assertEqual(clean("a     b"), "a b")


class TestAbbreviations(unittest.TestCase):
    """Step C - configured standalone abbreviations."""

    def test_citation_markers(self):
        self.assertEqual(clean("G.R. No. 64100"), "G.R. Number 64100")
        self.assertEqual(clean("G.R. Nos. 64100, 64101"), "G.R. Numbers 64100, 64101")
        self.assertEqual(clean("CA-G.R. SP No. 12345"), "CA-G.R. SP Number 12345")
        self.assertEqual(clean("NLRC LAC No. 5"), "NLRC LAC Number 5")
        self.assertEqual(clean("NLRC NCR Case No. 9"), "NLRC NCR Case Number 9")

    def test_agency_expansions(self):
        self.assertEqual(
            clean("per RTC and NLRC"),
            "per Regional Trial Court and National Labor Relations Commission",
        )
        self.assertEqual(clean("the CA ruled"), "the Court of Appeals ruled")

    def test_no_is_only_a_citation_marker_before_a_number(self):
        self.assertEqual(clean("No person shall do this"), "No person shall do this")

    def test_hyphenated_acronyms_are_not_split_apart(self):
        # Regression: bare "CA" must not eat the "CA-" of "CA-G.R.".
        self.assertEqual(clean("not CA-G.R. 1"), "not CA-G.R. 1")

    def test_longest_key_wins_and_output_is_not_rescanned(self):
        # Regression: substituting "NLRC LAC No." and then "NLRC" used to
        # rewrite the acronym inside its own replacement, yielding
        # "National Labor Relations Commission LAC Number".
        self.assertEqual(clean("NLRC LAC No. 5"), "NLRC LAC Number 5")
        self.assertEqual(clean("NLRC NCR Case No. 9"), "NLRC NCR Case Number 9")

    def test_abbr_period_does_not_gain_a_space(self):
        # Regression: the spacing pass turned "G.R. Number" into "G. R. Number".
        self.assertEqual(clean("G.R. No. 64100"), "G.R. Number 64100")
        self.assertEqual(clean("CA-G.R. SP No. 12345"), "CA-G.R. SP Number 12345")

    def test_ordinary_sentence_endings_still_get_a_space(self):
        self.assertEqual(clean("The order was reversed.The court ruled"), "The order was reversed. The court ruled")

    def test_underscore_comment_keys_are_ignored(self):
        # config.json carries human-readable notes as "_comment" keys; they
        # must never be treated as abbreviations to expand.
        for key in ("_comment", "_caution"):
            self.assertNotIn(key, {k for k, _ in lp._config_pairs(lp.load_config()["abbreviations"])})


class TestNumbers(unittest.TestCase):
    """Step D and number_to_words."""

    def test_currency_usd(self):
        self.assertEqual(clean("USD 641.00"), "Six Hundred Forty-One United States Dollars")
        self.assertEqual(clean("USD 60,000.00"), "Sixty Thousand United States Dollars")
        self.assertEqual(
            clean("USD 104,866.00"),
            "One Hundred Four Thousand Eight Hundred Sixty-Six United States Dollars",
        )

    def test_currency_php_and_centavos(self):
        self.assertEqual(clean("PHP 400,000.00"), "Four Hundred Thousand Pesos")
        self.assertEqual(clean("PHP 50,000.00"), "Fifty Thousand Pesos")
        self.assertEqual(clean("PHP 611,039.00"), "Six Hundred Eleven Thousand Thirty-Nine Pesos")
        self.assertEqual(
            clean("PHP 91,575.65"),
            "Ninety-One Thousand Five Hundred Seventy-Five Pesos and Sixty-Five Centavos",
        )

    def test_amounts_keep_their_grouping_and_cents(self):
        # Regression: the footnote rule used to delete ",000.00" from a price
        # because those digits followed a comma.
        self.assertEqual(clean("The award is P100,000.00."), "The award is One Hundred Thousand Pesos.")
        self.assertEqual(clean("Total: P1,234.56"), "Total: One Thousand Two Hundred Thirty-Four Pesos and Fifty-Six Centavos")

    def test_percentages_are_lower_case(self):
        self.assertEqual(clean("10%"), "ten percent")
        self.assertEqual(clean("50%"), "fifty percent")
        self.assertEqual(clean("100%"), "one hundred percent")

    def test_number_to_words_examples(self):
        self.assertEqual(number_to_words(641), "Six Hundred Forty-One")
        self.assertEqual(number_to_words(60000), "Sixty Thousand")
        self.assertEqual(
            number_to_words(104866),
            "One Hundred Four Thousand Eight Hundred Sixty-Six",
        )
        self.assertEqual(number_to_words(611039), "Six Hundred Eleven Thousand Thirty-Nine")

    def test_number_to_words_edges(self):
        self.assertEqual(number_to_words(0), "Zero")
        self.assertEqual(number_to_words(100), "One Hundred")
        self.assertEqual(number_to_words(101), "One Hundred One")
        self.assertEqual(number_to_words(1000), "One Thousand")
        self.assertEqual(number_to_words(1000000), "One Million")

    def test_case_numbers_are_left_as_digits(self):
        # Only currency and percentages are spelled out; a case number must
        # stay intact for the reader.
        self.assertEqual(clean("G.R. No. 64100"), "G.R. Number 64100")


class TestProtectedSegments(unittest.TestCase):
    """Quoted and blockquoted text must survive byte for byte."""

    def test_double_quoted_text_is_untouched(self):
        src = 'The court said "G.R. No. 64100, USD 50,000.00" today.'
        self.assertEqual(clean(src), 'The court said "G.R. No. 64100, USD 50,000.00" today.')

    def test_blockquoted_text_is_untouched(self):
        src = "> It is  (awÞhi( ordered[1].\nThe court agreed."
        out = clean(src)
        self.assertIn("> It is  (awÞhi( ordered[1].", out)
        self.assertIn("The court agreed.", out)

    def test_quote_spanning_several_lines(self):
        src = 'He said "G.R. No. 64100\nand 1s more" loudly.'
        out = clean(src)
        self.assertIn('"G.R. No. 64100', out)
        self.assertIn("and 1s more", out)

    def test_cleaning_outside_quotes_still_happens(self):
        self.assertEqual(clean('per [1] "quoted" Certiorari1'), 'per "quoted" Certiorari')


class TestWhitespaceAndStructure(unittest.TestCase):
    """Step E."""

    def test_blank_runs_collapse_to_a_single_blank_line(self):
        self.assertEqual(clean("a\n\n\n\nb"), "a\n\nb")

    def test_lines_are_rstripped(self):
        self.assertEqual(clean("a   \nb"), "a\nb")

    def test_space_after_punctuation(self):
        self.assertEqual(clean("a,b"), "a, b")
        self.assertEqual(clean("a:b"), "a: b")
        self.assertEqual(clean("a;b"), "a; b")
        self.assertEqual(clean("a.  b"), "a. b")


class TestConfig(unittest.TestCase):
    """Configuration loading and resilience."""

    def test_repo_config_is_used_and_is_valid_json(self):
        config = lp.load_config(reload=True)
        self.assertEqual(config["abbreviations"]["No."], "Number")
        self.assertEqual(config["currency_map"]["USD"], "United States Dollars")
        raw = Path(__file__).resolve().parents[1] / "config.json"
        self.assertTrue(raw.is_file(), "config.json should ship with the repo")
        json.loads(raw.read_text(encoding="utf-8"))

    def test_missing_config_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope.json")
            with mock.patch.object(lp, "_config_candidates", return_value=[Path(missing)]):
                config = lp.load_config(reload=True)
        self.assertEqual(config["abbreviations"]["No."], "Number")
        self.assertEqual(
            config["abbreviations"]["CA"], lp.DEFAULTS["abbreviations"]["CA"]
        )

    def test_malformed_config_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            with mock.patch.object(lp, "_config_candidates", return_value=[bad]):
                config = lp.load_config(reload=True)
        self.assertEqual(config["abbreviations"]["No."], "Number")

    def test_empty_input(self):
        self.assertEqual(clean(""), "")

    def test_kokoro_pause_markers_survive(self):
        # The web UI documents "/" at the start of a word as a short pause, and
        # the engine's chunker reads it. The preprocessor must not swallow it
        # or mangle the word that follows.
        self.assertEqual(clean("/hello world"), "/hello world")
        self.assertEqual(clean("/G.R. No. 64100 and /goodbye"), "/G.R. Number 64100 and /goodbye")


class TestPipelineBehaviour(unittest.TestCase):
    def test_input_is_not_mutated(self):
        src = "Certiorari1  [1]  G.R. No. 64100"
        copy = str(src)
        preprocess_legal_text(src)
        self.assertEqual(src, copy)

    def test_multiline_document_end_to_end(self):
        src = (
            "DECISION\n"
            "\n"
            "Certiorari1  [1]\n"
            "G.R. No. 64100  November 8, 1989\n"
            "\n"
            "PER     CURIAM:\n"
            "\n"
            "Upon   review,  the RTC ruled  that  prope1iies  of  the  parties\n"
            "were  misappropriated  in  the  amount  of  P100,000.00.\n"
            "The  CA  affirmed  in  CA-G.R. SP No. 12345.\n"
            "\n"
            "Hence, judgment is affirmed.\n"
        )
        out = clean(src)
        self.assertIn("PER CURIAM:", out)
        self.assertIn("G.R. Number 64100", out)
        self.assertIn("properties", out)
        self.assertIn("One Hundred Thousand Pesos", out)
        self.assertIn("CA-G.R. SP Number 12345", out)
        self.assertIn("The Court of Appeals affirmed", out)
        self.assertNotIn("[1]", out)
        self.assertNotIn("Certiorari1", out)
        self.assertNotIn("  ", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
