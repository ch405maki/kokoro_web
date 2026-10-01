"""Clean Philippine Supreme Court decisions and similar legal text for TTS.

Raw opinions are hostile to a speech engine. Footnote markers become "Certiorari
one" and "(emphasis in the original)" is read aloud, case citations arrive with
no spaces after periods, and every peso amount is read as digits. This module
normalises all of that before the text reaches Kokoro.

The pipeline is five ordered steps, A through E, each of which runs only on the
*unprotected* parts of the input so that quoted material - block quotes marked
with ``>`` and anything between double quotes - survives byte for byte.

Two ordering decisions are deliberate and worth knowing about:

* **Standalone single-letter brackets are removed after word repairs.** Doing it
  the other way round turns ``[j]ust`` into ``ust``. A pattern like ``[s]`` is
  only treated as a stray marker when no word follows it, so the two never
  collide.
* **Punctuation spacing runs after abbreviation expansion, and skips
  abbreviations.** Expanding ``G.R. No.`` first and then blindly inserting a
  space after every period yields ``G. R. Number``. The spacing rule therefore
  skips periods belonging to short all-caps tokens, single letters, and known
  abbreviations.

Nothing here imports torch or kokoro, so it is cheap to test and can be run
standalone. Configure behaviour through ``config.json`` rather than by editing
this file.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

__all__ = ["preprocess_legal_text", "number_to_words", "load_config", "DEFAULTS"]

# --------------------------------------------------------------------------- #
# Defaults. Used when config.json is absent or incomplete, so a missing or
# malformed config degrades to built-in behaviour instead of raising.
# --------------------------------------------------------------------------- #

DEFAULTS: dict[str, Any] = {
    # Legal citations and institutional names. Longest-first application order
    # is enforced in code, so ordering here is only for readability.
    "abbreviations": {
        "G.R. No.": "G.R. Number",
        "G.R. Nos.": "G.R. Numbers",
        "CA-G.R. SP No.": "CA-G.R. SP Number",
        "NLRC LAC No.": "NLRC LAC Number",
        "NLRC NCR Case No.": "NLRC NCR Case Number",
        "No.": "Number",
        "Nos.": "Numbers",
        "RTC": "Regional Trial Court",
        "CA": "Court of Appeals",
        "NLRC": "National Labor Relations Commission",
        "LA": "labor arbiter",
        "POEA-SEC": "POEA Standard Employment Contract",
        "POEA": "Philippines Overseas Employment Administration",
        "OSG": "Office of the Solicitor General",
        "LRC": "Land Registration Court",
        "LRA": "Land Registration Authority",
        "OCT": "Original Certificate of Title",
        "TCT": "Transfer Certificate of Title",
        "GLRO": "General Land Registration Office",
        "AMOSUP": "Associated Marine Officers' and Seamen's Union of the Philippines",
        "AFP": "Armed Forces of the Philippines",
        "ILO": "International Labour Organization",
        "CBA": "collective bargaining agreement",
        "CT": "computed tomography",
        "NSAIDs": "nonsteroidal anti-inflammatory drugs",
        "MMI": "Multinational Maritime, Inc.",
        "MMS": "MMS. Co., Ltd.",
        "AJSU": "All Japan Seamen's Union",
        "NSC": "National Steel Corporation",
        "HR": "Human Resources",
        "CCTV": "closed-circuit television",
        "SeNA": "single-entry approach",
        "NCR": "National Capital Region",
        "RAB": "Regional Arbitration Branch",
        "VAC": "Voluntary Arbitration Case",
    },
    # Amounts. Keys are matched case-sensitively as a standalone token.
    "currency_map": {
        "USD": "United States Dollars",
        "PHP": "Pesos",
    },
    # Alternate spellings that mean the same currency. Philippine decisions
    # overwhelmingly use "P" or the peso sign, never "PHP", so without this the
    # currency pass would miss most real amounts.
    "currency_aliases": {
        "P": "PHP",
        "PHP": "PHP",
        "USD": "USD",
        "US$": "USD",
        "$": "USD",
        "₱": "PHP",
    },
    # Sub-centavo wording, per currency.
    "cent_subunit": {"PHP": "Centavos", "USD": "Cents"},
    # OCR digit/letter confusions, applied on word boundaries only.
    "ocr_map": {
        "prope1iies": "properties",
        "1s": "is",
    },
    # Parenthesised editorial matter, removed outright.
    "editorial_markers": [
        "Emphasis in the original",
        "Emphasis supplied",
        "Citations omitted",
    ],
}

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_MISSING_REPORTED = False


def _config_pairs(mapping: dict[str, Any]):
    """Yield real (key, value) pairs, skipping "_"-prefixed documentation keys.

    config.json carries comments as sibling string entries so the file stays
    self-documenting. Those must never be treated as abbreviations to expand.
    """
    for key, value in mapping.items():
        if key.startswith("_"):
            continue
        yield key, value


def _config_candidates() -> list[Path]:
    """Places config.json may live, most specific first."""
    here = Path(__file__).resolve().parent
    return [here / "config.json", here.parent / "config.json", Path.cwd() / "config.json"]


def load_config(reload: bool = False) -> dict[str, Any]:
    """Return the merged configuration, loading it from disk at most once.

    A missing config.json is not an error: the built-in DEFAULTS are used and a
    single warning is emitted. This keeps the app importable on a fresh clone
    and in the container image, where nobody has created the file yet.
    """
    global _CONFIG_CACHE, _CONFIG_MISSING_REPORTED

    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None and not reload:
            return _CONFIG_CACHE

        merged: dict[str, Any] = {
            key: (dict(value) if isinstance(value, dict) else list(value))
            for key, value in DEFAULTS.items()
        }

        for path in _config_candidates():
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    user = json.load(handle)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                # A broken config must not take the app down; fall back to
                # defaults for the keys we could not read.
                continue
            if not isinstance(user, dict):
                continue
            for key, value in user.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = value
                else:
                    merged[key] = value
            _CONFIG_CACHE = merged
            return merged

        if not _CONFIG_MISSING_REPORTED:
            _CONFIG_MISSING_REPORTED = True
        _CONFIG_CACHE = merged
        return merged


# --------------------------------------------------------------------------- #
# Number to words
# --------------------------------------------------------------------------- #

_ONES = (
    "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
    "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
    "Seventeen", "Eighteen", "Nineteen",
)
_TENS = (
    "", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety",
)
_SCALES = (
    (1_000_000_000_000, "Trillion"),
    (1_000_000_000, "Billion"),
    (1_000_000, "Million"),
    (1_000, "Thousand"),
)


def _under_thousand(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    hundreds, rest = divmod(n, 100)
    head = f"{_ONES[hundreds]} Hundred"
    return head + (f" {_under_thousand(rest)}" if rest else "")


def number_to_words(n: int) -> str:
    """Spell out a non-negative integer in the American style Kokoro expects.

    Deliberately omits the "and" that British usage inserts inside hundreds
    groups, because the target phrasing is "One Hundred Four Thousand Eight
    Hundred Sixty-Six", not "One Hundred and Four Thousand".
    """
    if n < 0:
        return "minus " + number_to_words(-n)
    if n == 0:
        return "Zero"
    if n < 1_000:
        return _under_thousand(n)

    parts: list[str] = []
    for value, name in _SCALES:
        if n >= value:
            count, n = divmod(n, value)
            parts.append(f"{number_to_words(count)} {name}")
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Quoted-text protection
# --------------------------------------------------------------------------- #

_QUOTE_RE = re.compile(r'(?<!\\)"')


def _split_protected(text: str) -> list[tuple[bool, str]]:
    """Split into alternating (is_protected, segment) pairs covering all of text.

    A line is protected wholesale if it starts with ``>``. Otherwise double
    quotes toggle protected state, so a quotation spanning several lines keeps
    its interior intact. A blank line ends an open quotation, since a paragraph
    break almost always terminates one in these documents.

    Every character of the input ends up in exactly one segment, newlines
    included, so the caller can rebuild the text with ``"".join(...)``. The
    line breaks must not be treated as segment boundaries: within a single
    line a quote opens and closes several times, and an earlier line-based
    version lost track of which boundaries were real newlines, which injected
    stray line breaks around quotations.
    """
    segments: list[tuple[bool, str]] = []
    in_quote = False

    def push(protected: bool, chunk: str) -> None:
        if not chunk:
            return
        if segments and segments[-1][0] == protected:
            segments[-1] = (protected, segments[-1][1] + chunk)
        else:
            segments.append((protected, chunk))

    lines = text.split("\n")
    for index, line in enumerate(lines):
        # Keep the newline with its line so protected spans stay contiguous.
        chunk = line if index == len(lines) - 1 else line + "\n"

        if not line.strip():
            in_quote = False
            push(False, chunk)
            continue

        if line.lstrip().startswith(">"):
            push(True, chunk)
            continue

        if in_quote:
            closing = _QUOTE_RE.search(chunk)
            if closing is None:
                push(True, chunk)
                continue
            push(True, chunk[: closing.start() + 1])
            in_quote = False
            push(False, chunk[closing.start() + 1 :])
            continue

        cursor = 0
        while True:
            opening = _QUOTE_RE.search(chunk, cursor)
            if opening is None:
                break
            if opening.start() > cursor:
                push(False, chunk[cursor : opening.start()])
            closing = _QUOTE_RE.search(chunk, opening.end())
            if closing is None:
                in_quote = True
                push(True, chunk[opening.start() :])
                cursor = len(chunk)
                break
            push(True, chunk[opening.start() : closing.start() + 1])
            cursor = closing.end()
        push(False, chunk[cursor:])

    return segments


def _map_mutable(segments: list[tuple[bool, str]], fn) -> list[tuple[bool, str]]:
    return [(prot, fn(txt) if not prot else txt) for prot, txt in segments]


# Sentinels for the two masking passes. They must differ: the span placeholders
# below live for the whole pipeline, while _step_e_spacing masks abbreviations
# on top of an already masked string.
_SPAN_OPEN = "\x00"
_SPAN_CLOSE = "\x00"
_ABBREV_OPEN = "\x01"
_ABBREV_CLOSE = "\x01"


def _mask(text: str) -> tuple[str, list[str]]:
    """Swap protected segments for placeholders so a whole-document pass can run.

    Applying the steps to each mutable segment separately is not equivalent. The
    trailing-whitespace rule is line based, so the space in 'the court said
    "..."' is at the end of a segment and would be stripped, gluing the
    quotation to the words before it. Masking keeps the text in one piece and
    lets each rule see real line boundaries.
    """
    spans: list[str] = []
    out: list[str] = []
    for protected, chunk in _split_protected(text):
        if protected:
            spans.append(chunk)
            out.append(f"{_SPAN_OPEN}{len(spans) - 1}{_SPAN_CLOSE}")
        else:
            out.append(chunk)
    return "".join(out), spans


def _unmask(text: str, spans: list[str]) -> str:
    """Put the protected segments back."""
    if not spans:
        return text
    for _ in range(len(spans) + 1):
        if _SPAN_OPEN not in text:
            break
        text = re.sub(
            rf"{_SPAN_OPEN}(\d+){_SPAN_CLOSE}",
            lambda m: spans[int(m.group(1))],
            text,
        )
    return text


# --------------------------------------------------------------------------- #
# Step A - footnote artifacts
# --------------------------------------------------------------------------- #

# Digits fused to the end of a word or a closing bracket are footnote calls,
# e.g. "Certiorari1". A following letter blocks the match so alphanumeric tokens
# like "R2D2" survive. The extra guard keeps an amount intact: in "P100,000.00"
# the "100" is preceded by a letter but is plainly part of a number, so a
# trailing ",<digit>" or ".<digit>" cancels the match.
_FOOTNOTE_DIGITS = re.compile(
    r"(?<=[A-Za-z)\]\"'])[0-9\u00b9\u00b2\u00b3]+(?![A-Za-z0-9])(?![,.][0-9])"
)
# Footnote digits after a sentence period, e.g. "respondents.2". The period is
# kept, the digits dropped. Requiring a letter before the period is what keeps
# decimal and thousands separators safe: in "1,234.56" the "." follows a digit.
_FOOTNOTE_DIGITS_AFTER_PERIOD = re.compile(
    r"(?<=[A-Za-z])\.[0-9\u00b9\u00b2\u00b3]+(?![0-9A-Za-z])"
)
# Whole numbers in square brackets: [1], [2], [33]
_BRACKETED_REF = re.compile(r"\[\s*[0-9\u00b9\u00b2\u00b3]{1,4}\s*\]")
# Bracketed markers that stand alone. The negative lookahead is what keeps
# "[j]ust" and "[s]ir" alive for step B to repair.
_STRAY_BRACKET = re.compile(r"\[\s*(?:s|j|t)\s*\](?![A-Za-z])")
_SIC = re.compile(r"\[\s*sic\s*\]", re.IGNORECASE)
_AF = re.compile(r"\[\s*a\.?f\.?\s*\]", re.IGNORECASE)
# OCR garbage seen in the scanned opinions, e.g. "(awit)".
_OCR_MARKER = re.compile(r"\(\s*aw[Þþ]hi\s*\(?\s*\)?")


def _step_a(text: str, config: dict[str, Any]) -> str:
    text = _FOOTNOTE_DIGITS.sub("", text)
    text = _FOOTNOTE_DIGITS_AFTER_PERIOD.sub(".", text)
    text = _BRACKETED_REF.sub("", text)
    text = _OCR_MARKER.sub("", text)
    text = _SIC.sub("", text)
    text = _AF.sub("", text)
    text = _STRAY_BRACKET.sub("", text)

    for marker in config.get("editorial_markers", []):
        # Matched in both bracket styles; the trailing junk of a mangled
        # "(Emphasis supplied, citations omitted)." is absorbed too.
        pattern = re.compile(
            r"[\[(]\s*" + re.escape(marker) + r"(?:\s*,[^)\]]*)?\s*[\])]\s*\.?",
            re.IGNORECASE,
        )
        text = pattern.sub("", text)
    return text


# --------------------------------------------------------------------------- #
# Step B - OCR and formatting artifacts
# --------------------------------------------------------------------------- #

# "[S]ir" -> "Sir", "[j]ust" -> "just", "[i]nternet" -> "internet": the drop-cap
# reconstruction the scanner left behind. One rule covers every case and
# preserves the original capitalisation.
_BRACKETED_INITIAL = re.compile(r"\[([A-Za-z])\]([A-Za-z])")
# Brackets around a single letter with no word attached.
_LONE_LETTER_BRACKET = re.compile(r"\[([A-Za-z])\]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def _step_b(text: str, config: dict[str, Any]) -> str:
    for wrong, right in _config_pairs(config.get("ocr_map", {})):
        if not isinstance(right, str) or not right:
            continue
        text = re.sub(rf"(?<![A-Za-z]){re.escape(wrong)}(?![A-Za-z])",
                      right.replace("\\", "\\\\"), text, flags=re.IGNORECASE)

    # "Comi" is OCR for "Court". The word boundary keeps "Commission" and
    # "Committee" intact.
    text = re.sub(r"(?<![A-Za-z])Comi(?![A-Za-z])", "Court", text)

    # NOTE: "legal feet to stand on" is intentionally left alone. The scanner
    # produces "foots" there, and it is a fixed idiom that reads correctly
    # either way, so over-correcting it introduces more damage than it removes.
    text = _BRACKETED_INITIAL.sub(r"\1\2", text)
    text = _LONE_LETTER_BRACKET.sub(r"\1", text)

    text = _MULTISPACE.sub(" ", text)
    text = _TRAILING_WS.sub("", text)
    return text


# --------------------------------------------------------------------------- #
# Step C - abbreviations
# --------------------------------------------------------------------------- #

def _abbrev_branch(key: str) -> str:
    """Return one alternation branch matching ``key`` with safe boundaries.

    A leading ``\\b`` is wrong for keys like "G.R." because the last character
    is punctuation and ``\\b`` would then require the following character to be
    a word character. Instead: never match immediately after a word character
    or a period, and never run on into a word character.
    """
    body = re.escape(key)

    # "No." / "Nos." only make sense as a citation marker when a number
    # follows, which is also what protects "No person shall...".
    if key in {"No.", "Nos."}:
        return rf"(?<![A-Za-z0-9.]){body}(?=\s+[0-9])"

    if key.endswith("."):
        return rf"(?<![A-Za-z0-9.]){body}(?![A-Za-z0-9])"

    # A bare "CA" must not swallow the "CA-" of "CA-G.R.", and "G.R." must not
    # match inside a longer dotted token.
    return rf"(?<![A-Za-z0-9.\-]){body}(?![A-Za-z0-9\-])"


def _build_abbrev_re(abbrevs: dict[str, Any]):
    """Compile every abbreviation into one alternation.

    This must be a single pattern rather than a loop of substitutions. Applying
    "NLRC LAC No." and then "NLRC" separately rewrites the acronym *inside the
    replacement the first one just produced*, giving "National Labor Relations
    Commission LAC Number". A combined alternation consumes the match and never
    rescans its own output, so the longest key wins and the result is stable.
    """
    usable = [(k, v) for k, v in _config_pairs(abbrevs) if isinstance(v, str) and v]
    usable.sort(key=lambda kv: (-len(kv[0]), kv[0]))
    if not usable:
        return None, {}

    branches: list[str] = []
    lookup: dict[str, str] = {}
    for index, (key, value) in enumerate(usable):
        name = f"a{index}"
        lookup[name] = value.replace("\\", r"\\")
        branches.append(f"(?P<{name}>{_abbrev_branch(key)})")
    return re.compile("|".join(branches)), lookup


def _step_c(text: str, config: dict[str, Any]) -> str:
    pattern, lookup = _build_abbrev_re(config.get("abbreviations", {}))
    if pattern is None:
        return text
    return pattern.sub(lambda m: lookup[m.lastgroup], text)


# --------------------------------------------------------------------------- #
# Step D - currency and percentages
# --------------------------------------------------------------------------- #

_CENT = re.compile(r"(?<![0-9])([0-9]{1,3}(?:,[0-9]{3})*)\.([0-9]{2})(?![0-9])")
_AMOUNT = re.compile(r"(?<![A-Za-z0-9])([A-Za-z$\u20b1]{1,4})\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)(?![\d,])")
_PERCENT = re.compile(r"(?<![0-9A-Za-z.])([0-9]+(?:\.[0-9]+)?)\s*%")


def _split_amount(raw: str) -> tuple[int, int]:
    """Split a grouped amount such as "1,234.56" into (1234, 56).

    The fraction has to be removed *before* the whole part is parsed, otherwise
    int() sees "1234.56".
    """
    whole_text, _, frac = raw.replace(",", "").partition(".")
    whole = int(whole_text or "0")
    cents = int((frac + "00")[:2]) if frac else 0
    return whole, cents


def _spell_percent(value: str) -> str:
    """Percentages are spoken in lower case - "ten percent", not "Ten Percent"."""
    if "." in value:
        whole, frac = value.split(".", 1)
        digits = " ".join(_ONES[int(d)] for d in frac if d.isdigit())
        return f"{number_to_words(int(whole))} point {digits} percent".lower()
    return f"{number_to_words(int(value))} percent".lower()


def _step_d(text: str, config: dict[str, Any]) -> str:
    currency = config.get("currency_map", {})
    aliases = config.get("currency_aliases", {})
    subunits = config.get("cent_subunit", {})

    def money(match: re.Match[str]) -> str:
        symbol, raw = match.group(1), match.group(2)
        code = aliases.get(symbol) or aliases.get(symbol.upper())
        if not code or code not in currency:
            return match.group(0)

        whole, cents = _split_amount(raw)
        name = currency[code]
        out = f"{number_to_words(whole)} {name}"
        if cents:
            unit = subunits.get(code, "Cents")
            out += f" and {number_to_words(cents)} {unit}"
        return out

    text = _AMOUNT.sub(money, text)
    text = _PERCENT.sub(lambda m: _spell_percent(m.group(1)), text)
    return text


# --------------------------------------------------------------------------- #
# Step E - whitespace
# --------------------------------------------------------------------------- #

_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
_LEADING = re.compile(r"^[ \t]+", re.MULTILINE)
# Only collapse when punctuation is already followed by whitespace, so we
# never split a token or a decimal point.
_SPACE_AFTER_COMMA = re.compile(r",(?!\s)(?=[^\s\d])")
_SPACE_AFTER_COLON = re.compile(r":(?!\s)(?=\S)")
_SPACE_AFTER_SEMICOLON = re.compile(r";(?!\s)(?=\S)")
_SPACE_AFTER_PERIOD = re.compile(r"\.(?!\s)(?=[A-Z])")
# A period that terminates an abbreviation must not gain a space, otherwise
# "G.R. Number" becomes "G. R. Number". Only a dotted run of short all-caps
# tokens or a known short lowercase abbreviation is masked, so an ordinary
# sentence ending such as "held." is still spaced correctly. The lookahead
# deliberately stops before a capital so that a run-down "G.R.The" still gets
# the space it is missing.
_ABBREV_TOKENS = (
    r"[A-Z][A-Z0-9]{0,3}\.|"
    r"v\.|vs\.|nos?\.|arts?\.|secs?\.|paras?\.|pp\.|id\.|et\.|al\.|"
    r"inc\.|ltd\.|co\.|corp\.|jr\.|sr\.|ms\.|mrs?\.|dr\.|fig\.|ch\."
)
_ABBREV_PERIOD = re.compile(rf"(?:{_ABBREV_TOKENS}){{1,4}}(?=\s|$|[),;:])")


def _step_e_spacing(text: str, config: dict[str, Any]) -> str:
    text = _SPACE_AFTER_COMMA.sub(", ", text)
    text = _SPACE_AFTER_COLON.sub(": ", text)
    text = _SPACE_AFTER_SEMICOLON.sub("; ", text)

    # Insert a space after a sentence period, but never after an abbreviation
    # period. Done by temporarily masking the ones we must not touch.
    protected: list[str] = []

    def mask(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"{_ABBREV_OPEN}{len(protected) - 1}{_ABBREV_CLOSE}"

    text = _ABBREV_PERIOD.sub(mask, text)
    text = _SPACE_AFTER_PERIOD.sub(". ", text)

    # Restore, repeatedly, since one abbreviation can hold several markers.
    for _ in range(len(protected) + 1):
        if _ABBREV_OPEN not in text:
            break
        text = re.sub(
            rf"{_ABBREV_OPEN}(\d+){_ABBREV_CLOSE}",
            lambda m: protected[int(m.group(1))],
            text,
        )
    return text


def _step_e_structure(text: str) -> str:
    """Normalise whitespace for the whole (masked) document.

    Trailing whitespace is removed at real line ends and runs of blank lines
    collapse to a single blank line. The document edges are left alone here
    because they may hold masked quotations; ``_finish`` trims them.
    """
    text = _TRAILING.sub("", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text


def _finish(text: str) -> str:
    """Trim the document edges and give a non-empty result one trailing newline."""
    text = _TRAILING.sub("", text).strip()
    return text + "\n" if text else ""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def preprocess_legal_text(text: str) -> str:
    """Clean legal document text for TTS input.

    Removes footnote artifacts, repairs OCR damage, expands legal
    abbreviations, spells out currency amounts and percentages, and normalises
    whitespace. Quoted passages are left byte for byte intact.

    Args:
        text: Raw text as pasted from a decision or other legal document.

    Returns:
        Cleaned text ready to be sent to the Kokoro TTS engine.
    """
    if not text or not text.strip():
        return ""

    config = load_config()

    # Quoted passages are masked out for the whole pipeline and restored at the
    # end, so every rule sees one continuous document while still being unable
    # to touch a quotation.
    work, spans = _mask(text)
    work = _step_a(work, config)
    work = _step_b(work, config)
    work = _step_c(work, config)
    work = _step_d(work, config)
    work = _step_e_structure(work)
    work = _step_e_spacing(work, config)

    return _finish(_unmask(work, spans))
