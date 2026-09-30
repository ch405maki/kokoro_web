r"""End-to-end smoke test.  Run with the server already listening.

    .venv\Scripts\python.exe -m tests.test_api
    BASE=http://127.0.0.1:8000 .venv\Scripts\python.exe -m tests.test_api

Exits non-zero on the first failure.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

BASE = os.getenv("BASE", "http://127.0.0.1:8000").rstrip("/")

passed = 0
failed: list[str] = []


class Headers(dict):
    """HTTP header names are case-insensitive; Starlette sends them lowercase."""

    def get(self, key, default=None):
        return super().get(key.lower(), default)

    def __missing__(self, key):
        return self.get(key)  # not reached via .get, but keeps dict semantics sane


def _headers(raw) -> Headers:
    return Headers({k.lower(): v for k, v in raw.items()})


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}{(' -- ' + detail) if detail else ''}")
    else:
        failed.append(label)
        print(f"  FAIL  {label}{(' -- ' + detail) if detail else ''}")


def get(path: str) -> tuple[int, Headers, bytes]:
    req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, _headers(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, _headers(e.headers), e.read()


def post_json(path: str, payload: dict) -> tuple[int, Headers, bytes]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, _headers(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, _headers(e.headers), e.read()


def post_form(path: str, fields: dict) -> tuple[int, Headers, bytes]:
    boundary = "----kokorotest7f3a"
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, _headers(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, _headers(e.headers), e.read()


def wav_duration(data: bytes) -> float:
    """Read duration straight from the RIFF header, no deps.

    RIFF chunk ids are followed by a size field, so a `fmt ` chunk reads:
      +0 audio_format | +2 channels | +4 sample_rate | +8 byte_rate
    relative to the start of the chunk *body* (i.e. id + 8). Byte rate is
    therefore at pos+16. Using the sample-rate field here (pos+12) reports
    16-bit mono audio as exactly twice its true length.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE payload")
    pos = 12
    byte_rate = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            channels, sample_rate = struct.unpack("<H", data[pos + 10:pos + 12])[0], \
                struct.unpack("<I", data[pos + 12:pos + 16])[0]
            bits = struct.unpack("<H", data[pos + 22:pos + 24])[0]
            byte_rate = struct.unpack("<I", data[pos + 16:pos + 20])[0]
            # Trust the header arithmetic over the declared byte rate, which
            # some encoders get wrong.
            derived = sample_rate * channels * bits // 8
            if derived:
                byte_rate = derived
        if cid == b"data":
            if not byte_rate:
                raise ValueError("fmt chunk missing")
            return size / byte_rate
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def main() -> int:
    print(f"\nKokoro API smoke test -> {BASE}\n" + "-" * 52)

    # ----------------------------------------------------- auth posture
    # KOKORO_API_KEY is opt-in. If it is set, this suite cannot supply the
    # header, so fail loudly rather than reporting a wall of 401s.
    print("\n[auth]")
    if os.getenv("KOKORO_API_KEY", "").strip():
        print("  SKIP  KOKORO_API_KEY is set; this suite sends no credentials")
    else:
        status, _, _ = get("/voices")
        check("auth is off by default (no key required)", status == 200, f"got {status}")
        print("        to test auth, start the server with KOKORO_API_KEY set")

    # ---------------------------------------------------------- GET /health
    print("\n[health]")
    status, _, raw = get("/health")
    check("GET /health returns 200", status == 200, f"got {status}")
    health = json.loads(raw)
    check("reports ok", health.get("status") == "ok")
    check("reports a device", health.get("device") in {"cpu", "cuda", "mps"}, str(health.get("device")))
    check("sample rate is 24000", health.get("sample_rate") == 24000)
    check("discovers voices", health.get("voices", 0) > 10, f"{health.get('voices')} voices")
    print(f"        device={health['device']} ready={health.get('ready')} voices={health['voices']}")

    # ---------------------------------------------------------- GET /voices
    print("\n[voices]")
    status, _, raw = get("/voices")
    voices = json.loads(raw)
    check("GET /voices returns 200", status == 200)
    check("default voice is present", voices["default"] in [v["name"] for v in voices["voices"]])
    check("every voice has a locale", all(v.get("locale") for v in voices["voices"]))
    check("every voice is flagged available or not", all("available" in v for v in voices["voices"]))
    check("default voice is available",
          next(v["available"] for v in voices["voices"] if v["name"] == voices["default"]))
    check("English voices are available",
          all(v["available"] for v in voices["voices"] if v["name"].startswith(("a", "b"))),
          f"{sum(1 for v in voices['voices'] if v['available'])} of {len(voices['voices'])} usable")

    _, _, raw = get("/voices?lang=b")
    british = json.loads(raw)
    check("?lang=b filters to British voices", british["count"] > 0 and all(
        v["name"].startswith("b") for v in british["voices"]),
        f"{british['count']} voices")

    _, _, raw = get("/voices?lang=af")
    check("?lang=af filters to American female", json.loads(raw)["count"] > 0)

    _, _, raw = get("/voices?lang=zz")
    check("unknown locale returns empty, not error", json.loads(raw)["count"] == 0)

    # -------------------------------------------------------- POST /warmup
    print("\n[warmup]")
    t0 = time.perf_counter()
    status, _, raw = post_json("/warmup", {})
    warm = json.loads(raw)
    check("POST /warmup returns 200", status == 200)
    check("warmup reports load time", warm.get("load_seconds") is not None,
          f"{warm.get('load_seconds')}s (request took {time.perf_counter() - t0:.1f}s)")

    # -------------------------------------------------------- POST /speak
    print("\n[speak -> wav]")
    payload = {"text": "The quick brown fox jumps over the lazy dog.", "voice": "af_heart"}
    t0 = time.perf_counter()
    status, headers, body = post_json("/speak", payload)
    elapsed = time.perf_counter() - t0
    check("POST /speak returns 200", status == 200, f"got {status}")
    check("content-type is audio/wav", headers.get("Content-Type") == "audio/wav", headers.get("Content-Type", "?"))
    check("sends a download filename", "kokoro-af_heart" in headers.get("Content-Disposition", ""))
    check("payload is a valid WAV", body[:4] == b"RIFF")
    dur = wav_duration(body)
    check("WAV has audio", dur > 1.0, f"{dur:.2f}s of audio in {elapsed:.2f}s ({dur / elapsed:.1f}x realtime)")
    check("X-Chunk-Count header present", headers.get("X-Chunk-Count") == "1")

    # --------------------------------------------------- POST /speak params
    print("\n[speak -> parameters]")
    _, h, fast = post_json("/speak", {"text": "Faster now.", "voice": "af_heart", "speed": 1.5})
    d_fast = wav_duration(fast)
    _, _, slow = post_json("/speak", {"text": "Faster now.", "voice": "af_heart", "speed": 0.7})
    d_slow = wav_duration(slow)
    check("speed=1.5 is shorter than speed=0.7", d_fast < d_slow, f"{d_fast:.2f}s vs {d_slow:.2f}s")

    voice_a = post_json("/speak", {"text": "Same words, different voice.", "voice": "af_bella"})[2]
    voice_b = post_json("/speak", {"text": "Same words, different voice.", "voice": "bm_george"})[2]
    check("different voices produce different audio", voice_a != voice_b)

    _, h, mp3 = post_json("/speak", {"text": "Encoded to mp3.", "voice": "af_heart", "format": "mp3"})
    check("format=mp3 returns audio/mpeg", h.get("Content-Type") == "audio/mpeg")
    check("mp3 is much smaller than wav", len(mp3) < len(body), f"{len(mp3)} vs {len(body)} bytes")

    multi = "First paragraph here.\n\nSecond paragraph right after it.\n\nAnd a third one."
    _, h, joined = post_json("/speak", {"text": multi, "voice": "af_heart", "gap": 0.2})
    check("blank lines split into multiple chunks", h.get("X-Chunk-Count") == "3", f"got {h.get('X-Chunk-Count')}")

    # ------------------------------------------------------ POST /speak/meta
    print("\n[speak/meta]")
    status, _, raw = post_json("/speak/meta", {"text": "Metadata round trip.", "voice": "bf_emma"})
    meta = json.loads(raw)
    check("returns 200", status == 200)
    check("returns phonemes", bool(meta["chunks"][0]["phonemes"]), repr(meta["chunks"][0]["phonemes"]))
    check("phonemes are IPA, not ascii-escaped", any(ord(c) > 127 for c in meta["chunks"][0]["phonemes"]))
    check("returns duration", meta["total_duration"] > 0.5, f"{meta['total_duration']}s")
    check("returns base64 audio", len(meta["audio_base64"]) > 1000)
    check("base64 decodes to valid WAV", base64.b64decode(meta["audio_base64"])[:4] == b"RIFF")
    check("returns graphemes for the UI breakdown", bool(meta["chunks"][0].get("graphemes")))

    # ------------------------------------------------------------------ gap
    # `gap` used to be accepted and then silently dropped, so multi-paragraph
    # audio had no pauses at all. Derive the expected silence from the real
    # chunk count rather than assuming how the text will be split.
    print("\n[gap]")
    three = "Alpha bravo charlie.\n\nDelta echo foxtrot.\n\nGolf hotel india."
    _, _, raw0 = post_json("/speak/meta", {"text": three, "voice": "af_bella", "gap": 0.0})
    meta0 = json.loads(raw0)
    n = len(meta0["chunks"])
    lo = wav_duration(base64.b64decode(meta0["audio_base64"]))

    _, _, raw1 = post_json("/speak/meta", {"text": three, "voice": "af_bella", "gap": 0.4})
    meta1 = json.loads(raw1)
    hi = wav_duration(base64.b64decode(meta1["audio_base64"]))

    check("gap=0 and gap>0 produce the same chunking", len(meta1["chunks"]) == n,
          f"{n} vs {len(meta1['chunks'])} chunks")
    check("text splits into multiple chunks", n >= 2, f"{n} chunks")
    expect = 0.4 * (n - 1)
    grew = hi - lo
    check("gap>0 lengthens the render", grew > 0.1, f"{lo:.2f}s -> {hi:.2f}s (+{grew:.2f}s)")
    check("gap matches the requested silence", abs(grew - expect) < 0.25,
          f"expected ~{expect:.2f}s over {n - 1} boundaries, got {grew:.2f}s")

    _, _, raw_form = post_form("/speak/form", {
        "text": three, "voice": "af_bella", "format": "wav", "gap": "0.4",
    })
    grew_form = wav_duration(raw_form) - lo
    check("/speak/form honours gap", abs(grew_form - expect) < 0.25,
          f"expected ~{expect:.2f}s, got {grew_form:.2f}s")

    _, _, raw_zero = post_form("/speak/form", {
        "text": three, "voice": "af_bella", "format": "wav", "gap": "0",
    })
    check("/speak/form gap=0 matches the ungapped render",
          abs(wav_duration(raw_zero) - lo) < 0.05,
          f"{wav_duration(raw_zero):.2f}s vs {lo:.2f}s")

    # ------------------------------------------------------ POST /speak/form
    print("\n[speak/form]")
    status, headers, body = post_form("/speak/form", {
        "text": "Multipart form submission.", "voice": "af_nicole", "speed": "1.0", "format": "wav",
    })
    check("returns audio", status == 200 and body[:4] == b"RIFF", f"got {status}")

    status, _, raw = post_form("/speak/form", {
        "text": "This one gets written to disk.", "voice": "af_bella", "save": "true",
    })
    saved = json.loads(raw)
    check("save=true writes to output/", status == 200 and "path" in saved, saved.get("path", ""))
    if "path" in saved:
        check("saved file exists and is non-empty", os.path.getsize(saved["path"]) > 1000)
        fname = os.path.basename(saved["path"])
        st, _, blob = get(f"/download/{fname}")
        check("GET /download serves it back", st == 200 and blob[:4] == b"RIFF")

    # ------------------------------------------------------- error handling
    print("\n[error handling]")
    check("empty text -> 422", post_json("/speak", {"text": ""})[0] == 422)
    check("speed out of range -> 422", post_json("/speak", {"text": "hi", "speed": 99})[0] == 422)
    check("bad format -> 422", post_json("/speak", {"text": "hi", "format": "aiff"})[0] == 422)
    check("missing required field -> 422", post_json("/speak", {})[0] == 422)
    check("path traversal blocked", get("/download/..%2f..%2fpyproject.toml")[0] == 404)
    st, _, raw = post_json("/speak", {"text": "こんにちは", "voice": "jf_alpha"})
    check("uninstalled language pack -> 422", st == 422, f"got {st}")
    check("  ...with an actionable message", b"misaki" in raw, raw[:90].decode("utf-8", "replace"))

    # ------------------------------------------------------------- web UI
    print("\n[web ui]")
    status, headers, body = get("/")
    check("GET / returns HTML", status == 200 and headers.get("Content-Type", "").startswith("text/html"))
    check("page contains the app markup", b"Kokoro TTS" in body)
    status, _, _ = get("/docs")
    check("GET /docs serves OpenAPI UI", status == 200)
    status, _, raw = get("/openapi.json")
    check("GET /openapi.json is valid", status == 200 and "paths" in json.loads(raw))

    # -------------------------------------------------------------- summary
    print("\n" + "-" * 52)
    if failed:
        print(f"{passed} passed, {len(failed)} FAILED:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"All {passed} checks passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
