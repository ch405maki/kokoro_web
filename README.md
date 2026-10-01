# Kokoro TTS — local setup

Text-to-speech running entirely on this machine using [Kokoro-82M](https://github.com/hexgrad/kokoro),
an 82-million-parameter open-weight model that rivals much larger systems while staying fast enough
for CPU-only inference. No API keys, no network calls at inference time, no per-character cost.

Three ways in, all sharing one engine:

| Interface | Entry point | Good for |
|---|---|---|
| Browser UI | `run_server.bat` then open `http://127.0.0.1:8000` | Trying voices, quick jobs |
| REST API | `POST http://127.0.0.1:8000/speak` | Other apps, scripts, automation |
| CLI | `speak.bat "some text"` | Batch jobs, pipelines, one-offs |

---

## 1. Status on this machine

| | |
|---|---|
| Device | `cpu` (Intel Xeon E5-2603 v4 @ 1.70 GHz, no CUDA GPU detected) |
| Torch | 2.14.0+cpu |
| Transformers | 4.57.6 (pinned `<5`; see [deployment gotchas](#two-things-that-break-a-systemd-deployment-silently)) |
| Model | `hexgrad/Kokoro-82M`, cached locally |
| Sample rate | 24000 Hz, mono |
| Voices | 54 published, **41 usable** with the packs installed |
| Languages | `a` `b` (English) `e` `f` `h` `i` `p` (espeak) |
| Throughput | ~1.3x realtime, warm model load ~13 s, first-ever run ~48 s |

Measured on this machine with `.venv/bin/python -m tests.bench`:

```
device = cpu
model load (warm from disk) = 13.15s

case              chars    audio      gen     xRT  chunks
---------------------------------------------------------
one sentence         44    3.25s    1.98s    1.6x       1
one paragraph       219   14.57s   10.82s    1.3x       1
three paragraphs    307   19.03s   14.41s    1.3x       3
```

`xRT` is audio seconds produced per second of compute, so `1.3x` means a one-minute
narration takes about 45 seconds. A discrete NVIDIA GPU raises this substantially;
see [Using a GPU](#using-a-gpu).

**Throughput is very much per-CPU.** The numbers above come from a 2016-era Xeon
E5-2603 v4 at 1.7 GHz sharing 6 cores with nginx and several containers. The same
code on a Ryzen 7 5700G benchmarks around `2.7x`, roughly twice as fast, because
Kokoro is a small model and this is almost entirely single-thread work. Judge
throughput on your own box with `tests.bench` before assuming a duration. Note
also that a *first* CLI or one-shot process looks far slower than these figures
(0.2x or worse) simply because it pays the ~13 s model load inside the measured
window; the served API loads once at startup, so requests are steady-state.

---

## 2. How it works

The whole path from text to audio is four stages. Understanding stage 2 is the
whole reason Kokoro sounds intelligible, and it is where the surprises live.

### Stage 1 — Text arrives

Nothing happens yet. Your text is passed to the pipeline verbatim.

### Stage 2 — Graphemes to phonemes (G2P)

Kokoro does not read letters. It reads **IPA phonemes** — `ˈmɛtədˌAtə` rather than
`metadata`. The converter is [`misaki`](https://github.com/hexgrad/misaki), and how
it runs depends entirely on the language. This is the single most important thing
to understand about the system, because it determines what you can and cannot ask
for:

| `lang_code` | Language | G2P backend | Extra install |
|---|---|---|---|
| `a` | American English | `misaki.en.G2P` + espeak-ng fallback | installed |
| `b` | British English | `misaki.en.G2P` + espeak-ng fallback | installed |
| `e` | Spanish | `misaki.espeak.EspeakG2P` | none, uses espeak-ng |
| `f` | French | `misaki.espeak.EspeakG2P` | none, uses espeak-ng |
| `h` | Hindi | `misaki.espeak.EspeakG2P` | none, uses espeak-ng |
| `i` | Italian | `misaki.espeak.EspeakG2P` | none, uses espeak-ng |
| `p` | Brazilian Portuguese | `misaki.espeak.EspeakG2P` | none, uses espeak-ng |
| `j` | Japanese | `misaki.ja.JAG2P` | `uv pip install "misaki[ja]"` |
| `z` | Mandarin Chinese | `misaki.zh.ZHG2P` | `uv pip install "misaki[zh]"` |

Two consequences worth internalising:

- **English uses a curated lexicon, not espeak.** `misaki.en.G2P` looks words up in
  a hand-built dictionary and only falls back to espeak-ng for words it has never
  seen (product names, jargon, invented words). That is why English output sounds
  natural while the espeak languages sound merely correct. Asking an English voice
  to read a nonsense word gives you espeak's pronunciation of it.
- **The espeak languages need no pip package, only the `espeak-ng` binary**, which
  ships inside this venv via `espeakng_loader`. Importing `misaki.espeak` points
  `phonemizer` at the bundled DLL and data directory at import time, so there is
  nothing to configure. This is why the Windows install needs no MSI installer.

Rather than let you discover a missing pack by hitting an error, the service probes
what is importable at startup and marks each voice `available: true/false` in
`/voices`. Requesting an unavailable voice returns HTTP 422 with the exact
`uv pip install` command that fixes it.

You can inspect this layer directly:

```powershell
.venv\Scripts\python.exe -m app.cli "metadata" --voice bf_emma --show-phonemes
```

```
[0] (2.30s)
    graphemes: metadata
    phonemes : mˈɛtədˌAtə
```

### Stage 3 — Phonemes to audio

The **StyleTTS 2** architecture produces a raw waveform. Three inputs matter:

- **Voice** — a `.pt` embedding that sets timbre. The 54 files live in the model
  repo under `voices/<name>.pt` and are fetched on first use (a few hundred KB each),
  then cached. `af_heart` is the recommended default.
- **Speed** — stretches or compresses the timing. 0.5 to 2.0. Pushing past ~1.5
  starts sounding rushed.
- **`lang_code` must match the voice prefix.** `af_heart` needs `lang_code='a'`.
  Mismatching them produces a warning and a voice that fights the phonemes. This
  wrapper derives `lang_code` from the voice name, so you cannot get it wrong.

Model weights download once from Hugging Face (~330 MB) into
`%USERPROFILE%\.cache\huggingface`, then never again. After that the server starts
offline.

### Stage 4 — Chunks to one file

Long text is split before synthesis, not after. Kokoro chunks on sentence or
paragraph boundaries, generates each chunk independently, and this wrapper
concatenates them with configurable silence (`gap`, default 120 ms).

Two chunking controls:

- **Automatic** — the API and CLI chunk on `\n+` when `gap > 0`, so blank lines in
  your input become natural pauses.
- **Explicit** — set `split_pattern` yourself. Upstream also supports inline
  control marks: `Hello, /world.` inserts a short pause after the comma, and a
  paragraph can be given explicit phonemes with `[ˈnuːnˈθəŋk]` markup.

Kokoro has no maximum-length guard. Without chunking it will silently truncate very
long input, which is the main reason the wrapper chunks for you.

Audio is returned as 24 kHz mono. WAV is written directly in memory with
`soundfile`; MP3 shells out to the `ffmpeg` already on your PATH and returns a
clear error if it is ever missing. MP3 defaults to `96k`, not the `192k` you
might expect: this is one narrow-band voice, and past ~96k the extra bits are
invisible while the file gets twice as big. Override with `KOKORO_MP3_BITRATE`
or per request with `bitrate`.

### Architecture

```
  browser ─┐
  curl    ─┼─> FastAPI (app/server.py) ─┐
  Python  ─┘                            ├─> KokoroEngine (app/engine.py)
  terminal ─> app/cli.py ───────────────┘        │
                                                   ├─ device: cuda > mps > cpu
                                                   ├─ one KPipeline per lang_code, cached
                                                   ├─ synthesise(): text -> [Synthesized]
                                                   └─ encode(): chunks -> wav / mp3 bytes
```

`app/engine.py` is the only place that talks to kokoro. The CLI, the API and the
web UI all call it, so they cannot drift apart. `KokoroEngine` caches one
`KPipeline` per language and serialises synthesis behind a lock, because the
underlying generator is not re-entrant and concurrent requests would corrupt audio.
**The service handles one request at a time.** That is fine up to a few concurrent
users; see [Scaling up](#scaling-up).

---

## 3. Accessing it

### 3a. Browser UI

```powershell
run_server.bat
```

Then open **http://127.0.0.1:8000**.

Type text, click a voice, adjust speed, press **Generate speech**. You get a
waveform with playback and a full-width download button. The
light/dark theme follows your OS setting.

Text you type or paste is auto-formatted to title case: each word is capitalised
with the rest lowercased, so an all-caps `SAMPLE` becomes `Sample`, while short
joining words (`the`, `and`, `on`, ...) stay lowercase unless they open the text
or a sentence.

### Auto-Format for legal text

Pasted text is sent through `POST /preprocess` and the cleaned result is written
back into the same **Text to speak** box, so there is only ever one copy of the
text. It also runs again before **Generate speech** if you have edited the text
since, which means the audio is always built from cleaned text. **Auto-Format**
next to the character count re-runs it on demand.

It is a text pass, not a model call, so it is effectively instant. It does nine
things, in order:

| Step | What it does | Example |
|---|---|---|
| 1 | Drops stray OCR markers | `(awÞhi(`, `[sic]`, `[a.f.]`, `[s]`, `[R]` |
| 2 | Repairs OCR damage and collapses runs of spaces | `prope1iies` → `properties`, `1s` → `is`, `[j]ust` → `just` |
| 3 | Removes editorial markers | `(Emphasis supplied)`, `(Citations omitted)` |
| 4 | Drops a redundant parenthetical (a configured pair, a repeated token, or the initials of what precedes it) | `Court of Appeals (CA)` → `Court of Appeals`, `Juadines (Juadines)` → `Juadines` |
| 5 | Expands remaining standalone abbreviations | `CA` → `Court of Appeals`, `G.R. No.` → `G.R. Number` |
| 6 | Expands corporate and honorific suffixes | `Inc.` → `Incorporated`, `Capt.` → `Captain` |
| 7 | Spells out currency amounts | `PHP 91,575.65` → `Ninety-One Thousand Five Hundred Seventy-Five Pesos and Sixty-Five Centavos` |
| 8 | Spells out percentages | `10%` → `ten percent` |
| 9 | Normalises whitespace and punctuation | blank-line runs collapse, single space after `,` `:` `;` |

For the configured pairs, steps 4 and 5 run as one left-to-right scan, longest
key first, so a stripped full form is never re-expanded and no duplicate is
produced: `Court of Appeals (CA)` becomes `Court of Appeals`, and `POEA Standard
Employment Contract (POEA-SEC)` is not rewritten into a phrase containing the
abbreviation it just lost. Step 4 also covers the party names the config cannot
enumerate: a parenthetical is dropped when it provably repeats the words before
it - a token already present (`Jaime C. Juadines (Juadines)`) or the initials of
a contiguous run of the preceding capitalised words (`The Supreme Court (SC)`,
`Court of Appeals (Ca)`). An unrelated short form (`Agency, Inc. (Arsia)`) and
role tags (`(petitioner)`) are deliberately left alone. After the abbreviations,
corporate and honorific suffixes are expanded in a second pass.

Three deliberate limits: a `No.` is only read as a citation marker when a number
follows, so `No person shall...` is untouched; text inside `>` blockquotes or
double quotes is reproduced byte for byte, because a quoted passage is often
being read aloud as written; and a single bracketed letter (`[j]ust`) is a drop
cap to repair rather than a stray marker to delete.

Mappings live in `config.json` at the repo root, so you can retune them without
touching the code. It is parsed once at startup; if it is missing or malformed
the built-in defaults are used and the server still starts. Add a `"_comment"`
key alongside any entry and it is ignored by the matcher. Each optional pass can
be switched off individually through the `formatting_flags` section - for
example `"spell_out_percentages": false` leaves `10%` as it is - and
`strip_name_parentheticals` (off by default) removes parentheticals that look
like personal names.

#### How fast it is

Measured on this machine against a synthetic 130,132-character / 1,609-line
decision, best of seven runs after warm-up:

| | before | after | |
|---|---|---|---|
| whole document | 119.7 ms | 38.2 ms | **3.1× faster** |
| throughput | 1.09 MB/s | 3.40 MB/s | |
| `POST /preprocess` round trip | — | 45 ms | incl. HTTP |
| a 13 kB paste | — | 7 ms | |

That optimization was byte-for-byte behaviour-preserving. It was verified by a
differential harness that ran the committed version and the optimized one side
by side over 4,195 cases — every configured key in several
contexts, shuffled and prefix-colliding keys, all OCR fixes and editorial
markers, the 44 specification examples, and 3,000 randomized adversarial
strings built from the real config keys. Zero mismatches, and that check is
worth re-running after any change to the matcher.

The speed comes from four things, all of which are behaviour-preserving:

- **One scan instead of one pass per key.** Steps A, B and C are all "swap this
  literal for that value", so they share a matcher that groups keys by first
  character. A cheap character-class trigger decides which group to try at each
  position, and each group is tried only there. Abbreviation expansion alone went
  from ~84 ms to ~3 ms. Keys are still tried longest-first and a replacement is
  never re-scanned, so `NLRC LAC No.` cannot have its own output rewritten by
  the bare `NLRC` rule.
- **No masking when there is nothing to protect.** Most input has no quotation
  in it, so the split/mask/unmask round trip is skipped after a single
  containment test.
- **Compiled patterns are cached, not rebuilt per request.** The editorial
  marker and placeholder-restoration patterns used to be re-compiled from an
  f-string on every call. They are cached against the exact config object they
  were built from, so editing `config.json` and reloading invalidates them
  correctly.
- **A lock-free fast path for the cached config.** Every request read the
  config, and taking a mutex just to read a cache hit serialised the app behind
  one lock.

Two things were measured and deliberately *not* done: an `lru_cache` on
`number_to_words` (6,398 cache hits bought only 1.02×, because the regex
scanning, not the spelling, dominates) and a combined-alternation scan for the
OCR fixes (slower than the trigger dispatch, 13.3 ms vs 8.6 ms). Both are noted
here so nobody re-derives them.

This is pure standard library and platform independent, so the same numbers
apply on Windows and CentOS; there is no C extension, no compiler step and no
platform-specific branch in the preprocessor.

The character count next to the **Text to speak** label tracks every edit, so a
paste reports its size the moment it lands. The page is branded **LawPhil TTS
Studio** and serves its logo from `/static` rather than hotlinking it, so it
still renders on a client with no route to the internet.

The UI only ever *downloads* audio. It does not write to `output/`; the
`save=true` path below is still available to API callers, it is just not wired
to a button.

The page renders from a single `POST /speak/meta` call, so the audio and the
server's own synthesis timing come from the *same* pass. Asking `/speak` as well
would double the work, since synthesis is the expensive part. The grapheme→IPA
breakdown that endpoint returns is API-only: to inspect it, call `/speak/meta`
directly or run the CLI with `--show-phonemes`. A **Pause between chunks** slider
controls the inter-chunk silence.

`run_server.bat 8000 0.0.0.0` binds a different port or interface — see
[Exposing it on your network](#exposing-it-on-your-network).

### 3b. REST API

Interactive reference with a working "try it" form: **http://127.0.0.1:8000/docs**

#### `POST /preprocess` — clean legal text without synthesising

| Field | Type | Default | Notes |
|---|---|---|---|
| `text` | string | `""` | may be empty |

Returns `{"text": "...", "changed": true|false}`, where `text` is the cleaned
result and `changed` is `false` when the input needed no cleaning. `changed`
ignores surrounding whitespace, so a tidy document is not reported as edited
just because the response ends with a newline. See
[Auto-Format for legal text](#auto-format-for-legal-text) for what it changes.

#### `POST /speak` — the main endpoint

Returns raw audio bytes.

| Field | Type | Default | Notes |
|---|---|---|---|
| `text` | string | *required* | min length 1 |
| `voice` | string | `af_heart` | 54 published, 41 usable |
| `speed` | float | `1.0` | 0.5–2.0 |
| `format` | string | `wav` | `wav` or `mp3` |
| `bitrate` | string | `KOKORO_MP3_BITRATE` (`96k`) | MP3 only. `32k`–`320k`, or a bare number meaning `k` |
| `gap` | float | `0.12` | seconds of silence between chunks |
| `split_pattern` | string | `null` | custom chunker regex |

#### PowerShell

`curl.exe -d '{...}'` mangles JSON on Windows — curl's argument parser strips the
inner double quotes. Pipe the payload instead:

```powershell
'{ "text": "Kokoro runs on this machine.", "voice": "af_heart" }' |
  curl.exe -s -X POST http://127.0.0.1:8000/speak `
    -H "Content-Type: application/json" --data-binary "@-" -o out.wav
```

Or use the built-in cmdlet, which needs no quoting workaround:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/speak -Method Post `
  -ContentType "application/json" `
  -Body (@{ text = "Kokoro runs on this machine."; voice = "af_heart" } | ConvertTo-Json) `
  -OutFile out.wav
```

#### macOS / Linux

```bash
curl -s -X POST http://127.0.0.1:8000/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"Kokoro runs on this machine.","voice":"af_heart"}' \
  --output out.wav
```

#### From another language

```python
import requests
r = requests.post("http://127.0.0.1:8000/speak",
                  json={"text": "Kokoro runs on this machine.", "voice": "af_heart"},
                  timeout=300)
open("out.wav", "wb").write(r.content)
```

```javascript
// Needs CORS enabled for cross-origin browser calls; see the network section.
const r = await fetch("http://127.0.0.1:8000/speak", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Kokoro runs on this machine.", voice: "af_heart" }),
});
const blob = await r.blob();
new Audio(URL.createObjectURL(blob)).play();
```

Response headers carry timing metadata: `x-chunk-count`, `x-generation-seconds`,
`x-sample-rate`.

#### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, device, load time, voice count |
| `GET /voices` | Full voice catalog with `available` flags and locales |
| `GET /voices?lang=b` | Filter by locale prefix (`a`/`b`/`af`/`bf`) |
| `POST /warmup` | Load the model now so the first real request is fast |
| `POST /speak/meta` | Same audio as base64 **plus** per-chunk phonemes and durations |
| `POST /speak/form` | `multipart/form-data`; `save=true` writes to `output/`. No longer used by the web UI |
| `GET /download/{name}` | Fetch a file previously saved with `save=true` |
| `GET /docs` | Swagger UI |
| `GET /openapi.json` | OpenAPI schema |

Use `/speak/meta` when you want to see *why* something sounded wrong:

```powershell
'{ "text": "metadata", "voice": "bf_emma" }' |
  curl.exe -s -X POST http://127.0.0.1:8000/speak/meta `
    -H "Content-Type: application/json" --data-binary "@-"
```

```json
{
  "chunks": [{ "index": 0, "graphemes": "metadata", "phonemes": "mˈɛtədˌAtə", "duration": 1.42 }],
  "total_duration": 1.42, "elapsed": 0.51, "device": "cpu",
  "audio_base64": "UklGRi...", "content_type": "audio/wav"
}
```

#### Status codes

`200` success · `422` validation error (empty text, speed out of range, unknown
format, or a voice whose language pack is missing — the message says which and how
to fix it) · `500` synthesis or ffmpeg failure.

### 3c. Command line

```powershell
speak.bat "Kokoro is running locally."
speak.bat --text-file script.txt --voice bf_emma --speed 0.9
"piped input" | speak.bat --stdin
```

Or call the module directly, which is what the batch file wraps:

```powershell
.venv\Scripts\python.exe -m app.cli "Hello." --voice bm_george --speed 1.1
.venv\Scripts\python.exe -m app.cli "Long narration." --format mp3 --out narration.mp3
.venv\Scripts\python.exe -m app.cli "Multi paragraph." --no-join   # one file per chunk
.venv\Scripts\python.exe -m app.cli --list-voices                  # 41 usable voices
.venv\Scripts\python.exe -m app.cli --warmup                      # pre-download + load
```

| Flag | Meaning |
|---|---|
| `--voice`, `-v` | Voice id |
| `--speed`, `-s` | 0.5–2.0 |
| `--format` | `wav` or `mp3` |
| `--out`, `-o` | Output path (default: timestamped file in `output/`) |
| `--text-file`, `-t` / `--stdin` | Read text from a file or a pipe |
| `--gap` | Silence between chunks, seconds |
| `--bitrate` | MP3 only, e.g. `96k` (default: `KOKORO_MP3_BITRATE`) |
| `--split-pattern` | Custom chunker regex |
| `--no-join` | One file per chunk instead of one joined file |
| `--show-phonemes` | Print graphemes and IPA per chunk |
| `--play` | Open the result in the default player |
| `--list-voices` | List the 41 usable voices (`--all` shows all 54) |
| `--warmup` | Pre-download and load the model, then exit |

---

## 4. Verifying it works

```powershell
# full API suite: 64 checks, server must already be running
run_server.bat                                # terminal 1
.venv\Scripts\python.exe tests\test_api.py   # terminal 2

# preprocessor suite: 75 tests, no server needed
.venv\Scripts\python.exe -m unittest tests.test_preprocessor -v

# throughput table
.venv\Scripts\python.exe tests\bench.py
```

Run `test_api.py` **as a script path**, not with `-m`. It is a plain script with a
`main()`, not a `unittest.TestCase`, so `-m tests.test_api` imports it, finds no
test cases, and exits reporting "Ran 0 tests" without complaining - an easy way to
mistake a broken server for a passing suite. `test_preprocessor.py` is a real
`unittest` module, so `-m` is correct there.

Set `PYTHONIOENCODING=utf-8` first. The console defaults to cp1252 on Windows and
the suite prints IPA, and will die with `UnicodeEncodeError` partway through
otherwise.

Latest run on this machine: **64/64 API checks and 75/75 preprocessor tests
passed.** The API suite covers every endpoint, speed and voice differentiation,
WAV header parsing, chunk splitting, MP3 encoding, path-traversal rejection, the
unavailable-language-pack path, and `POST /preprocess`. It
also asserts the auth posture (off by default) and that `gap` genuinely inserts
silence: two renders of the same multi-chunk text are compared and the difference
must equal `gap x (chunks - 1)` within 250 ms. That last one exists because `gap`
was originally accepted by the API and then dropped on the floor, so every
paragraph ran together with no pause.

Start the server *after* pulling changes. A stale process still listening on
8000 will happily answer the old endpoint set, and new checks fail with a 404
that looks like a routing bug.

---

## 5. Configuration

All optional, via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `KOKORO_DEVICE` | auto-detected | Force `cpu`, `cuda`, or `mps` |
| `KOKORO_THREADS` | CPU count | Torch thread limit |
| `KOKORO_MODEL_REPO` | `hexgrad/Kokoro-82M` | Alternative model repo |
| `KOKORO_MP3_BITRATE` | `96k` | Default MP3 bitrate. 24 kHz mono speech tops out well below 192k |
| `KOKORO_OUTPUT_DIR` | `./output` | Where saved files land |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | CLI and API default |
| `KOKORO_DEFAULT_LANG` | `a` | Startup warmup language |

```powershell
$env:KOKORO_DEVICE = "cpu"
$env:KOKORO_THREADS = "8"
```

Adding a language is one command:

```powershell
uv pip install --python .venv\Scripts\python.exe "misaki[ja]"
# restart the server -- /voices will then mark jf_* and jm_* voices available
```

---

## 6. Troubleshooting

**Web UI loads but says `engine offline`.** The server is not running or is on a
different port. Check `run_server.bat` output.

**First request takes 30+ seconds, later ones ~1s.** Expected. The first-ever run
downloads 330 MB of weights and the spaCy English model. `POST /warmup` or
`python -m app.cli --warmup` moves that cost to setup time.

**A voice returns 422 about a language pack.** The message includes the fix.
Japanese and Mandarin need `misaki[ja]` / `misaki[zh]`.

**`ffmpeg not found on PATH`** when requesting MP3. Already installed here; WAV
always works. `winget install ffmpeg` if you removed it.

**Windows console `UnicodeEncodeError` on phonemes.** The CLI already forces UTF-8
output. If you call `engine.py` from your own script, set
`PYTHONIOENCODING=utf-8` first, or use `/speak/meta` and let JSON carry the IPA.

**Audio is silent or truncated on very long input.** Chunking is on by default. If
you passed a custom `split_pattern`, check it actually matches — a pattern that
never matches leaves Kokoro to truncate. Try the default `\n+`.

**Primes are wrong, or jargon is mispronounced.** That is espeak-ng fallback
behaviour for out-of-dictionary words. Try a different voice, or force a
mispronunciation inline with phoneme markup: `[ˈnuːnˈθəŋk]phone`.

**Text comes out as one run-on sentence.** Add sentence-ending punctuation, or
blank lines to force paragraph chunks.

---

## 7. Going further

### Using a GPU

Nothing is hardcoded to CPU. `detect_device()` prefers CUDA, then Apple MPS, then
CPU, and `KOKORO_DEVICE` overrides it. Install the CUDA build instead of the CPU
one:

```powershell
uv pip install --python .venv\Scripts\python.exe torch ^
  --index-url https://download.pytorch.org/whl/cu124
```

On Apple Silicon: `set PYTORCH_ENABLE_MPS_FALLBACK=1` before running.

### Exposing it on your network

The default `127.0.0.1` bind is local-only, which is the right default. To reach
it from another machine on your LAN:

```powershell
run_server.bat 8000 0.0.0.0
```

Out of the box there is **no authentication and no rate limit**, so anyone who can
reach the port can make it burn CPU. Two opt-in environment variables close that,
and both are off by default so single-user local use is unaffected:

| Variable | Effect when set |
|---|---|
| `KOKORO_API_KEY` | Requires `X-API-Key: <key>` or `Authorization: Bearer <key>` on every route except `GET /health`. Compared with `hmac.compare_digest`. Returns 401 on mismatch. |
| `KOKORO_CORS_ORIGINS` | Comma-separated allowlist, e.g. `https://app.example.com`. Adds CORS headers so a browser app on another origin can call the API. |

```powershell
$env:KOKORO_API_KEY = (openssl rand -hex 32)
$env:KOKORO_CORS_ORIGINS = "https://app.example.com"
run_server.bat 8000 0.0.0.0
```

The web UI has no way to attach a custom header to a same-origin post, so with
`KOKORO_API_KEY` set, `/` itself stops working from a browser. For a browser-facing
deployment, leave the key unset and gate the site with HTTP Basic auth at a reverse
proxy instead - see `DEPLOY.md`, which has a ready nginx config that does exactly
this and also rate limits per client IP.

Rate limiting is not built in. Synthesis is CPU-bound and single-worker, so one
caller can queue the server for everyone else; put nginx or a rate-limit-aware proxy
in front if the port is not on a trusted network.

#### Two things that break a systemd deployment silently

Both of these fail *after* the server starts and `/health` reports `ok`, so they
are easy to miss.

**`MemoryDenyWriteExecute=yes` breaks every synthesis request.** torch bundles
oneDNN, whose CPU backend JIT-compiles kernels into anonymous executable memory.
Deny W^X and that mmap fails, so `/warmup` succeeds and then every `/speak`
returns `500 {"detail":"could not create a primitive"}`. This is *not* limited to
`torch.compile`/inductor, which is what the hardening comment in
`deploy/kokoro-tts.service` assumed — plain CPU inference trips it. The shipped
unit is corrected to `MemoryDenyWriteExecute=no`; every other sandbox directive
stays on.

**An unpinned `transformers` breaks model loading.** `kokoro` 0.9.4 declares
`transformers` with no upper bound, so a fresh install resolves 5.x, which
imports `torch._dynamo` at module scope. On torch 2.14 that raises
`Artifact of type=precompile already registered in mega-cache artifact factory`
partway through loading the model. `pyproject.toml` now pins `transformers<5`.

On SELinux hosts there is a third, which fails at `systemctl reload nginx`
rather than at runtime: a `listen` on a port outside the policy's
`http_port_t` list gets `bind() ... (13: Permission denied)` even though
`nginx -t` passes. Fix with `semanage port -a -t http_port_t -p tcp <port>`.
Note 8082 already ships as `us_cli_port_t`, so that one needs `-m`, not `-a`.

### Scaling up

Synthesis is serialised behind one lock, so requests queue rather than run
concurrently. That is the right trade for a single user. For real concurrency,
run several processes under a load balancer:

```powershell
uvicorn app.server:app --host 0.0.0.0 --port 8001 --workers 2
```

Each worker loads its own copy of the weights (~330 MB RAM) so size the box
accordingly.

---

## 8. Files

```
app/
  engine.py           model loading, synthesis, voice catalog, wav/mp3 encoding
  server.py           FastAPI app: /speak /preprocess /voices /health /warmup /docs
  cli.py              argparse CLI
  static/index.html   web UI, self-contained, no build step
legal_preprocessor.py  legal-text cleaner behind /preprocess
tests/
  test_api.py         64-check end-to-end suite
  test_preprocessor.py  75-test preprocessor suite (unittest, no server needed)
  bench.py            throughput benchmark
config.json           abbreviation / OCR / currency mappings for the preprocessor
output/               generated audio (gitignored)
deploy/
  install-centos.sh   idempotent CentOS Stream/Rocky/Alma 9 installer
  kokoro-tts.service  hardened systemd unit
  nginx-kokoro.conf   reverse proxy: TLS, Basic auth, rate limit
DEPLOY.md             server deployment guide
setup.bat             rebuild the venv from scratch
run_server.bat        start the API + web UI
speak.bat             one-shot CLI wrapper
pyproject.toml        dependency manifest
```

`.venv/` holds everything: torch, kokoro, misaki, spaCy, FastAPI, and the bundled
`espeak-ng` DLL with its data directory. Deleting it and running `setup.bat`
reproduces the whole environment.

---

## 9. Licences

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — **Apache-2.0**, weights included.
  Commercial use permitted.
- [kokoro](https://pypi.org/project/kokoro/) inference library — Apache-2.0
- [misaki](https://github.com/hexgrad/misaki) G2P — Apache-2.0
- StyleTTS 2 architecture — [@yl4579](https://huggingface.co/yl4579)

Apache-2.0 imposes no restriction on commercial use, distribution, or
modification. There is no copyleft obligation and no network-use clause.

---

## 10. Credits

Kokoro is a Japanese word meaning *heart* or *spirit*, and also a character in the
Terminator franchise, alongside Misaki. Upstream project:
[github.com/hexgrad/kokoro](https://github.com/hexgrad/kokoro), community at
[discord.gg/QuGxSWBfQy](https://discord.gg/QuGxSWBfQy).
