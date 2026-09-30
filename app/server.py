"""FastAPI application exposing Kokoro over HTTP."""

from __future__ import annotations

import hmac
import os
import time
import urllib.parse
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import engine as E

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Opt-in hardening for when this runs as a shared service rather than a
# single-user desktop tool. Both default to off, so local behaviour is unchanged.
API_KEY = os.getenv("KOKORO_API_KEY", "").strip()
CORS_ORIGINS = [o.strip() for o in os.getenv("KOKORO_CORS_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(
    title="Kokoro TTS API",
    version="1.0.0",
    description="Local text-to-speech using Kokoro-82M (82M params, Apache-2.0).",
    docs_url="/docs",
    openapi_url="/openapi.json",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

REQUEST_LOG: list[dict] = []

# Endpoints that stay open so a load balancer or uptime check can reach them.
_OPEN_PATHS = {"/health"}


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Constant-time shared-secret check, no-op unless KOKORO_API_KEY is set.

    Accepts either `X-API-Key: <key>` or `Authorization: Bearer <key>`.
    The web UI cannot send a header, so it relies on a same-origin request.
    """
    if not API_KEY or request.url.path in _OPEN_PATHS:
        return await call_next(request)

    supplied = request.headers.get("X-API-Key", "")
    if not supplied:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if not hmac.compare_digest(supplied, API_KEY):
        return JSONResponse(
            {"detail": "invalid or missing API key; send it as X-API-Key or "
                       "Authorization: Bearer"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)



# ------------------------------------------------------------------ schemas

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to speak.")
    voice: str = Field(E.DEFAULT_VOICE, description="Voice id, e.g. af_heart")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern="^(wav|mp3)$")
    bitrate: str | None = Field(
        None,
        description="MP3 only. e.g. '96k'. Defaults to KOKORO_MP3_BITRATE.",
    )
    gap: float = Field(0.12, ge=0.0, le=2.0, description="Silence between chunks (s)")
    split_pattern: str | None = Field(
        None, description="Regex chunker. Defaults to paragraphs when gap > 0."
    )


class SpeakMeta(BaseModel):
    chunks: list[dict]
    sample_rate: int
    device: str
    total_duration: float
    elapsed: float
    text: str
    voice: str
    speed: float


# ------------------------------------------------------------------- routes

@app.get("/health")
def health() -> dict:
    eng = E.get_engine()
    return {
        "status": "ok",
        "ready": eng.ready,
        "device": eng.device,
        "model": E.MODEL_REPO,
        "sample_rate": E.SAMPLE_RATE,
        "load_seconds": eng.load_seconds,
        "voices": len(E.list_voices()),
    }


@app.get("/voices")
def voices(lang: str | None = Query(None, max_length=4, description="Locale prefix, e.g. a / b / af")) -> dict:
    items = E.list_voices(lang)
    return {
        "count": len(items),
        "available_count": sum(1 for v in items if v["available"]),
        "default": E.DEFAULT_VOICE,
        "sample_rate": E.SAMPLE_RATE,
        "available_languages": sorted(E.available_lang_codes()),
        "voices": items,
    }


@app.post("/warmup")
def warmup() -> dict:
    """Pre-load the model so the first real request is not slow."""
    return E.get_engine().warm_up()


@app.post("/speak", response_class=Response)
def speak_json(req: SpeakRequest) -> Response:
    """Synthesize speech. Returns raw audio bytes (audio/wav or audio/mpeg)."""
    started = time.perf_counter()
    try:
        chunks = E.get_engine().synthesize(
            text=req.text,
            voice=req.voice,
            speed=req.speed,
            split_pattern=req.split_pattern,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = round(time.perf_counter() - started, 3)
    # An unusable bitrate is the caller's mistake, so validate before encoding
    # rather than surfacing it as a 500 from deep inside ffmpeg.
    try:
        E.normalise_bitrate(req.bitrate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        audio, content_type = E.encode(chunks, req.format, gap=req.gap,
                                       bitrate=req.bitrate)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _log(req.text, req.voice, req.speed, req.format, elapsed, len(chunks))
    stem = f"kokoro-{_safe_voice(req.voice)}-{uuid.uuid4().hex[:8]}"
    return Response(
        content=audio,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{stem}.{req.format}"',
            "X-Chunk-Count": str(len(chunks)),
            "X-Generation-Seconds": str(elapsed),
            "X-Sample-Rate": str(E.SAMPLE_RATE),
        },
    )


@app.post("/speak/meta")
def speak_meta(req: SpeakRequest) -> JSONResponse:
    """Same as /speak but returns phoneme metadata + a data URL instead of a file."""
    started = time.perf_counter()
    try:
        chunks = E.get_engine().synthesize(
            text=req.text, voice=req.voice, speed=req.speed, split_pattern=req.split_pattern
        )
        audio, _ = E.encode(chunks, req.format, gap=req.gap, bitrate=req.bitrate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = round(time.perf_counter() - started, 3)
    import base64

    _log(req.text, req.voice, req.speed, req.format, elapsed, len(chunks))
    payload = SpeakMeta(
        chunks=[
            {
                "index": i,
                "graphemes": c.graphemes.strip(),
                "phonemes": c.phonemes.strip(),
                "duration": c.duration,
            }
            for i, c in enumerate(chunks)
        ],
        sample_rate=E.SAMPLE_RATE,
        device=E.get_engine().device,
        total_duration=round(sum(c.duration for c in chunks), 3),
        elapsed=elapsed,
        text=req.text,
        voice=req.voice,
        speed=req.speed,
    )
    return JSONResponse(
        content=jsonable(payload) | {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "content_type": E.CONTENT_TYPES[req.format],
        }
    )


@app.post("/speak/form")
def speak_form(
    text: str = Form(...),
    voice: str = Form(E.DEFAULT_VOICE),
    speed: float = Form(1.0),
    format: str = Form("wav"),
    bitrate: str = Form(None),
    gap: float = Form(0.12),
    save: bool = Form(False),
):
    """Browser-friendly multipart endpoint used by the bundled web UI."""
    req = SpeakRequest(text=text, voice=voice, speed=speed, format=format,
                       gap=gap, bitrate=bitrate)
    chunks = E.get_engine().synthesize(
        text=req.text, voice=req.voice, speed=req.speed, split_pattern=req.split_pattern
    )
    audio, content_type = E.encode(chunks, req.format, gap=req.gap,
                                   bitrate=req.bitrate)

    if save:
        E.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = E.OUTPUT_DIR / f"{_safe_voice(req.voice)}-{uuid.uuid4().hex[:8]}.{req.format}"
        path.write_bytes(audio)
        return JSONResponse({"path": str(path), "bytes": path.stat().st_size})

    stem = f"kokoro-{_safe_voice(req.voice)}-{uuid.uuid4().hex[:8]}.{req.format}"
    return Response(
        content=audio,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{stem}"'},
    )


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    safe = Path(filename).name
    path = (E.OUTPUT_DIR / safe).resolve()
    if not str(path).startswith(str(E.OUTPUT_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="application/octet-stream", filename=safe)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():  # pragma: no cover
        return HTMLResponse("<h1>Kokoro API is running</h1><p>See <a href='/docs'>/docs</a>.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.on_event("startup")
def _startup() -> None:
    E.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ helpers

def _safe_voice(voice: str) -> str:
    return "".join(c for c in voice if c.isalnum() or c in "-_") or "voice"


def _log(text: str, voice: str, speed: float, fmt: str, elapsed: float, chunks: int) -> None:
    REQUEST_LOG.append(
        {
            "voice": voice,
            "speed": speed,
            "format": fmt,
            "chars": len(text),
            "chunks": chunks,
            "seconds": elapsed,
            "at": time.strftime("%H:%M:%S"),
        }
    )
    del REQUEST_LOG[:-100]


def jsonable(model: SpeakMeta) -> dict:
    return model.model_dump()
