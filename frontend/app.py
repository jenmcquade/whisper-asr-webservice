"""Podchaser-themed HTMX frontend for the Whisper ASR backend.

Runs on port 9001. The browser only ever talks to this process, which proxies
uploads, SSE, audio, probe, and result fetches through to the backend on
``ASR_BACKEND_URL`` (default ``http://127.0.0.1:9000``). This keeps CORS out
of the picture and lets us render Jinja fragments that HTMX SSE swaps into
the live job page.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv(override=False)

logger = logging.getLogger("whisper-asr-frontend")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

BACKEND_URL = os.getenv("ASR_BACKEND_URL", "http://127.0.0.1:9000").rstrip("/")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Whisper ASR Frontend")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


LANGUAGE_CHOICES = [
    ("", "Auto-detect"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("zh", "Chinese"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
    ("ru", "Russian"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("pl", "Polish"),
    ("tr", "Turkish"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("no", "Norwegian"),
    ("hu", "Hungarian"),
    ("ro", "Romanian"),
    ("he", "Hebrew"),
    ("id", "Indonesian"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("uk", "Ukrainian"),
    ("el", "Greek"),
]

STAGES = [
    ("probe", "Inspect audio"),
    ("decode", "Decode to WAV"),
    ("transcribe", "Transcribe"),
    ("align", "Align words"),
    ("diarize", "Diarize speakers"),
]

# User-facing model tiers. We intentionally omit the "-v3" suffix from
# the labels — "Large" resolves to ``large-v3`` on the backend and
# "Turbo" to ``large-v3-turbo``. Descriptions match the sizes/perf
# guidance from https://github.com/openai/whisper.
MODEL_CHOICES = [
    ("tiny",   "Tiny",   "~1 GB VRAM · ~10× faster than large"),
    ("base",   "Base",   "~1 GB VRAM · ~7× faster than large"),
    ("small",  "Small",  "~2 GB VRAM · ~4× faster than large"),
    ("medium", "Medium", "~5 GB VRAM · ~2× faster than large"),
    ("large",  "Large",  "~10 GB VRAM · highest quality"),
    ("turbo",  "Turbo",  "~6 GB VRAM · ~8× faster than large, minimal quality loss"),
]
DEFAULT_MODEL_CHOICE = os.getenv("FRONTEND_DEFAULT_MODEL", "base")
VALID_MODEL_IDS = {m[0] for m in MODEL_CHOICES}


def _client() -> httpx.AsyncClient:
    # A per-request client keeps connection lifetime aligned with the request
    # (especially important for long-lived SSE streams).
    return httpx.AsyncClient(base_url=BACKEND_URL, timeout=None)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "languages": LANGUAGE_CHOICES,
            "models": MODEL_CHOICES,
            "default_model": DEFAULT_MODEL_CHOICE,
            "stages": STAGES,
            "backend_url": BACKEND_URL,
        },
    )


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str):
    diarize_enabled = True
    already_done = False
    job_active = True  # whether the Stop button should be visible on load
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            r = await client.get(f"{BACKEND_URL}/asr/jobs/{job_id}")
        if r.status_code == 200:
            status = r.json()
            if "params" in status:
                params = status.get("params") or {}
                diarize_enabled = _truthy(params.get("diarize"), default=False)
            already_done = status.get("status") == "done"
            if status.get("status") in ("done", "error", "cancelled"):
                job_active = False
    except Exception:
        pass

    stages = [
        (code, label)
        for code, label in STAGES
        if code != "diarize" or diarize_enabled
    ]

    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "job_id": job_id,
            "stages": stages,
            "diarize_enabled": diarize_enabled,
            "already_done": already_done,
            "job_active": job_active,
            "backend_url": BACKEND_URL,
        },
    )


def _truthy(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Proxies to backend
# ---------------------------------------------------------------------------


@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    audio_file: UploadFile = File(...),
    task: str = Form(default="transcribe"),
    language: str = Form(default=""),
    initial_prompt: str = Form(default=""),
    model: str = Form(default=""),
    diarize: str | None = Form(default=None),
    min_speakers: str = Form(default=""),
    max_speakers: str = Form(default=""),
):
    form_data: dict = {"task": task}
    if language:
        form_data["language"] = language
    if initial_prompt:
        form_data["initial_prompt"] = initial_prompt
    if model and model in VALID_MODEL_IDS:
        form_data["model"] = model
    # The upload form sends ``diarize`` only when the checkbox is ticked
    # (typically value ``"on"`` or ``"true"``); curl/API callers may send
    # ``"false"`` explicitly. Treat common falsy strings as not-enabled.
    diarize_enabled = False
    if diarize is not None:
        diarize_enabled = diarize.strip().lower() not in {"", "0", "false", "no", "off"}
    form_data["diarize"] = "true" if diarize_enabled else "false"
    if min_speakers:
        form_data["min_speakers"] = min_speakers
    if max_speakers:
        form_data["max_speakers"] = max_speakers

    payload = await audio_file.read()
    files = {
        "audio_file": (
            audio_file.filename or "upload.bin",
            payload,
            audio_file.content_type or "application/octet-stream",
        )
    }

    async with _client() as client:
        resp = await client.post("/asr/jobs", data=form_data, files=files)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    body = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        raise HTTPException(status_code=502, detail="backend returned no job_id")

    # HTMX sent hx-post and wants an HX-Redirect for a full-page swap to the
    # job page; plain browsers fall back to a 303 redirect.
    if request.headers.get("HX-Request"):
        return HTMLResponse(
            "",
            status_code=200,
            headers={"HX-Redirect": f"/jobs/{job_id}"},
        )
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/detect-language", response_class=HTMLResponse)
async def detect_language(request: Request, audio_file: UploadFile = File(...)):
    payload = await audio_file.read()
    files = {
        "audio_file": (
            audio_file.filename or "upload.bin",
            payload,
            audio_file.content_type or "application/octet-stream",
        )
    }
    async with _client() as client:
        resp = await client.post("/detect-language", files=files)
    if resp.status_code >= 400:
        return HTMLResponse(
            f"<div class='result-card error'>Language detection failed: "
            f"{resp.status_code}</div>",
            status_code=200,
        )
    body = resp.json()
    return HTMLResponse(
        f"<div class='result-card'>Detected: <strong>{body.get('detected_language')}</strong> "
        f"<span class='muted'>({body.get('language_code')})</span></div>"
    )


@app.get("/jobs/{job_id}/events")
async def jobs_events(job_id: str, request: Request):
    """Proxy the backend SSE stream.

    We render each event as a pre-formatted HTML fragment so HTMX can
    ``sse-swap`` it directly into the DOM. The frontend keeps an eye on
    ``done`` and closes the connection client-side.
    """

    async def event_stream() -> AsyncIterator[bytes]:
        async with _client() as client:
            async with client.stream("GET", f"/asr/jobs/{job_id}/events") as upstream:
                if upstream.status_code != 200:
                    body = await upstream.aread()
                    yield _sse_frame("error", f"Upstream status {upstream.status_code}: {body.decode(errors='ignore')}")
                    return

                current_event = "message"
                buffer: list[str] = []
                async for raw_line in upstream.aiter_lines():
                    if raw_line == "":
                        # End of one SSE record -> render + forward.
                        data_str = "\n".join(buffer).strip()
                        buffer = []
                        if data_str:
                            async for out in _render_event(current_event, data_str):
                                yield out
                        current_event = "message"
                    elif raw_line.startswith("event:"):
                        current_event = raw_line[len("event:") :].strip()
                    elif raw_line.startswith("data:"):
                        buffer.append(raw_line[len("data:") :].lstrip())
                    elif raw_line.startswith(":"):
                        # Keep-alive comment; pass through.
                        yield (raw_line + "\n\n").encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
async def jobs_cancel(job_id: str):
    """Proxy POST /asr/jobs/{id}/cancel to the backend.

    Returns a tiny HTML fragment so the HTMX-driven button can swap a
    confirmation notice in place. The SSE ``cancelled`` event handles
    the broader UI transition.
    """

    try:
        async with _client() as client:
            resp = await client.post(f"/asr/jobs/{job_id}/cancel")
    except httpx.HTTPError as exc:
        logger.warning("cancel proxy failed for %s: %s", job_id, exc)
        return HTMLResponse(
            "<div class='notice notice-error'>Could not reach backend to cancel.</div>",
            status_code=502,
        )
    if resp.status_code >= 400:
        return HTMLResponse(
            f"<div class='notice notice-error'>Cancel failed: HTTP {resp.status_code}</div>",
            status_code=resp.status_code,
        )
    return HTMLResponse(
        "<div class='notice notice-cancelled'>Cancellation requested…</div>"
    )


@app.get("/jobs/{job_id}/audio")
async def jobs_audio(job_id: str, request: Request):
    headers = {}
    if "range" in request.headers:
        headers["Range"] = request.headers["range"]

    async def iter_body(resp: httpx.Response) -> AsyncIterator[bytes]:
        async for chunk in resp.aiter_bytes():
            yield chunk

    client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=None)
    try:
        resp = await client.send(
            client.build_request("GET", f"/asr/jobs/{job_id}/audio", headers=headers),
            stream=True,
        )
    except Exception:
        await client.aclose()
        raise

    async def passthrough() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    forwarded_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() in {"content-type", "content-length", "content-range", "accept-ranges"}
    }
    return StreamingResponse(
        passthrough(),
        status_code=resp.status_code,
        headers=forwarded_headers,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
    )


@app.get("/jobs/{job_id}/probe")
async def jobs_probe(job_id: str):
    async with _client() as client:
        resp = await client.get(f"/asr/jobs/{job_id}/probe")
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/jobs/{job_id}/result")
async def jobs_result(job_id: str):
    async with _client() as client:
        resp = await client.get(f"/asr/jobs/{job_id}/result")
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/jobs/{job_id}/result.json")
async def jobs_result_download(job_id: str):
    """Same as /result, but sent as an attachment download."""

    async with _client() as client:
        resp = await client.get(f"/asr/jobs/{job_id}/result")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return Response(
        content=resp.content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'},
    )


# ---------------------------------------------------------------------------
# SSE rendering helpers
# ---------------------------------------------------------------------------


async def _render_event(event: str, data_str: str) -> AsyncIterator[bytes]:
    """Yield one or more SSE frames. Rendered fragments are HTML; JSON-only
    events (``result``) also get a second frame carrying the raw JSON under a
    distinct SSE event name so the client JS can build the word array."""

    if event == "ping":
        yield _sse_frame("ping", "")
        return

    try:
        data = json.loads(data_str) if data_str else {}
    except json.JSONDecodeError:
        yield _sse_frame("error", f"<div class='error'>Bad event payload from backend</div>")
        return

    if event == "stage":
        stage = data.get("stage")
        yield _sse_frame(f"stage:{stage}", _stage_started_html(stage))
        return

    if event == "progress":
        stage = data.get("stage")
        yield _sse_frame(f"progress:{stage}", _progress_html(stage, data))
        return

    if event == "probe":
        yield _sse_frame("probe", _probe_html(data))
        return

    if event == "result":
        yield _sse_frame("result", _result_html(data))
        yield _sse_frame("result-json", json.dumps(data))
        return

    if event == "error":
        msg = data.get("message") or str(data)
        yield _sse_frame(
            "error",
            f"<div class='error'>Pipeline error: {_esc(msg)}</div>",
        )
        return

    if event == "done":
        yield _sse_frame("done", f"<span class='badge badge-{_esc(data.get('status', ''))}'>"
                                 f"{_esc(data.get('status', 'done'))}</span>")
        return

    if event == "cancelled":
        yield _sse_frame(
            "cancelled",
            "<div class='notice notice-cancelled'>Job cancelled.</div>",
        )
        return

    # Passthrough unknown events so we don't drop data silently.
    yield _sse_frame(event, _esc(data_str))


def _sse_frame(event: str, data: str) -> bytes:
    # Normalise any embedded newlines/tabs so the HTML fragment is sent as a
    # single SSE data line. Multi-data-line SSE events are legal but some
    # browsers (and HTMX's sse-swap extension) reconstruct them with literal
    # ``\n`` joined in, which can corrupt HTML attributes.
    flat = data.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return f"event: {event}\ndata: {flat}\n\n".encode("utf-8")


def _stage_started_html(stage: str | None) -> str:
    return (
        f"<div class='bar-fill' style='width: 2%'></div>"
        f"<span class='bar-label'>Starting {_esc(_stage_label(stage))}…</span>"
    )


def _progress_html(stage: str | None, data: dict) -> str:
    pct = float(data.get("percent") or 0.0)
    eta = data.get("eta_seconds")
    eta_html = f" ETA {_fmt_eta(eta)}" if eta is not None else ""
    return (
        f"<div class='bar-fill' style='width: {pct:.1f}%'></div>"
        f"<span class='bar-label'>{pct:.0f}%{_esc(eta_html)}</span>"
    )


def _probe_html(probe: dict) -> str:
    duration = probe.get("duration_seconds")
    tags = probe.get("tags") or {}

    def _chip(key: str, value: str) -> str:
        # Long free-text values (podcast descriptions, comments, etc.) get a
        # block-level class so they wrap cleanly instead of rendering as one
        # giant pill.
        cls = "chip"
        if len(value) > 80 or "\n" in value:
            cls = "chip chip-long"
        return f"<span class='{cls}'>{_esc(str(key))}: {_esc(str(value))}</span>"

    tag_pills = "".join(_chip(k, str(v)) for k, v in list(tags.items())[:8])
    return (
        "<div class='probe'>"
        f"  <div class='probe-row'><span class='muted'>Container</span> "
        f"<strong>{_esc(str(probe.get('format_name') or '—'))}</strong></div>"
        f"  <div class='probe-row'><span class='muted'>Codec</span> "
        f"<strong>{_esc(str(probe.get('codec_name') or '—'))}</strong></div>"
        f"  <div class='probe-row'><span class='muted'>Sample rate</span> "
        f"<strong>{_esc(_fmt_hz(probe.get('sample_rate')))}</strong></div>"
        f"  <div class='probe-row'><span class='muted'>Channels</span> "
        f"<strong>{_esc(str(probe.get('channels') or '—'))}</strong></div>"
        f"  <div class='probe-row'><span class='muted'>Bitrate</span> "
        f"<strong>{_esc(_fmt_bitrate(probe.get('bit_rate')))}</strong></div>"
        f"  <div class='probe-row'><span class='muted'>Duration</span> "
        f"<strong>{_esc(_fmt_duration(duration))}</strong></div>"
        f"  <div class='probe-tags'>{tag_pills}</div>"
        "</div>"
    )


def _result_html(result: dict) -> str:
    segments = result.get("segments_grouped") or []
    if not segments:
        return "<div class='muted'>No transcript available.</div>"
    out = [
        "<div class='segments'>",
        f"<div class='result-head'><span class='muted'>Language</span> "
        f"<strong>{_esc(str(result.get('language') or '—'))}</strong>"
        f"  <span class='muted'>· {len(segments)} segments</span></div>",
    ]
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        speaker = seg.get("speaker")
        speaker_html = (
            f"<span class='speaker'>{_esc(str(speaker))}</span>" if speaker else ""
        )
        words_html = "".join(_word_span(w) for w in seg.get("words", []))
        out.append(
            f"<article class='segment' data-seek='{start:.3f}' "
            f"tabindex='0' role='button' aria-label='Play segment {seg.get('index', 0) + 1}'>"
            f"<header>{speaker_html}"
            f"<span class='time'>{_esc(_fmt_duration(start))}</span></header>"
            f"<p class='words'>{words_html}</p>"
            f"</article>"
        )
    out.append("</div>")
    return "".join(out)


def _word_span(word: dict) -> str:
    start = word.get("start")
    end = word.get("end")
    text = _esc(str(word.get("word", "")))
    attrs = ""
    if isinstance(start, (int, float)):
        attrs += f" data-start='{float(start):.3f}'"
    if isinstance(end, (int, float)):
        attrs += f" data-end='{float(end):.3f}'"
    return f"<span class='word'{attrs}>{text} </span>"


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    if h:
        return f"{h}:{m:02d}:{sec:05.2f}"
    return f"{m}:{sec:05.2f}"


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(round(float(seconds)))
    if s <= 0:
        return "a moment"
    if s < 60:
        return f"{s}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m}m {rem:02d}s"
    h, rem_m = divmod(m, 60)
    return f"{h}h {rem_m:02d}m"


def _fmt_bitrate(b: int | None) -> str:
    if not b:
        return "—"
    kbps = int(b) / 1000
    if kbps >= 1000:
        return f"{kbps / 1000:.1f} Mbps"
    return f"{kbps:.0f} kbps"


def _fmt_hz(hz: int | None) -> str:
    if not hz:
        return "—"
    khz = int(hz) / 1000
    return f"{khz:.1f} kHz"


def _stage_label(stage: str | None) -> str:
    for code, label in STAGES:
        if code == stage:
            return label
    return stage or ""


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok", "backend": BACKEND_URL, "time": datetime.utcnow().isoformat()})
