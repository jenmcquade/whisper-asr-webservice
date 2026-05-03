"""Whisper ASR webservice powered by WhisperX.

Job-based API:

  * ``POST /asr/jobs`` — upload + start. Returns ``{ "job_id": ..., ... }``.
  * ``GET /asr/jobs/{id}/events`` — SSE progress stream.
  * ``GET /asr/jobs/{id}/audio`` — original upload with ``Range`` support.
  * ``GET /asr/jobs/{id}/probe`` — ffprobe JSON (available once probed).
  * ``GET /asr/jobs/{id}/result`` — final JSON (available when done).

Backwards-compat synchronous endpoints:

  * ``POST /asr`` — runs the pipeline and returns the same JSON as
    ``/asr/jobs/{id}/result`` once it completes (no streaming).
  * ``POST /detect-language`` — unchanged.

Results always include ``word_segments`` (single-word granularity) and
``segments_grouped`` (40-word speaker-aware chunks for the UI).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import mimetypes
import os
import tempfile
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env (and any parent .env on the CWD chain) before reading env vars.
# `override=False` means real environment variables still win over the file,
# which matches how docker-compose env_file works.
load_dotenv(override=False)

import ffmpeg
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    applications,
)
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from . import jobs, pipeline
from .jobs import Job, STREAM_DONE

logger = logging.getLogger("whisper-asr-webservice")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

SAMPLE_RATE = 16000

# Default output location:
#   - In Docker, compose sets ``UPLOAD_DIR=/data/uploads`` on a shared volume.
#   - For local runs we write to ``<project-root>/output`` so artefacts are
#     easy to find, inspect, and clean up (the directory is gitignored).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_UPLOAD_DIR = os.path.join(_PROJECT_ROOT, "output")
_UPLOAD_DIR_FROM_ENV = os.getenv("UPLOAD_DIR")
UPLOAD_DIR = _UPLOAD_DIR_FROM_ENV or _DEFAULT_UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

# When the user didn't set ``UPLOAD_DIR`` explicitly we assume a local
# development run and keep artefacts on disk by default; otherwise respect
# ``KEEP_ARTEFACTS`` (already parsed in ``app.jobs``) as-is.
if _UPLOAD_DIR_FROM_ENV is None and os.getenv("KEEP_ARTEFACTS") is None:
    jobs.KEEP_ARTEFACTS = True

logger.info(
    "Using UPLOAD_DIR=%s (keep_artefacts=%s)",
    UPLOAD_DIR,
    jobs.KEEP_ARTEFACTS,
)


from faster_whisper.tokenizer import _LANGUAGE_CODES  # type: ignore

LANGUAGE_CODES = sorted(_LANGUAGE_CODES)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    logger.info(
        "Starting whisper-asr-webservice on device=%s compute_type=%s model=%s",
        pipeline.DEVICE,
        pipeline.COMPUTE_TYPE,
        pipeline.ASR_MODEL,
    )
    pipeline.get_asr_model()
    cleanup_task = asyncio.create_task(jobs.ttl_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    lifespan=_lifespan,
)

assets_path = os.path.join(os.getcwd(), "swagger-ui-assets")
if (
    os.path.exists(os.path.join(assets_path, "swagger-ui.css"))
    and os.path.exists(os.path.join(assets_path, "swagger-ui-bundle.js"))
):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def _swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = _swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index() -> str:
    return "/docs"


# ---------------------------------------------------------------------------
# Job-based API
# ---------------------------------------------------------------------------


async def _save_upload_to_disk(upload: UploadFile, job: Job) -> None:
    """Stream the upload to ``UPLOAD_DIR/<job_id>.<ext>`` and record on the job."""

    original_name = upload.filename or "upload.bin"
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".bin"
    upload_path = os.path.join(UPLOAD_DIR, f"{job.id}{ext}")
    wav_path = os.path.join(UPLOAD_DIR, f"{job.id}.wav")

    # Stream to disk so huge uploads don't blow up RAM.
    chunk = 1024 * 1024
    with open(upload_path, "wb") as fh:
        while True:
            data = await upload.read(chunk)
            if not data:
                break
            fh.write(data)

    job.upload_path = upload_path
    job.wav_path = wav_path
    job.original_filename = original_name


@app.post("/asr/jobs", tags=["Jobs"])
async def create_asr_job(
    request: Request,
    audio_file: UploadFile = File(...),
    task: str = Form(default="transcribe"),
    language: str | None = Form(default=None),
    initial_prompt: str | None = Form(default=None),
    model: str | None = Form(default=None),
    diarize: bool = Form(default=False),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
):
    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail="task must be transcribe|translate")
    if language not in (None, "", *LANGUAGE_CODES):
        raise HTTPException(status_code=400, detail=f"unknown language: {language}")

    job = jobs.create_job()
    job.params = {
        "task": task,
        "language": language or None,
        "initial_prompt": initial_prompt or None,
        "model": model or None,
        "diarize": bool(diarize),
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }

    await _save_upload_to_disk(audio_file, job)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, pipeline.run_job, job)

    return JSONResponse(
        {
            "job_id": job.id,
            "status": job.status,
            "events_url": f"/asr/jobs/{job.id}/events",
            "audio_url": f"/asr/jobs/{job.id}/audio",
            "result_url": f"/asr/jobs/{job.id}/result",
            "probe_url": f"/asr/jobs/{job.id}/probe",
        }
    )


@app.get("/asr/jobs/{job_id}/events", tags=["Jobs"])
async def job_events(job_id: str, request: Request):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def generator():
        # Replay snapshot so a late-connecting client still sees current state.
        if job.probe is not None:
            yield {"event": "probe", "data": json.dumps(job.probe)}

        # Replay completed stages at 100% so their bars aren't stuck at
        # "Waiting…" when we reconnect to a job that's already past them.
        for done_stage in job.stages_done:
            if done_stage == job.stage:
                continue  # currently-running stage gets its own replay below
            yield {
                "event": "stage",
                "data": json.dumps({"stage": done_stage, "started": True}),
            }
            yield {
                "event": "progress",
                "data": json.dumps(
                    {"stage": done_stage, "percent": 100.0, "eta_seconds": 0.0}
                ),
            }

        if job.stage is not None:
            yield {
                "event": "stage",
                "data": json.dumps({"stage": job.stage, "started": True}),
            }
            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "stage": job.stage,
                        "percent": round(job.stage_percent, 2),
                        "eta_seconds": (
                            round(job.stage_eta_seconds, 2)
                            if job.stage_eta_seconds is not None
                            else None
                        ),
                    }
                ),
            }
        if job.status == "error":
            yield {"event": "error", "data": json.dumps({"message": job.error or ""})}
        if job.status == "cancelled":
            yield {
                "event": "cancelled",
                "data": json.dumps({"stage": job.stage}),
            }
        if job.status == "done" and job.result is not None:
            yield {"event": "result", "data": json.dumps(job.result)}
        if job.status in ("done", "error", "cancelled"):
            yield {"event": "done", "data": json.dumps({"status": job.status})}
            return

        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(job.queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Keep-alive comment so proxies don't close idle SSE.
                yield {"event": "ping", "data": "{}"}
                continue
            if event is STREAM_DONE:
                break
            yield {
                "event": event.get("event", "message"),
                "data": json.dumps(event.get("data", {})),
            }

    return EventSourceResponse(generator())


@app.get("/asr/jobs/{job_id}/audio", tags=["Jobs"])
async def job_audio(job_id: str):
    job = jobs.get_job(job_id)
    if job is None or not job.upload_path or not os.path.exists(job.upload_path):
        raise HTTPException(status_code=404, detail="audio not found")
    media_type, _ = mimetypes.guess_type(job.original_filename or job.upload_path)
    return FileResponse(
        job.upload_path,
        media_type=media_type or "application/octet-stream",
        filename=job.original_filename,
    )


@app.get("/asr/jobs/{job_id}/probe", tags=["Jobs"])
async def job_probe(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.probe is None:
        raise HTTPException(status_code=409, detail="probe not ready")
    return JSONResponse(job.probe)


@app.get("/asr/jobs/{job_id}/result", tags=["Jobs"])
async def job_result(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "pipeline failed")
    if job.result is None:
        raise HTTPException(status_code=409, detail="result not ready")
    return JSONResponse(job.result)


@app.post("/asr/jobs/{job_id}/cancel", tags=["Jobs"])
async def cancel_job(job_id: str):
    """Request cancellation of a running job.

    Returns 200 with the (possibly updated) status. No-op if the job is
    already in a terminal state. The pipeline worker polls the cancel flag
    at stage boundaries and raises ``JobCancelled`` to unwind cleanly.
    """

    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    requested = job.request_cancel()
    return JSONResponse(
        {
            "job_id": job.id,
            "status": job.status,
            "cancel_requested": job.cancel_requested,
            "cancel_accepted": requested,
        }
    )


@app.get("/asr/jobs/{job_id}", tags=["Jobs"])
async def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(
        {
            "id": job.id,
            "status": job.status,
            "stage": job.stage,
            "stage_percent": round(job.stage_percent, 2),
            "stage_eta_seconds": (
                round(job.stage_eta_seconds, 2)
                if job.stage_eta_seconds is not None
                else None
            ),
            "asr_checkpoint": job.asr_checkpoint,
            "params": job.params,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
        }
    )


# ---------------------------------------------------------------------------
# Legacy / misc endpoints
# ---------------------------------------------------------------------------


@app.post("/asr", tags=["Endpoints"])
async def transcribe(
    audio_file: UploadFile = File(...),
    task: str = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: str | None = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: str | None = Query(default=None),
    model: str | None = Query(default=None, description="Model tier: tiny|base|small|medium|large|turbo"),
    diarize: bool = Query(default=False, description="Run speaker diarization"),
    min_speakers: int | None = Query(default=None, ge=1),
    max_speakers: int | None = Query(default=None, ge=1),
):
    """Synchronous transcribe. Internally runs a job and waits for completion."""

    job = jobs.create_job()
    job.params = {
        "task": task,
        "language": language,
        "initial_prompt": initial_prompt,
        "model": model,
        "diarize": diarize,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }
    await _save_upload_to_disk(audio_file, job)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, pipeline.run_job, job)

    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "pipeline failed")
    return JSONResponse(job.result or {})


@app.post("/detect-language", tags=["Endpoints"])
async def language_detection(audio_file: UploadFile = File(...)):
    audio_path = _save_upload_as_wav(audio_file)
    try:
        language_code = pipeline.detect_language_sync(audio_path)
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    try:
        from whisper.tokenizer import LANGUAGES as _LANGS  # type: ignore

        language_name = _LANGS.get(language_code, language_code)
    except Exception:  # pragma: no cover
        language_name = language_code
    return {"detected_language": language_name, "language_code": language_code}


def _save_upload_as_wav(upload: UploadFile) -> str:
    """Decode ``upload`` to a 16 kHz mono s16le WAV and return the path.

    Kept only for ``/detect-language``, which doesn't need progress reporting.
    """

    fd, out_path = tempfile.mkstemp(prefix="asr_", suffix=".wav")
    os.close(fd)
    try:
        data = upload.file.read()
        (
            ffmpeg.input("pipe:", threads=0)
            .output(out_path, acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE, f="wav")
            .overwrite_output()
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=data)
        )
    except ffmpeg.Error as exc:
        try:
            os.remove(out_path)
        except OSError:
            pass
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else str(exc)
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {stderr}")
    return out_path
