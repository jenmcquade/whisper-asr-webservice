"""Synchronous ASR pipeline: decode -> probe -> transcribe -> align -> diarize.

Kept separate from :mod:`app.webservice` so the FastAPI routes stay focused on
request/response shapes. This module does the heavy lifting on a worker thread
(started by ``loop.run_in_executor``) and reports progress through the
:class:`~app.jobs.Job.publish` hook.

Each stage emits two events:

  * ``{"event": "stage", "data": {"stage": ..., "started": true}}``
  * ``{"event": "progress", "data": {"stage": ..., "percent": ..., "eta_seconds": ...}}``
    (zero or more times while the stage runs)

Plus the terminal events:

  * ``probe`` (once, after ffprobe)
  * ``result`` (once, at the end)
  * ``error`` (if anything goes wrong)
  * ``done`` (always, last)
"""

from __future__ import annotations

import json
import logging
import os
import time
from threading import Lock
from typing import Any

import torch
import whisperx
from fastapi import HTTPException

from .ffprobe import FFmpegError, decode_to_wav, probe_file
from .jobs import Job, JobCancelled
from .models import resolve_checkpoint
from .segments import group_word_segments

logger = logging.getLogger("whisper-asr-webservice.pipeline")

# ``ASR_MODEL`` sets the *default* model when a request doesn't pick one.
# It accepts either a tier id (``large``, ``turbo``, ...) or a raw
# checkpoint name (``medium.en``, ``large-v3``, ...). Defaults to
# ``large`` which resolves to ``large-v3``.
ASR_MODEL = os.getenv("ASR_MODEL", "large")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("PYANNOTE_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))


def _resolve_device() -> str:
    override = os.getenv("DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_compute_type(device: str) -> str:
    override = os.getenv("COMPUTE_TYPE")
    if override:
        return override
    if device == "cuda":
        return "float16"
    return "int8"


DEVICE = _resolve_device()
COMPUTE_TYPE = _resolve_compute_type(DEVICE)

# Cache of loaded WhisperX models, keyed by checkpoint id (e.g. "base",
# "large-v3", "medium.en"). Keeping them around means switching models
# at request time is a no-op after the first load of each.
_asr_models: dict[str, Any] = {}
_asr_model_lock = Lock()
_align_cache: dict[str, tuple[object, dict]] = {}
_align_lock = Lock()
_diarize_pipeline: Any = None
_diarize_lock = Lock()
_model_lock = Lock()


def get_asr_model(checkpoint: str | None = None) -> Any:
    """Return a cached WhisperX model for ``checkpoint``.

    Falls back to the ``ASR_MODEL`` default (resolved to a multilingual
    checkpoint) when no checkpoint is given — that's the path used at
    startup to warm the default model.
    """

    ckpt = checkpoint or resolve_checkpoint(ASR_MODEL, language=None)
    cached = _asr_models.get(ckpt)
    if cached is not None:
        return cached
    with _asr_model_lock:
        cached = _asr_models.get(ckpt)
        if cached is not None:
            return cached
        logger.info(
            "Loading WhisperX checkpoint=%s device=%s compute_type=%s",
            ckpt,
            DEVICE,
            COMPUTE_TYPE,
        )
        model = whisperx.load_model(
            ckpt,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        _asr_models[ckpt] = model
        return model


def _get_align_model(language_code: str):
    with _align_lock:
        cached = _align_cache.get(language_code)
        if cached is not None:
            return cached
        logger.info("Loading align model for language=%s", language_code)
        model, metadata = whisperx.load_align_model(
            language_code=language_code, device=DEVICE
        )
        _align_cache[language_code] = (model, metadata)
        return model, metadata


def _get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is not None:
        return _diarize_pipeline
    with _diarize_lock:
        if _diarize_pipeline is not None:
            return _diarize_pipeline
        if not HF_TOKEN:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Diarization requested but HF_TOKEN is not set. Set HF_TOKEN "
                    "in the environment and accept the pyannote/speaker-diarization-3.1 "
                    "terms on HuggingFace."
                ),
            )
        logger.info("Loading diarization pipeline on device=%s", DEVICE)
        from whisperx.diarize import DiarizationPipeline

        _diarize_pipeline = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
        return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage progress tracker
# ---------------------------------------------------------------------------


class _Stage:
    """Helper that emits ``stage`` + ``progress`` events with ETA.

    Progress is expected to arrive as a float 0-100. ETA is computed by rolling
    elapsed time over observed percent. Emission is throttled — we skip events
    where percent hasn't changed by at least ``min_delta``.
    """

    def __init__(
        self,
        job: Job,
        stage: str,
        *,
        min_delta: float = 1.0,
        min_interval_seconds: float = 0.25,
    ):
        self.job = job
        self.stage = stage
        self.min_delta = min_delta
        self.min_interval_seconds = min_interval_seconds
        self.started_at: float | None = None
        self._last_percent: float = -1.0
        self._last_emit: float = 0.0

    def start(self) -> None:
        self.started_at = time.time()
        self.job.stage = self.stage
        self.job.stage_percent = 0.0
        self.job.stage_eta_seconds = None
        self.job.publish({
            "event": "stage",
            "data": {"stage": self.stage, "started": True},
        })
        self.emit(0.0)

    def emit(self, percent: float) -> None:
        now = time.time()
        pct = max(0.0, min(100.0, float(percent)))
        if (
            pct - self._last_percent < self.min_delta
            and now - self._last_emit < self.min_interval_seconds
            and pct < 100.0
        ):
            return
        self._last_percent = pct
        self._last_emit = now

        eta: float | None = None
        if self.started_at is not None and pct > 0:
            elapsed = now - self.started_at
            if pct < 100.0:
                eta = elapsed / pct * (100.0 - pct)

        self.job.stage_percent = pct
        self.job.stage_eta_seconds = eta
        self.job.publish({
            "event": "progress",
            "data": {
                "stage": self.stage,
                "percent": round(pct, 2),
                "eta_seconds": round(eta, 2) if eta is not None else None,
            },
        })

    def finish(self) -> None:
        self.emit(100.0)
        if self.stage and self.stage not in self.job.stages_done:
            self.job.stages_done.append(self.stage)


# ---------------------------------------------------------------------------
# Worker entrypoint
# ---------------------------------------------------------------------------


def run_job(job: Job) -> None:
    """Run the full pipeline for ``job``. Invoked on a worker thread.

    Reads ``job.upload_path`` + ``job.wav_path`` (wav_path is where we'll write
    the decoded wav) and ``job.params``. Publishes events along the way and
    finally writes ``job.result`` / ``job.error`` + publishes ``done``.
    """

    job.status = "running"
    try:
        _run_unlocked(job)
    except JobCancelled:
        logger.info("Job %s cancelled by user", job.id)
        job.status = "cancelled"
        job.error = None
        job.publish({"event": "cancelled", "data": {"stage": job.stage}})
    except HTTPException as exc:  # pyannote / HF_TOKEN etc.
        job.status = "error"
        job.error = str(exc.detail)
        job.publish({"event": "error", "data": {"message": job.error}})
    except FFmpegError as exc:
        # When the caller requested cancellation, the ffmpeg decode raises
        # FFmpegError with a cancelled-marker message — translate into the
        # cancelled flow rather than reporting a pipeline error.
        if job.cancel_requested and "cancelled" in str(exc).lower():
            logger.info("Job %s cancelled during decode", job.id)
            job.status = "cancelled"
            job.error = None
            job.publish({"event": "cancelled", "data": {"stage": job.stage}})
        else:
            job.status = "error"
            job.error = f"ffmpeg failed: {exc.stderr or exc}"
            job.publish({"event": "error", "data": {"message": job.error}})
    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job.id)
        job.status = "error"
        job.error = str(exc) or exc.__class__.__name__
        job.publish({"event": "error", "data": {"message": job.error}})
    else:
        job.status = "done"
    finally:
        job.finished_at = time.time()
        job.publish({"event": "done", "data": {"status": job.status}})
        job.publish_done()


def _run_unlocked(job: Job) -> None:
    assert job.upload_path is not None
    assert job.wav_path is not None
    params = job.params

    # ----- probe ----------------------------------------------------------
    job.check_cancel()
    probe_stage = _Stage(job, "probe")
    probe_stage.start()
    probe = probe_file(job.upload_path)
    job.probe = probe.to_dict()
    probe_stage.finish()
    job.publish({"event": "probe", "data": job.probe})

    # ----- decode ---------------------------------------------------------
    job.check_cancel()
    decode_stage = _Stage(job, "decode")
    decode_stage.start()
    decode_to_wav(
        input_path=job.upload_path,
        output_path=job.wav_path,
        total_duration_seconds=probe.duration_seconds,
        on_progress=decode_stage.emit,
        should_cancel=lambda: job.cancel_requested,
    )
    # If ffmpeg returned normally but cancellation was requested mid-stream,
    # raise here so we skip the transcribe/align/diarize stages.
    job.check_cancel()
    decode_stage.finish()

    # ----- transcribe -----------------------------------------------------
    job.check_cancel()
    transcribe_stage = _Stage(job, "transcribe")
    transcribe_stage.start()

    audio = whisperx.load_audio(job.wav_path)
    asr_options: dict = {"task": params.get("task", "transcribe")}
    # Note: ``initial_prompt`` is not a kwarg on whisperx's transcribe(); it's
    # applied at model-load time via ASR_OPTIONS. Accepting it from the
    # request is intentionally a no-op today — TODO wire into load_model
    # kwargs if/when per-request prompts matter.
    if params.get("initial_prompt"):
        logger.debug(
            "initial_prompt provided but not yet plumbed through whisperx transcribe()"
        )

    # Resolve which WhisperX checkpoint to actually load. ``model`` from
    # the request (e.g. "base", "large") is a user-facing tier; map it
    # through models.resolve_checkpoint which also flips to the ``.en``
    # variant when the language is English.
    #
    # Translation tasks must use the multilingual checkpoint even if the
    # target language is English (``.en`` models can't translate).
    req_model = params.get("model") or ASR_MODEL
    req_language = params.get("language")
    use_en_fastpath = (
        params.get("task", "transcribe") == "transcribe"
        and (req_language or "").lower().startswith("en")
    )

    if use_en_fastpath:
        checkpoint = resolve_checkpoint(req_model, language="en")
    elif req_language:
        # Non-English, non-auto — load the multilingual checkpoint for that tier.
        checkpoint = resolve_checkpoint(req_model, language=None)
    else:
        # Auto-detect path: use the multilingual checkpoint for a quick
        # language probe, then swap to ``.en`` for the actual transcribe
        # if we detect English and the tier has a ``.en`` variant.
        checkpoint = resolve_checkpoint(req_model, language=None)

    model = get_asr_model(checkpoint)

    # Auto-detect path: run a cheap language probe on the multilingual
    # model, then upgrade to the ``.en`` checkpoint for transcribe when
    # English is detected.
    if not req_language and params.get("task", "transcribe") == "transcribe":
        try:
            with _model_lock:
                detected = model.detect_language(audio)
        except Exception as exc:
            logger.warning("Language detection pre-pass failed: %s", exc)
            detected = None
        if detected and detected.lower().startswith("en"):
            en_ckpt = resolve_checkpoint(req_model, language="en")
            if en_ckpt != checkpoint:
                logger.info(
                    "Detected English; swapping checkpoint %s -> %s",
                    checkpoint,
                    en_ckpt,
                )
                checkpoint = en_ckpt
                model = get_asr_model(checkpoint)
            # Pin the language so transcribe skips its own detection pass.
            asr_options["language"] = "en"

    job.asr_checkpoint = checkpoint

    with _model_lock:
        result = model.transcribe(
            audio,
            batch_size=BATCH_SIZE,
            language=asr_options.pop("language", req_language),
            progress_callback=transcribe_stage.emit,
            **asr_options,
        )
    transcribe_stage.finish()
    job.check_cancel()

    detected_language = result.get("language", params.get("language") or "en")

    word_segments: list[dict] = []
    segments_grouped: list[dict] = []

    if result.get("segments"):
        # ----- align -------------------------------------------------------
        align_stage = _Stage(job, "align")
        align_stage.start()
        try:
            align_model, align_metadata = _get_align_model(detected_language)
            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
                progress_callback=align_stage.emit,
            )
        except Exception as exc:
            logger.warning(
                "Alignment failed for language=%s: %s", detected_language, exc
            )
            aligned = {"segments": result["segments"], "word_segments": []}
        align_stage.finish()
        job.check_cancel()

        # ----- diarize ----------------------------------------------------
        if params.get("diarize"):
            diarize_stage = _Stage(job, "diarize")
            diarize_stage.start()
            pipeline = _get_diarize_pipeline()
            diarize_kwargs: dict = {"progress_callback": diarize_stage.emit}
            if params.get("min_speakers") is not None:
                diarize_kwargs["min_speakers"] = params["min_speakers"]
            if params.get("max_speakers") is not None:
                diarize_kwargs["max_speakers"] = params["max_speakers"]
            diarize_segments = pipeline(job.wav_path, **diarize_kwargs)
            aligned = whisperx.assign_word_speakers(diarize_segments, aligned)
            diarize_stage.finish()

        word_segments = aligned.get("word_segments", []) or []
        segments_grouped = group_word_segments(word_segments)

    job.result = {
        "language": detected_language,
        "word_segments": word_segments,
        "segments_grouped": segments_grouped,
    }

    # Persist the result JSON alongside the upload + decoded WAV so users can
    # pick it up from the output directory without hitting the API. Failures
    # here are non-fatal — the in-memory result is still served via HTTP.
    if job.wav_path:
        try:
            result_path = os.path.splitext(job.wav_path)[0] + ".json"
            with open(result_path, "w", encoding="utf-8") as fh:
                json.dump(job.result, fh, ensure_ascii=False, indent=2)
            job.result_path = result_path
        except OSError as exc:
            logger.warning("Failed to write result JSON for job %s: %s", job.id, exc)

    job.publish({"event": "result", "data": job.result})


# ---------------------------------------------------------------------------
# Synchronous helper for the legacy /asr endpoint and /detect-language
# ---------------------------------------------------------------------------


def detect_language_sync(wav_path: str) -> str:
    audio = whisperx.load_audio(wav_path)
    with _model_lock:
        return get_asr_model().detect_language(audio)
