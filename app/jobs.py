"""In-memory job registry for async ASR jobs.

Each job lives in a ``dict[str, Job]`` guarded by ``_LOCK``. A job carries
its status + last-known progress snapshot (so HTTP poll endpoints can return
useful data) plus an :class:`asyncio.Queue` that the SSE endpoint drains.

The pipeline runs on a worker thread (via ``loop.run_in_executor``). The
worker publishes progress events by calling :meth:`Job.publish` from the
worker thread, which hops back onto the event loop via
``loop.call_soon_threadsafe`` to land them in the asyncio queue safely.

A background task sweeps expired jobs every ``TTL_SWEEP_SECONDS`` and deletes
the on-disk upload + decoded WAV once the job is older than
``JOB_TTL_SECONDS``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
import uuid
from threading import Lock
from typing import Any

logger = logging.getLogger("whisper-asr-webservice.jobs")

JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
TTL_SWEEP_SECONDS = int(os.getenv("TTL_SWEEP_SECONDS", "300"))

# When ``KEEP_ARTEFACTS`` is true the sweeper still evicts jobs from the
# in-memory registry (so the dict doesn't leak) but leaves the on-disk
# upload + decoded WAV in place. Useful for local development where
# ``UPLOAD_DIR=./output`` and you want to inspect or re-use artefacts.
_KEEP_ARTEFACTS_ENV = os.getenv("KEEP_ARTEFACTS")
KEEP_ARTEFACTS = (
    (_KEEP_ARTEFACTS_ENV or "").strip().lower() in {"1", "true", "yes", "on"}
)

# Sentinel pushed into a job's queue when the pipeline finishes so the SSE
# generator can close cleanly without waiting for timeout.
STREAM_DONE = object()


class JobCancelled(Exception):
    """Raised by the pipeline when the user cancels a job mid-flight."""


@dataclasses.dataclass
class Job:
    """A single ASR job. Mutable; fields updated from the worker thread."""

    id: str
    created_at: float
    upload_path: str | None = None
    wav_path: str | None = None
    original_filename: str | None = None
    params: dict[str, Any] = dataclasses.field(default_factory=dict)

    status: str = "queued"  # queued | running | done | error | cancelled
    stage: str | None = None
    stage_percent: float = 0.0
    stage_eta_seconds: float | None = None

    probe: dict | None = None
    result: dict | None = None
    error: str | None = None
    asr_checkpoint: str | None = None  # set by the pipeline once model is resolved

    stages_done: list[str] = dataclasses.field(default_factory=list)

    result_path: str | None = None

    # Cancellation: set via POST /asr/jobs/{id}/cancel. The pipeline worker
    # polls :meth:`check_cancel` at stage boundaries and raises ``JobCancelled``
    # to unwind cleanly. In-flight ffmpeg decodes are interrupted by killing
    # their subprocess.
    cancel_requested: bool = False
    cancelled_at: float | None = None

    queue: "asyncio.Queue[Any]" = dataclasses.field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = None
    finished_at: float | None = None

    def publish(self, event: dict) -> None:
        """Thread-safe: push ``event`` onto the SSE queue from the worker."""

        loop = self.loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:  # loop closed, client gone
            pass

    def publish_done(self) -> None:
        loop = self.loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self.queue.put_nowait, STREAM_DONE)
        except RuntimeError:  # pragma: no cover
            pass

    def request_cancel(self) -> bool:
        """Mark the job for cancellation. Returns True if it took effect.

        Terminal states (done/error/cancelled) are no-ops; the caller can
        use the return value to short-circuit a redundant response.
        """

        if self.status in ("done", "error", "cancelled"):
            return False
        if self.cancel_requested:
            return False
        self.cancel_requested = True
        self.cancelled_at = time.time()
        return True

    def check_cancel(self) -> None:
        """Raise :class:`JobCancelled` if cancellation has been requested.

        Called from the pipeline worker at stage boundaries.
        """

        if self.cancel_requested:
            raise JobCancelled()


_LOCK = Lock()
_JOBS: dict[str, Job] = {}


def create_job() -> Job:
    job = Job(
        id=uuid.uuid4().hex,
        created_at=time.time(),
        loop=asyncio.get_running_loop(),
    )
    with _LOCK:
        _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def all_jobs() -> list[Job]:
    with _LOCK:
        return list(_JOBS.values())


def remove_job(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.pop(job_id, None)


# ---------------------------------------------------------------------------
# TTL sweeper
# ---------------------------------------------------------------------------

async def ttl_cleanup_loop(
    ttl_seconds: int = JOB_TTL_SECONDS,
    sweep_interval: int = TTL_SWEEP_SECONDS,
    keep_artefacts: bool | None = None,
) -> None:
    """Periodically delete expired jobs + their on-disk files.

    If ``keep_artefacts`` is True, the in-memory job entry is still evicted
    when it exceeds ``ttl_seconds`` but the upload/WAV files are left on
    disk for manual inspection.
    """

    if keep_artefacts is None:
        keep_artefacts = KEEP_ARTEFACTS

    while True:
        try:
            _sweep_once(ttl_seconds, keep_artefacts=keep_artefacts)
        except Exception as exc:  # pragma: no cover
            logger.warning("TTL sweep failed: %s", exc)
        await asyncio.sleep(sweep_interval)


def _sweep_once(ttl_seconds: int, *, keep_artefacts: bool = False) -> None:
    now = time.time()
    victims: list[Job] = []
    with _LOCK:
        for job in list(_JOBS.values()):
            anchor = job.finished_at or job.created_at
            if (now - anchor) >= ttl_seconds:
                victims.append(job)
                _JOBS.pop(job.id, None)

    for job in victims:
        if keep_artefacts:
            logger.info(
                "Expired job %s evicted from registry (artefacts kept at %s)",
                job.id,
                job.upload_path,
            )
            continue
        for p in (job.upload_path, job.wav_path, job.result_path):
            if not p:
                continue
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except OSError as exc:  # pragma: no cover
                logger.warning("Failed to delete %s: %s", p, exc)
        logger.info("Expired job %s cleaned up", job.id)
