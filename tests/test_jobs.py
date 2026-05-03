"""Tests for the job TTL sweeper, especially the KEEP_ARTEFACTS path."""

from __future__ import annotations

import os
import time

import pytest

from app import jobs


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test runs against an empty in-memory job registry."""

    with jobs._LOCK:
        jobs._JOBS.clear()
    yield
    with jobs._LOCK:
        jobs._JOBS.clear()


def _make_expired_job(tmp_path, *, age_seconds: float = 10_000.0) -> jobs.Job:
    upload = tmp_path / "job.mp3"
    wav = tmp_path / "job.wav"
    result = tmp_path / "job.json"
    upload.write_bytes(b"fake-mp3")
    wav.write_bytes(b"fake-wav")
    result.write_text('{"ok": true}')

    job = jobs.Job(
        id="expired",
        created_at=time.time() - age_seconds,
        upload_path=str(upload),
        wav_path=str(wav),
        result_path=str(result),
        finished_at=time.time() - age_seconds,
        status="done",
    )
    with jobs._LOCK:
        jobs._JOBS[job.id] = job
    return job


def test_sweep_deletes_files_by_default(tmp_path):
    job = _make_expired_job(tmp_path)

    jobs._sweep_once(ttl_seconds=60)

    assert not os.path.exists(job.upload_path)
    assert not os.path.exists(job.wav_path)
    assert not os.path.exists(job.result_path)
    assert jobs.get_job(job.id) is None


def test_sweep_keeps_files_when_keep_artefacts(tmp_path):
    job = _make_expired_job(tmp_path)

    jobs._sweep_once(ttl_seconds=60, keep_artefacts=True)

    # In-memory record is still evicted so the registry can't grow unbounded,
    # but the on-disk artefacts are preserved for local inspection.
    assert jobs.get_job(job.id) is None
    assert os.path.exists(job.upload_path)
    assert os.path.exists(job.wav_path)
    assert os.path.exists(job.result_path)


def test_sweep_skips_unexpired_jobs(tmp_path):
    # young job -> should remain in the registry with files intact.
    young = jobs.Job(
        id="young",
        created_at=time.time(),
        upload_path=str(tmp_path / "young.mp3"),
        wav_path=str(tmp_path / "young.wav"),
        status="running",
    )
    (tmp_path / "young.mp3").write_bytes(b"x")
    (tmp_path / "young.wav").write_bytes(b"y")
    with jobs._LOCK:
        jobs._JOBS[young.id] = young

    jobs._sweep_once(ttl_seconds=60)

    assert jobs.get_job(young.id) is young
    assert os.path.exists(young.upload_path)
