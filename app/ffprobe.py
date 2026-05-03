"""ffmpeg + ffprobe helpers with progress-reporting decode.

``whisperx.load_audio`` shells out to the ``ffmpeg`` CLI in a blocking
``subprocess.run`` with no progress reporting. For the SSE UI we need a
progress signal per decode, so we roll our own: spawn ``ffmpeg -progress pipe:1``
and parse ``out_time_us=...`` vs. the total duration we got from ffprobe.

We also use ffprobe to populate the "file info" block on the job page:
container / codec / sample rate / channels / duration / bitrate / tags.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable

logger = logging.getLogger("whisper-asr-webservice.ffprobe")

SAMPLE_RATE = 16000


class FFmpegError(RuntimeError):
    """Wraps a non-zero ffmpeg exit, carrying captured stderr."""

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"{name} binary not found on PATH")
    return path


@dataclass
class ProbeResult:
    """Normalized subset of ffprobe output for the UI."""

    format_name: str | None
    codec_name: str | None
    duration_seconds: float | None
    bit_rate: int | None
    sample_rate: int | None
    channels: int | None
    tags: dict[str, str]
    raw: dict

    def to_dict(self) -> dict:
        return {
            "format_name": self.format_name,
            "codec_name": self.codec_name,
            "duration_seconds": self.duration_seconds,
            "bit_rate": self.bit_rate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "tags": self.tags,
        }


def probe_file(path: str) -> ProbeResult:
    """Run ffprobe on ``path`` and return a normalized :class:`ProbeResult`."""

    ffprobe = _require_binary("ffprobe")
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-hide_banner",
                "-loglevel", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise FFmpegError("ffprobe failed", stderr=exc.stderr or "")

    raw = json.loads(completed.stdout or "{}")
    fmt = raw.get("format", {}) or {}
    streams = raw.get("streams", []) or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None) or {}

    def _maybe_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _maybe_int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return ProbeResult(
        format_name=fmt.get("format_name"),
        codec_name=audio.get("codec_name"),
        duration_seconds=_maybe_float(fmt.get("duration") or audio.get("duration")),
        bit_rate=_maybe_int(fmt.get("bit_rate") or audio.get("bit_rate")),
        sample_rate=_maybe_int(audio.get("sample_rate")),
        channels=_maybe_int(audio.get("channels")),
        tags={str(k): str(v) for k, v in (fmt.get("tags") or {}).items()},
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Decoding with progress
# ---------------------------------------------------------------------------

_PROGRESS_LINE = re.compile(r"^([A-Za-z_]+)=(.*)$")


def parse_progress_block(lines: Iterable[str]) -> dict[str, str]:
    """Parse one ``-progress pipe:1`` block into a dict.

    ffmpeg emits key=value pairs separated by newlines and terminated by a
    ``progress=continue`` or ``progress=end`` line. We accept any iterable
    of lines covering one block and return the raw kv map.
    """

    out: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = _PROGRESS_LINE.match(line)
        if not m:
            continue
        out[m.group(1)] = m.group(2)
    return out


def decode_to_wav(
    input_path: str,
    output_path: str,
    total_duration_seconds: float | None,
    on_progress: Callable[[float], None] | None = None,
    sample_rate: int = SAMPLE_RATE,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Decode ``input_path`` to 16 kHz mono s16le WAV at ``output_path``.

    ``on_progress`` (if given) is called with a float 0-100 as ffmpeg reports
    ``out_time_us``. The final call is always ``100.0`` on success.

    ``should_cancel`` (if given) is polled between progress blocks. When it
    returns truthy we SIGTERM the ffmpeg subprocess and raise
    :class:`FFmpegError` with a "cancelled" message so the caller can
    unwind.
    """

    ffmpeg = _require_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-y",
        output_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    block: list[str] = []
    cancelled = False
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            block.append(line)
            if not line.startswith("progress="):
                continue
            kv = parse_progress_block(block)
            block = []
            if on_progress is not None and total_duration_seconds:
                pct = _pct_from_progress(kv, total_duration_seconds)
                if pct is not None:
                    on_progress(pct)
            if should_cancel is not None and should_cancel():
                cancelled = True
                try:
                    proc.terminate()
                except OSError:  # pragma: no cover
                    pass
                break
            if kv.get("progress") == "end":
                break
    finally:
        try:
            proc.stdout.close()
        except Exception:  # pragma: no cover
            pass
        ret = proc.wait()
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:  # pragma: no cover
                pass
        if cancelled:
            raise FFmpegError("ffmpeg decode cancelled", stderr=stderr)
        if ret != 0:
            raise FFmpegError(
                f"ffmpeg decode failed with exit code {ret}", stderr=stderr
            )
        if on_progress is not None:
            on_progress(100.0)


def _pct_from_progress(kv: dict[str, str], total_seconds: float) -> float | None:
    """Compute 0-100 from an ffmpeg ``-progress`` block."""

    out_time_us = kv.get("out_time_us") or kv.get("out_time_ms")
    if out_time_us is None:
        return None
    try:
        us = int(out_time_us)
    except ValueError:
        return None
    # ffmpeg's "out_time_ms" field is, confusingly, microseconds.
    seconds = us / 1_000_000.0
    if total_seconds <= 0:
        return None
    pct = max(0.0, min(100.0, (seconds / total_seconds) * 100.0))
    return pct
