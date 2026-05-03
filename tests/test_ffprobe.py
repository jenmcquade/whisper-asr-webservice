"""Tests for :mod:`app.ffprobe` — specifically the progress-block parser."""

from __future__ import annotations

from app.ffprobe import _pct_from_progress, parse_progress_block


def test_parses_key_value_block():
    lines = [
        "fps=0.00\n",
        "out_time_us=1234567\n",
        "speed=1.5x\n",
        "progress=continue\n",
    ]
    kv = parse_progress_block(lines)
    assert kv["fps"] == "0.00"
    assert kv["out_time_us"] == "1234567"
    assert kv["speed"] == "1.5x"
    assert kv["progress"] == "continue"


def test_ignores_blank_and_malformed_lines():
    lines = ["\n", "garbage", "out_time_us=500000\n", "", "progress=end\n"]
    kv = parse_progress_block(lines)
    assert kv == {"out_time_us": "500000", "progress": "end"}


def test_pct_from_progress_basic():
    # 5 seconds into a 10 second clip => 50%.
    kv = {"out_time_us": str(5_000_000)}
    assert _pct_from_progress(kv, 10.0) == 50.0


def test_pct_from_progress_clamped_to_100():
    kv = {"out_time_us": str(20_000_000)}
    assert _pct_from_progress(kv, 10.0) == 100.0


def test_pct_from_progress_handles_out_time_ms_alias():
    # ffmpeg's `out_time_ms` is actually microseconds despite the name.
    kv = {"out_time_ms": str(2_500_000)}
    assert _pct_from_progress(kv, 10.0) == 25.0


def test_pct_from_progress_none_without_data():
    assert _pct_from_progress({}, 10.0) is None


def test_pct_from_progress_none_when_duration_zero():
    kv = {"out_time_us": "1000000"}
    assert _pct_from_progress(kv, 0.0) is None
