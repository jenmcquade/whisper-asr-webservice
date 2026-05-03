"""Tests for :mod:`app.segments` — the 40-word speaker-aware chunker."""

from __future__ import annotations

from app.segments import group_word_segments


def _word(text: str, start: float, end: float, speaker: str | None = None) -> dict:
    w = {"word": text, "start": start, "end": end, "score": 1.0}
    if speaker is not None:
        w["speaker"] = speaker
    return w


def test_empty_input_returns_empty_list():
    assert group_word_segments([]) == []


def test_single_speaker_chunks_at_target_size():
    words = [_word(f"w{i}", i * 0.1, i * 0.1 + 0.08, "SPEAKER_00") for i in range(95)]
    groups = group_word_segments(words, target_words=40)

    # 40, 40, 15 -> 3 chunks.
    assert [len(g["words"]) for g in groups] == [40, 40, 15]
    assert all(g["speaker"] == "SPEAKER_00" for g in groups)
    assert [g["index"] for g in groups] == [0, 1, 2]
    assert groups[0]["start"] == 0.0
    assert groups[0]["end"] > groups[0]["start"]
    assert groups[-1]["end"] >= groups[-1]["start"]


def test_speaker_change_forces_split_even_when_short():
    words = (
        [_word(f"a{i}", i * 0.1, i * 0.1 + 0.08, "SPEAKER_00") for i in range(5)]
        + [_word(f"b{i}", 1 + i * 0.1, 1 + i * 0.1 + 0.08, "SPEAKER_01") for i in range(3)]
        + [_word(f"c{i}", 2 + i * 0.1, 2 + i * 0.1 + 0.08, "SPEAKER_00") for i in range(2)]
    )
    groups = group_word_segments(words, target_words=40)

    assert [g["speaker"] for g in groups] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [len(g["words"]) for g in groups] == [5, 3, 2]


def test_speaker_change_mid_target_splits_cleanly():
    # Cross the target-word boundary while speaker is also changing.
    first = [_word(f"a{i}", i * 0.1, i * 0.1 + 0.08, "SPEAKER_00") for i in range(30)]
    second = [_word(f"b{i}", 10 + i * 0.1, 10 + i * 0.1 + 0.08, "SPEAKER_01") for i in range(20)]
    groups = group_word_segments(first + second, target_words=40)

    assert [len(g["words"]) for g in groups] == [30, 20]
    assert groups[0]["speaker"] == "SPEAKER_00"
    assert groups[1]["speaker"] == "SPEAKER_01"


def test_missing_speaker_is_treated_as_none_and_grouped_together():
    words = [_word(f"w{i}", i * 0.1, i * 0.1 + 0.08) for i in range(10)]
    groups = group_word_segments(words, target_words=40)

    assert len(groups) == 1
    assert groups[0]["speaker"] is None


def test_text_joins_words_with_spaces():
    words = [
        _word("Hello", 0.1, 0.4, "SPEAKER_00"),
        _word("world.", 0.5, 0.9, "SPEAKER_00"),
    ]
    groups = group_word_segments(words, target_words=40)
    assert groups[0]["text"] == "Hello world."


def test_missing_start_end_tolerated():
    words = [
        {"word": ",", "speaker": "SPEAKER_00"},
        _word("Hello", 0.1, 0.4, "SPEAKER_00"),
    ]
    groups = group_word_segments(words, target_words=40)
    assert len(groups) == 1
    assert groups[0]["start"] == 0.1
    assert groups[0]["end"] == 0.4
    assert len(groups[0]["words"]) == 2
