"""40-word speaker-aware segment grouping.

Takes a flat list of ``word_segments`` (as produced by
:func:`whisperx.align` / :func:`whisperx.assign_word_speakers`) and groups
them into ~40-word chunks, always breaking when the speaker changes so that
a chunk never straddles speakers.

Each word dict is expected to roughly look like::

    {"word": "hello", "start": 0.12, "end": 0.43, "speaker": "SPEAKER_00"}

Some entries (punctuation-only, failed alignment) may be missing ``start`` /
``end``; we forward them into the chunk they belong to but rely on surrounding
words for the chunk's own ``start`` / ``end`` bounds.
"""

from __future__ import annotations

from typing import Iterable


DEFAULT_TARGET_WORDS = 40


def group_word_segments(
    words: Iterable[dict],
    target_words: int = DEFAULT_TARGET_WORDS,
) -> list[dict]:
    """Group ``words`` into ~``target_words``-sized segments, split by speaker.

    Rules:
      * A segment always breaks when the speaker label changes.
      * Otherwise, a segment closes as soon as its word count reaches
        ``target_words``.
      * A trailing short segment is kept as-is (not padded / merged).
      * Each returned segment carries ``index``, ``start``, ``end``, ``speaker``
        (the first word's speaker, or ``None``), ``text`` (joined words with
        spaces), and the raw ``words`` list.

    Empty input returns an empty list.
    """

    word_list = list(words)
    if not word_list:
        return []

    groups: list[dict] = []
    current: list[dict] = []
    current_speaker: str | None | object = _UNSET
    target = max(1, int(target_words))

    def _flush() -> None:
        if not current:
            return
        groups.append(_finalize(current, len(groups)))
        current.clear()

    for w in word_list:
        speaker = w.get("speaker")
        if current_speaker is _UNSET:
            current_speaker = speaker
        elif speaker != current_speaker:
            _flush()
            current_speaker = speaker

        current.append(w)

        if len(current) >= target:
            _flush()
            current_speaker = _UNSET

    _flush()
    return groups


_UNSET = object()


def _finalize(words: list[dict], index: int) -> dict:
    starts = [w["start"] for w in words if isinstance(w.get("start"), (int, float))]
    ends = [w["end"] for w in words if isinstance(w.get("end"), (int, float))]
    speaker = next((w.get("speaker") for w in words if w.get("speaker")), None)
    text = " ".join(str(w.get("word", "")).strip() for w in words if w.get("word"))
    return {
        "index": index,
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
        "speaker": speaker,
        "text": text,
        "words": list(words),
    }
