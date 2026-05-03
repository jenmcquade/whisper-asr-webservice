"""Whisper model catalog + resolver.

The frontend exposes a user-friendly set of model choices (tiny / base /
small / medium / large / turbo). Internally we need to map each choice to
the actual checkpoint id understood by WhisperX / faster-whisper, and
transparently prefer the English-only ``.en`` variants when the caller
has locked the language to English (or it gets detected as English).

Two rules to note:

* ``large`` and ``turbo`` refer to the v3 checkpoints (``large-v3`` and
  ``large-v3-turbo``). We don't mention the version in the UI — callers
  just pick "large" or "turbo" and we pick the latest known-good build.
* Only ``tiny``, ``base``, ``small``, and ``medium`` ship ``.en``
  variants; ``large*`` / ``turbo`` do not, so the English fast-path is a
  no-op for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModelChoice:
    """One user-facing model tier."""

    id: str  # stable identifier used in URLs/forms (never shown w/ version)
    label: str  # human-readable label shown in the UI
    description: str
    base_checkpoint: str  # multilingual checkpoint name
    en_checkpoint: str | None  # English-only checkpoint, if available


# Keep this list small and opinionated. Labels intentionally omit "-v3".
MODEL_CHOICES: tuple[ModelChoice, ...] = (
    ModelChoice(
        id="tiny",
        label="Tiny",
        description="Fastest. ~1 GB VRAM. Best for quick drafts.",
        base_checkpoint="tiny",
        en_checkpoint="tiny.en",
    ),
    ModelChoice(
        id="base",
        label="Base",
        description="Fast. ~1 GB VRAM. Great dev default.",
        base_checkpoint="base",
        en_checkpoint="base.en",
    ),
    ModelChoice(
        id="small",
        label="Small",
        description="Balanced. ~2 GB VRAM.",
        base_checkpoint="small",
        en_checkpoint="small.en",
    ),
    ModelChoice(
        id="medium",
        label="Medium",
        description="Accurate. ~5 GB VRAM.",
        base_checkpoint="medium",
        en_checkpoint="medium.en",
    ),
    ModelChoice(
        id="large",
        label="Large",
        description="Highest quality. ~10 GB VRAM.",
        base_checkpoint="large-v3",
        en_checkpoint=None,
    ),
    ModelChoice(
        id="turbo",
        label="Turbo",
        description="Optimized large, ~8× faster, minimal quality loss.",
        base_checkpoint="large-v3-turbo",
        en_checkpoint=None,
    ),
)


_BY_ID: dict[str, ModelChoice] = {m.id: m for m in MODEL_CHOICES}


def choices() -> Iterable[ModelChoice]:
    return MODEL_CHOICES


def get_choice(model_id: str) -> ModelChoice | None:
    return _BY_ID.get(model_id)


# Recognised checkpoint ids (for back-compat when callers pass the raw
# checkpoint name instead of our tier id).
_KNOWN_CHECKPOINTS: set[str] = set()
for _m in MODEL_CHOICES:
    _KNOWN_CHECKPOINTS.add(_m.base_checkpoint)
    if _m.en_checkpoint:
        _KNOWN_CHECKPOINTS.add(_m.en_checkpoint)


def resolve_checkpoint(
    model_id: str,
    *,
    language: str | None,
) -> str:
    """Return the actual checkpoint id to load.

    ``model_id`` may be:

    * one of our tier ids (``tiny``, ``base``, ``small``, ``medium``,
      ``large``, ``turbo``);
    * a raw checkpoint name (``large-v3``, ``medium.en``, ...) — used
      internally when ``ASR_MODEL`` is set via env.

    If ``language`` is English (``en``) and the tier has an
    ``en_checkpoint``, we transparently swap to that.
    """

    mid = (model_id or "").strip()
    is_english = (language or "").lower().startswith("en")

    choice = _BY_ID.get(mid)
    if choice is not None:
        if is_english and choice.en_checkpoint:
            return choice.en_checkpoint
        return choice.base_checkpoint

    # Fall-through: caller handed us a raw checkpoint name. Honour it
    # verbatim but still flip to .en when a matching variant exists and
    # the language is English.
    if is_english:
        # e.g. "medium" -> "medium.en", "tiny" -> "tiny.en"
        en_guess = f"{mid}.en" if mid and "." not in mid else None
        if en_guess and en_guess in _KNOWN_CHECKPOINTS:
            return en_guess
    return mid


def resolve_choice_id(model_id: str | None) -> str | None:
    """If ``model_id`` is a known tier id return it; otherwise ``None``."""

    if not model_id:
        return None
    return model_id if model_id in _BY_ID else None
