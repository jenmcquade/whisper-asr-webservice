"""Tests for the model-tier resolver."""

from __future__ import annotations

import pytest

from app.models import MODEL_CHOICES, get_choice, resolve_checkpoint


def test_tier_without_language_returns_multilingual():
    assert resolve_checkpoint("base", language=None) == "base"
    assert resolve_checkpoint("medium", language=None) == "medium"


def test_tier_with_english_flips_to_en_variant():
    assert resolve_checkpoint("tiny", language="en") == "tiny.en"
    assert resolve_checkpoint("base", language="en") == "base.en"
    assert resolve_checkpoint("small", language="en") == "small.en"
    assert resolve_checkpoint("medium", language="en") == "medium.en"


def test_large_and_turbo_have_no_en_variant():
    # Large and turbo don't ship an English-only checkpoint, so even
    # when the language is English they stay on the multilingual build.
    assert resolve_checkpoint("large", language="en") == "large-v3"
    assert resolve_checkpoint("turbo", language="en") == "large-v3-turbo"


def test_large_v3_branding_is_transparent():
    # The tier id is "large" (no version); the checkpoint is versioned.
    assert resolve_checkpoint("large", language=None) == "large-v3"
    assert resolve_checkpoint("turbo", language=None) == "large-v3-turbo"


def test_non_english_language_stays_multilingual():
    for lang in ("es", "fr", "ja", "zh"):
        assert resolve_checkpoint("medium", language=lang) == "medium"
        assert resolve_checkpoint("large", language=lang) == "large-v3"


def test_raw_checkpoint_passthrough():
    # ASR_MODEL may be set to a raw checkpoint like "medium.en" or
    # "large-v3"; the resolver should return it verbatim.
    assert resolve_checkpoint("large-v3", language=None) == "large-v3"
    assert resolve_checkpoint("medium.en", language=None) == "medium.en"


def test_raw_checkpoint_upgrades_to_en_when_english():
    # "medium" passed as a raw name should still flip to medium.en
    # when the resolved language is English.
    assert resolve_checkpoint("medium", language="en") == "medium.en"


@pytest.mark.parametrize("tier_id", [m.id for m in MODEL_CHOICES])
def test_every_tier_has_a_choice(tier_id: str):
    assert get_choice(tier_id) is not None
