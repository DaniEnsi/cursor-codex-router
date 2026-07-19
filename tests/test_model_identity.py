"""Seam tests for shared model-id vocabulary."""

from __future__ import annotations

from cursor_codex_router.model_identity import (
    TO_CODEX,
    parse_agent_models,
    peel_agent_model,
    split_model,
)


def test_split_and_peel_share_vocabulary() -> None:
    cases = [
        (
            "cursor-grok-4.5-high",
            ("cursor-grok-4.5", False, "high", False),
            ("cursor-grok-4.5", "high", False),
        ),
        (
            "cursor-grok-4.5-high-fast",
            ("cursor-grok-4.5", False, "high", True),
            ("cursor-grok-4.5", "high", True),
        ),
        (
            "claude-opus-4-thinking-medium",
            ("claude-opus-4", True, "medium", False),
            ("claude-opus-4-thinking", "medium", False),
        ),
        (
            "claude-opus-4-thinking",
            ("claude-opus-4", True, None, False),
            ("claude-opus-4-thinking", None, False),
        ),
        ("auto", ("auto", False, None, False), ("auto", None, False)),
        (
            "claude-4.5-opus-high-thinking",
            ("claude-4.5-opus", True, "high", False),
            ("claude-4.5-opus-thinking", "high", False),
        ),
    ]
    for mid, split_expected, peel_expected in cases:
        assert split_model(mid) == split_expected, mid
        assert peel_agent_model(mid) == peel_expected, mid


def test_peel_derived_from_split() -> None:
    mid = "composer-2.5-thinking-extra-high-fast"
    base, thinking, effort, fast = split_model(mid)
    slug = f"{base}-thinking" if thinking else base
    assert peel_agent_model(mid) == (slug, effort, fast)


def test_parse_agent_models_accepts_dash_variants() -> None:
    rows = parse_agent_models(
        "Available models\n"
        "cursor-grok-4.5-high - Cursor Grok 4.5 High\n"
        "auto — Auto (default)\n"
        "composer-2.5\n"
    )
    assert [mid for mid, _ in rows] == [
        "cursor-grok-4.5-high",
        "auto",
        "composer-2.5",
    ]
    assert rows[0][1] == "Cursor Grok 4.5 High"
    assert "default" not in rows[1][1].lower()


def test_to_codex_maps_cursor_efforts() -> None:
    assert TO_CODEX["none"] == "low"
    assert TO_CODEX["extra-high"] == "xhigh"
    assert TO_CODEX["high"] == "high"
