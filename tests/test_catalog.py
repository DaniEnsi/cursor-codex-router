"""Unit tests for catalog grouping / display."""

from __future__ import annotations

from cursor_codex_router.catalog import (
    assert_clean_catalog,
    nice_display,
    parse_agent_models,
    prune_slugs,
    split_model,
)


def test_parse_and_split() -> None:
    rows = parse_agent_models(
        "Available models\n"
        "cursor-grok-4.5-high - Cursor Grok 4.5 High\n"
        "cursor-grok-4.5-high-fast - Cursor Grok 4.5 High Fast\n"
        "auto - Auto (default)\n"
    )
    assert rows[0][0] == "cursor-grok-4.5-high"
    assert split_model("cursor-grok-4.5-high-fast") == ("cursor-grok-4.5", False, "high", True)
    assert split_model("auto") == ("auto", False, None, False)


def test_nice_display_branded() -> None:
    assert nice_display("gpt-5.6-sol", False, "5.6 Sol").startswith("GPT")
    assert "High" not in nice_display("cursor-grok-4.5", False, "Cursor Grok 4.5 High")
    assert "Thinking" not in nice_display("claude-opus-4", True, "Opus 4 Thinking")


def test_prune_prefers_thinking_twin() -> None:
    kept = prune_slugs(["composer-2.5", "composer-2.5-thinking"])
    assert "composer-2.5-thinking" in kept
    assert "composer-2.5" not in kept


def test_assert_clean_catalog() -> None:
    models = [
        {
            "slug": "cursor-grok-4.5",
            "display_name": "Cursor Grok 4.5",
            "supported_reasoning_levels": [{"effort": "high"}],
        }
    ]
    assert_clean_catalog(models)

    dirty = [
        {
            "slug": "cursor-grok-4.5-high",
            "display_name": "Cursor Grok 4.5 High",
            "supported_reasoning_levels": [{"effort": "high"}],
        }
    ]
    try:
        assert_clean_catalog(dirty)
        raise AssertionError("expected SystemExit")
    except SystemExit:
        pass
