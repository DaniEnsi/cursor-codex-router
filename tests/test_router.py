"""Unit tests for model resolution and prompt shaping."""

from __future__ import annotations

from pathlib import Path

import pytest

from cursor_codex_router import router


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_WORKSPACE", str(tmp_path / "ws"))
    # Reload path-derived module globals used by router
    monkeypatch.setattr(router, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(router, "KEY_PATH", tmp_path / "state" / "api_key")
    monkeypatch.setattr(router, "LOG_PATH", tmp_path / "state" / "router.log")
    monkeypatch.setattr(router, "WORKSPACE", tmp_path / "ws")
    monkeypatch.setattr(router, "_api_key_cache", None)
    monkeypatch.setattr(router, "_models_cache", {"ts": 0.0, "ids": []})
    router.set_effort_map_store(None)
    router.set_agent_runner(None)


def test_peel_agent_model() -> None:
    assert router.peel_agent_model("cursor-grok-4.5-high") == ("cursor-grok-4.5", "high", False)
    assert router.peel_agent_model("cursor-grok-4.5-high-fast") == ("cursor-grok-4.5", "high", True)
    assert router.peel_agent_model("claude-opus-4-thinking-medium") == (
        "claude-opus-4-thinking",
        "medium",
        False,
    )


def test_resolve_model_uses_effort_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cursor_codex_router.effort_map import (
        EffortMap,
        EffortSlot,
        MemoryEffortMapStore,
        ModelEfforts,
    )

    emap = EffortMap(
        {
            "cursor-grok-4.5": ModelEfforts(
                base="cursor-grok-4.5",
                thinking=False,
                default_effort="high",
                has_fast=True,
                efforts={
                    "low": EffortSlot(
                        normal="cursor-grok-4.5-low",
                        fast="cursor-grok-4.5-low-fast",
                    ),
                    "high": EffortSlot(
                        normal="cursor-grok-4.5-high",
                        fast="cursor-grok-4.5-high-fast",
                    ),
                },
            )
        }
    )
    router.set_effort_map_store(MemoryEffortMapStore(emap))
    monkeypatch.setattr(
        router,
        "list_models",
        lambda force=False: [
            "cursor-grok-4.5-high",
            "cursor-grok-4.5-high-fast",
            "cursor-grok-4.5-low",
            "cursor-grok-4.5-low-fast",
        ],
    )

    echo, agent = router.resolve_model(
        "cursor-grok-4.5",
        {"reasoning": {"effort": "high"}},
    )
    assert echo == "cursor-grok-4.5"
    assert agent == "cursor-grok-4.5-high"

    echo, agent = router.resolve_model(
        "cursor-grok-4.5",
        {"reasoning": {"effort": "low"}, "service_tier": "fast"},
    )
    assert echo == "cursor-grok-4.5"
    assert agent == "cursor-grok-4.5-low-fast"


def test_messages_strip_tool_schema() -> None:
    huge = '{"type": "function", "parameters": ' + ("x" * 3000) + ', "tool_choice": 1}'
    prompt = router.messages_to_prompt(
        [
            {"role": "system", "content": huge},
            {"role": "user", "content": "hi"},
        ]
    )
    assert "hi" in prompt
    assert "parameters" not in prompt


def test_ensure_state_creates_key(tmp_path: Path) -> None:
    key = router.ensure_state()
    assert len(key) > 20
    assert router.KEY_PATH.exists()
    assert router.KEY_PATH.stat().st_mode & 0o777 == 0o600
