"""Seam tests for catalog → runtime EffortMap."""

from __future__ import annotations

from pathlib import Path

from cursor_codex_router.effort_map import (
    EffortMap,
    EffortSlot,
    FileEffortMapStore,
    MemoryEffortMapStore,
    ModelEfforts,
)


def _sample_map() -> EffortMap:
    return EffortMap(
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


def test_pick_agent_respects_effort_and_fast() -> None:
    emap = _sample_map()
    assert emap.pick_agent("cursor-grok-4.5", "high", False) == "cursor-grok-4.5-high"
    assert emap.pick_agent("cursor-grok-4.5", "low", True) == "cursor-grok-4.5-low-fast"
    assert emap.pick_agent("missing", "high", False) is None


def test_pick_agent_defaults_and_nearest_effort() -> None:
    emap = _sample_map()
    assert emap.pick_agent("cursor-grok-4.5", None, False) == "cursor-grok-4.5-high"
    # xhigh missing → nearest available
    assert emap.pick_agent("cursor-grok-4.5", "xhigh", False) == "cursor-grok-4.5-high"
    # minimal remaps to low when present
    assert emap.pick_agent("cursor-grok-4.5", "minimal", False) == "cursor-grok-4.5-low"


def test_dict_round_trip_preserves_schema() -> None:
    emap = _sample_map()
    restored = EffortMap.from_dict(emap.to_dict())
    assert restored.to_dict() == emap.to_dict()
    assert "cursor-grok-4.5" in restored
    assert restored.get("cursor-grok-4.5") is not None


def test_file_and_memory_adapters(tmp_path: Path) -> None:
    emap = _sample_map()
    mem = MemoryEffortMapStore()
    mem.save(emap)
    assert mem.load().pick_agent("cursor-grok-4.5", "high", False) == "cursor-grok-4.5-high"

    path = tmp_path / "model_effort_map.json"
    store = FileEffortMapStore(path)
    store.save(emap)
    assert path.exists()
    loaded = store.load(force=True)
    assert loaded.pick_agent("cursor-grok-4.5", "low", True) == "cursor-grok-4.5-low-fast"
