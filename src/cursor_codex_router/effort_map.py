"""Catalog → runtime EffortMap seam.

Writer (catalog.build) and reader (router.resolve_model) share one typed
module. File + in-memory adapters sit at the seam so tests skip disk.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max", "ultra", "minimal")


@dataclass(frozen=True)
class EffortSlot:
    normal: str | None = None
    fast: str | None = None

    def pick(self, wants_fast: bool) -> str | None:
        if wants_fast and self.fast:
            return self.fast
        return self.normal or self.fast

    def to_dict(self) -> dict[str, str | None]:
        return {"normal": self.normal, "fast": self.fast}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EffortSlot:
        if not isinstance(data, dict):
            return cls()
        normal = data.get("normal")
        fast = data.get("fast")
        return cls(
            normal=normal if isinstance(normal, str) else None,
            fast=fast if isinstance(fast, str) else None,
        )


@dataclass(frozen=True)
class ModelEfforts:
    base: str
    thinking: bool
    default_effort: str
    has_fast: bool
    efforts: dict[str, EffortSlot]

    def pick_agent(self, effort: str | None, wants_fast: bool) -> str | None:
        """Pick a concrete agent id for this catalog slug."""
        if not self.efforts:
            return None

        eff = (effort or self.default_effort or "high").lower()
        if eff == "minimal" and "low" in self.efforts:
            eff = "low"

        slot = self.efforts.get(eff)
        if not slot:
            if eff in _EFFORT_ORDER:
                i = _EFFORT_ORDER.index(eff)
                for j in range(len(_EFFORT_ORDER)):
                    for cand in (
                        _EFFORT_ORDER[min(len(_EFFORT_ORDER) - 1, i + j)],
                        _EFFORT_ORDER[max(0, i - j)],
                    ):
                        if cand in self.efforts:
                            slot = self.efforts[cand]
                            break
                    if slot:
                        break
            if not slot:
                slot = next(iter(self.efforts.values()))

        return slot.pick(wants_fast)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thinking": self.thinking,
            "base": self.base,
            "default_effort": self.default_effort,
            "has_fast": self.has_fast,
            "efforts": {ce: slot.to_dict() for ce, slot in self.efforts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, slug: str = "") -> ModelEfforts:
        raw_efforts = data.get("efforts") if isinstance(data, dict) else None
        efforts: dict[str, EffortSlot] = {}
        if isinstance(raw_efforts, dict):
            for ce, slot in raw_efforts.items():
                if isinstance(ce, str):
                    efforts[ce] = EffortSlot.from_dict(slot if isinstance(slot, dict) else None)

        base = data.get("base") if isinstance(data, dict) else None
        if not isinstance(base, str) or not base:
            base = slug[: -len("-thinking")] if slug.endswith("-thinking") else (slug or "")

        thinking = (
            bool(data.get("thinking")) if isinstance(data, dict) else slug.endswith("-thinking")
        )
        default = data.get("default_effort") if isinstance(data, dict) else None
        if not isinstance(default, str) or not default:
            default = "high"
        has_fast = bool(data.get("has_fast")) if isinstance(data, dict) else False
        return cls(
            base=base,
            thinking=thinking,
            default_effort=default,
            has_fast=has_fast,
            efforts=efforts,
        )


class EffortMap:
    """Deep module: catalog slug → agent ids by Codex effort / fast."""

    def __init__(self, entries: dict[str, ModelEfforts] | None = None) -> None:
        self._entries = dict(entries or {})

    def __contains__(self, slug: str) -> bool:
        return slug in self._entries

    def __bool__(self) -> bool:
        return bool(self._entries)

    def get(self, slug: str) -> ModelEfforts | None:
        return self._entries.get(slug)

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    def put(self, slug: str, info: ModelEfforts) -> None:
        self._entries[slug] = info

    def pick_agent(self, slug: str, effort: str | None, wants_fast: bool) -> str | None:
        info = self.get(slug)
        if not info:
            return None
        return info.pick_agent(effort, wants_fast)

    def to_dict(self) -> dict[str, Any]:
        return {slug: info.to_dict() for slug, info in self._entries.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EffortMap:
        if not isinstance(data, dict):
            return cls()
        entries: dict[str, ModelEfforts] = {}
        for slug, raw in data.items():
            if isinstance(slug, str) and isinstance(raw, dict):
                entries[slug] = ModelEfforts.from_dict(raw, slug=slug)
        return cls(entries)


class MemoryEffortMapStore:
    """In-memory adapter — second adapter that makes the seam real."""

    def __init__(self, emap: EffortMap | None = None) -> None:
        self._emap = emap or EffortMap()

    def load(self, force: bool = False) -> EffortMap:
        return self._emap

    def save(self, emap: EffortMap) -> None:
        self._emap = emap


class FileEffortMapStore:
    """JSON file adapter used in production."""

    def __init__(self, path: Path, ttl: float = 30.0) -> None:
        self.path = path
        self.ttl = ttl
        self._cache: EffortMap | None = None
        self._ts = 0.0

    def load(self, force: bool = False) -> EffortMap:
        now = time.time()
        if not force and self._cache is not None and now - self._ts < self.ttl:
            return self._cache
        data: dict[str, Any] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}
        emap = EffortMap.from_dict(data)
        self._cache = emap
        self._ts = now
        return emap

    def save(self, emap: EffortMap) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(emap.to_dict(), indent=2) + "\n")
        self._cache = emap
        self._ts = time.time()
