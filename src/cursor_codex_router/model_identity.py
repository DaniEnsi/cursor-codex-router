"""Shared model-id vocabulary for catalog sync and runtime resolve.

One seam for peeling Cursor agent ids into catalog slug + effort + fast,
and for parsing `agent models` output. Catalog write and router resolve
both call this module — no duplicated effort tables or twin parsers.
"""

from __future__ import annotations

import re

# Cursor effort tokens (longest first) — used when peeling suffixes.
CURSOR_EFFORTS = (
    "extra-high",
    "xhigh",
    "ultra",
    "max",
    "high",
    "medium",
    "low",
    "none",
)

# Cursor effort token -> Codex reasoning.effort label
TO_CODEX = {
    "none": "low",  # Codex has no 'none'; closest
    "low": "low",
    "medium": "medium",
    "high": "high",
    "extra-high": "xhigh",
    "xhigh": "xhigh",
    "max": "max",
    "ultra": "ultra",
}

_LEGACY_THINKING_RE = re.compile(
    r"^(?P<base>.+)-(?P<effort>" + "|".join(CURSOR_EFFORTS) + r")-thinking$"
)
_MODEL_ID_RE = re.compile(r"^([A-Za-z0-9._:-]+)\s*(?:[-—]\s*(.*))?$")
_MARKER_RE = re.compile(r"\s*\((current|default)\)\s*$", re.I)


def _clean_display_name(name: str) -> str:
    return _MARKER_RE.sub("", name.strip()).strip()


def parse_agent_models(text: str) -> list[tuple[str, str]]:
    """Parse `agent models` stdout into (id, display_name) rows."""
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("available models"):
            continue
        if " - " in line:
            mid, _, name = line.partition(" - ")
            mid = mid.strip()
            name = _clean_display_name(name)
            if mid:
                rows.append((mid, name or mid))
            continue
        m = _MODEL_ID_RE.match(line)
        if not m:
            continue
        mid = m.group(1)
        name = _clean_display_name(m.group(2) or mid)
        rows.append((mid, name or mid))
    return rows


def split_model(mid: str) -> tuple[str, bool, str | None, bool]:
    """Return (base_slug, thinking, cursor_effort|None, fast)."""
    fast = mid.endswith("-fast")
    core = mid[:-5] if fast else mid

    if "-thinking-" in core:
        base, effort = core.split("-thinking-", 1)
        return base, True, effort or None, fast

    for e in CURSOR_EFFORTS:
        suf = "-" + e
        if core.endswith(suf):
            return core[: -len(suf)], False, e, fast

    # Odd legacy names like claude-4.5-opus-high-thinking (thinking as suffix word)
    m = _LEGACY_THINKING_RE.match(core)
    if m:
        return m.group("base"), True, m.group("effort"), fast
    if core.endswith("-thinking"):
        return core[: -len("-thinking")], True, None, fast

    return core, False, None, fast


def peel_agent_model(model: str) -> tuple[str, str | None, bool]:
    """Split a concrete agent id into (catalog_slug, cursor_effort|None, fast).

    Thinking variants keep a `-thinking` catalog slug; non-thinking peel to base.
    Derived from :func:`split_model` so sync and resolve share one vocabulary.
    """
    base, thinking, effort, fast = split_model(model)
    slug = f"{base}-thinking" if thinking else base
    return slug, effort, fast
