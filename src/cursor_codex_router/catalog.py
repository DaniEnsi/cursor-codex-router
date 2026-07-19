#!/usr/bin/env python3
"""Build a Codex model_catalog_json from `agent models`.

Groups Cursor's effort-suffixed ids into base models with
supported_reasoning_levels so Codex /model shows reasoning pickers
instead of listing every low/medium/high variant separately.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import paths as P

# Bump when catalog shape changes so Codex Desktop/app-server drops stale cache.
CATALOG_ETAG_PREFIX = "cursor-local-catalog-grouped-v8"


def _out() -> Path:
    return P.catalog_path()


def _map_out() -> Path:
    return P.effort_map_path()


def _agent() -> str:
    return P.agent_bin()


def _cache() -> Path:
    return P.codex_models_cache_path()


# Cursor effort tokens (longest first)
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

# Codex-supported effort labels we expose in the catalog
CODEX_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")

# Cursor effort -> Codex effort shown in /model
TO_CODEX = {
    "none": "low",          # Codex has no 'none'; closest
    "low": "low",
    "medium": "medium",
    "high": "high",
    "extra-high": "xhigh",
    "xhigh": "xhigh",
    "max": "max",
    "ultra": "ultra",
}

EFFORT_DESC = {
    "minimal": "Fastest / minimal reasoning",
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth for everyday tasks",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra high reasoning depth for complex problems",
    "max": "Maximum reasoning depth for the hardest problems",
    "ultra": "Maximum reasoning with automatic task delegation",
}

# Preferred default effort when multiple exist
DEFAULT_PREFERENCE = ["high", "medium", "xhigh", "max", "low", "extra-high", "none", "ultra"]


def parse_agent_models(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or " - " not in line:
            continue
        if line.lower().startswith("available models"):
            continue
        mid, _, name = line.partition(" - ")
        mid = mid.strip()
        name = re.sub(r"\s*\((current|default)\)\s*$", "", name.strip(), flags=re.I).strip()
        if mid:
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
    m = re.match(r"^(?P<base>.+)-(?P<effort>none|low|medium|high|extra-high|xhigh|max|ultra)-thinking$", core)
    if m:
        return m.group("base"), True, m.group("effort"), fast
    m = re.match(r"^(?P<base>.+)-thinking$", core)
    if m:
        return m.group("base"), True, None, fast

    return core, False, None, fast


_EFFORT_WORD_RE = re.compile(
    r"\s+(None|Low|Medium|High|Extra[\s-]?High|XHigh|Max|Ultra|Minimal|Fast)\b",
    re.I,
)


def _strip_effort_words(name: str) -> str:
    name = _EFFORT_WORD_RE.sub("", name)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def _version_tuple(ver: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in re.split(r"[.\-_]", ver):
        if p.isdigit():
            parts.append(int(p))
        elif parts:
            break
    return tuple(parts) if parts else (0,)


def classify_base(base: str) -> tuple[str, tuple[int, ...], str]:
    """Return (family, version, kind) for pruning/naming.

    kind is a short product hint: gpt|codex|opus|sonnet|fable|gemini|composer|grok|other
    """
    if base == "auto":
        return "auto", (0,), "other"

    m = re.match(r"^gpt-(?P<ver>\d+(?:\.\d+)*)(?:-(?P<rest>.+))?$", base)
    if m:
        ver = _version_tuple(m.group("ver"))
        rest = m.group("rest") or ""
        if rest.startswith("codex"):
            return "gpt", ver, "codex"
        return "gpt", ver, "gpt"

    m = re.match(r"^claude-opus-(?P<ver>\d+(?:[.\-]\d+)*)$", base)
    if m:
        return "claude-opus", _version_tuple(m.group("ver").replace("-", ".")), "opus"

    m = re.match(r"^claude-(?P<ver>\d+(?:\.\d+)*)-opus$", base)
    if m:
        return "claude-opus", _version_tuple(m.group("ver")), "opus"

    m = re.match(r"^claude-sonnet-(?P<ver>\d+(?:[.\-]\d+)*)$", base)
    if m:
        return "claude-sonnet", _version_tuple(m.group("ver").replace("-", ".")), "sonnet"

    m = re.match(r"^claude-(?P<ver>\d+(?:\.\d+)*)-sonnet$", base)
    if m:
        return "claude-sonnet", _version_tuple(m.group("ver")), "sonnet"

    m = re.match(r"^claude-fable-(?P<ver>\d+(?:[.\-]\d+)*)$", base)
    if m:
        return "claude-fable", _version_tuple(m.group("ver").replace("-", ".")), "fable"

    m = re.match(r"^gemini-(?P<ver>\d+(?:\.\d+)*)-(?P<tier>flash|pro)$", base)
    if m:
        return f"gemini-{m.group('tier')}", _version_tuple(m.group("ver")), "gemini"

    m = re.match(r"^composer-(?P<ver>\d+(?:\.\d+)*)$", base)
    if m:
        return "composer", _version_tuple(m.group("ver")), "composer"

    m = re.match(r"^cursor-grok-(?P<ver>\d+(?:\.\d+)*)$", base)
    if m:
        return "cursor-grok", _version_tuple(m.group("ver")), "grok"

    m = re.match(r"^glm-(?P<ver>\d+(?:\.\d+)*)$", base)
    if m:
        return "glm", _version_tuple(m.group("ver")), "other"

    m = re.match(r"^kimi-(?P<rest>.+)$", base)
    if m:
        ver_m = re.search(r"(\d+(?:\.\d+)*)", m.group("rest"))
        ver = _version_tuple(ver_m.group(1)) if ver_m else (0,)
        return "kimi", ver, "other"

    return f"other:{base}", (0,), "other"


def _slug_display(base: str) -> str:
    """Build a branded display label from the catalog base slug (never bare versions)."""
    if base == "auto":
        return "Auto"

    m = re.match(r"^gpt-(?P<ver>\d+(?:\.\d+)*)(?:-(?P<rest>.+))?$", base)
    if m:
        ver = m.group("ver")
        rest = (m.group("rest") or "").strip("-")
        if rest.startswith("codex"):
            suffix = rest[len("codex") :].strip("-")
            # Avoid effort-banned word "Max" in display_name (use plain Codex N.N).
            label = f"Codex {ver}"
            if suffix and suffix != "max":
                label += " " + suffix.replace("-", " ").title()
            return label
        label = f"GPT-{ver}"
        if rest:
            label += " " + rest.replace("-", " ").title()
        return label

    m = re.match(r"^claude-opus-(?P<a>\d+)(?:[.\-](?P<b>\d+))?$", base)
    if m:
        ver = m.group("a") + (f".{m.group('b')}" if m.group("b") else "")
        return f"Opus {ver}"

    m = re.match(r"^claude-(?P<ver>\d+(?:\.\d+)*)-opus$", base)
    if m:
        return f"Opus {m.group('ver')}"

    m = re.match(r"^claude-sonnet-(?P<a>\d+)(?:[.\-](?P<b>\d+))?$", base)
    if m:
        ver = m.group("a") + (f".{m.group('b')}" if m.group("b") else "")
        return f"Sonnet {ver}"

    m = re.match(r"^claude-(?P<ver>\d+(?:\.\d+)*)-sonnet$", base)
    if m:
        return f"Sonnet {m.group('ver')}"

    m = re.match(r"^claude-fable-(?P<a>\d+)(?:[.\-](?P<b>\d+))?$", base)
    if m:
        ver = m.group("a") + (f".{m.group('b')}" if m.group("b") else "")
        return f"Fable {ver}"

    m = re.match(r"^gemini-(?P<ver>\d+(?:\.\d+)*)-(?P<tier>flash|pro)$", base)
    if m:
        return f"Gemini {m.group('ver')} {m.group('tier').title()}"

    m = re.match(r"^composer-(?P<ver>\d+(?:\.\d+)*)$", base)
    if m:
        return f"Composer {m.group('ver')}"

    m = re.match(r"^cursor-grok-(?P<ver>\d+(?:\.\d+)*)$", base)
    if m:
        return f"Cursor Grok {m.group('ver')}"

    m = re.match(r"^glm-(?P<ver>\d+(?:\.\d+)*)$", base)
    if m:
        return f"GLM {m.group('ver')}"

    m = re.match(r"^kimi-(?P<rest>.+)$", base)
    if m:
        rest = m.group("rest").replace("-", " ").title()
        return f"Kimi {rest}"

    # Generic humanize — still force common brand tokens
    name = base.replace("-", " ")
    name = re.sub(r"\bgpt\b", "GPT", name, flags=re.I)
    name = re.sub(r"\bclaude\b", "Claude", name, flags=re.I)
    name = re.sub(r"\bgrok\b", "Grok", name, flags=re.I)
    name = re.sub(r"\bgemini\b", "Gemini", name, flags=re.I)
    name = re.sub(r"\bcodex\b", "Codex", name, flags=re.I)
    name = re.sub(r"\bcomposer\b", "Composer", name, flags=re.I)
    return name.title() if name == name.lower() else name


def _strip_display_markers(name: str) -> str:
    """Remove Thinking / NO ZDR markers from picker labels (kept in slug/effort map)."""
    name = re.sub(r"\s+Thinking\b", "", name, flags=re.I)
    name = re.sub(r"\(?\s*NO\s+ZDR\s*\)?", "", name, flags=re.I)
    name = re.sub(r"\s{2,}", " ", name).strip(" -")
    return name.strip()


def nice_display(base: str, thinking: bool, sample_name: str) -> str:
    """Branded display_name for Codex picker. Never bare versions like '5.2'.

    Thinking twins keep the -thinking slug for effort mapping, but the picker
    label reads like the base name (no 'Thinking' / 'NO ZDR' suffixes).
    """
    del thinking  # slug carries thinking; display never labels it
    sample = _strip_effort_words(sample_name or "")
    sample = _strip_display_markers(sample)

    # Slug is source of truth for product prefix (GPT-/Codex/Gemini/…).
    name = _slug_display(base)

    # Pull useful non-effort tokens from agent sample (e.g. 1M context badge).
    if re.search(r"\b1M\b", sample, re.I) and not re.search(r"\b1M\b", name, re.I):
        name = f"{name} 1M"

    # If sample already has a richer branded label sharing our prefix, prefer its
    # non-effort wording (keeps agent nuances) but never accept a bare version.
    brand_prefixes = (
        "GPT-",
        "GPT ",
        "Codex ",
        "Opus ",
        "Sonnet ",
        "Fable ",
        "Gemini ",
        "Composer ",
        "Cursor Grok ",
        "GLM ",
        "Kimi ",
        "Auto",
    )
    if sample and any(sample.startswith(p) or sample.upper().startswith(p.upper()) for p in brand_prefixes):
        # Keep sample when it already includes our brand and isn't effort-dirty.
        cand = _strip_effort_words(sample)
        cand = _strip_display_markers(cand)
        if cand and not re.match(r"^[\d.]+$", cand):
            # Prefer slug-built GPT-/Codex labels (agent sometimes drops brand).
            if name.startswith("GPT-") or name.startswith("Codex "):
                pass  # keep slug-built
            else:
                name = cand

    # Final guard: never ship effort/fast/Thinking/NO ZDR tokens; never bare versions.
    name = _strip_effort_words(name)
    name = _strip_display_markers(name)
    if re.match(r"^[\d.]+$", name) or not name:
        name = _slug_display(base)
    # Ensure GPT product models always keep the GPT- prefix (Codex UI otherwise
    # can look like a bare '5.2' next to branded Anthropic names).
    family, _ver, kind = classify_base(base)
    if family == "gpt" and kind == "gpt" and not name.upper().startswith("GPT"):
        name = _slug_display(base)
    if family == "gpt" and kind == "codex" and not name.upper().startswith("CODEX"):
        name = _slug_display(base)
    # Absolute GPT brand guarantee (title-case "Gpt …" / bare versions).
    if family == "gpt" and kind == "gpt":
        if not re.match(r"^GPT[\s\-]", name, re.I):
            name = _slug_display(base)
        elif not name.startswith("GPT"):
            # Normalize "gpt-5.5" / "Gpt 5.5" → slug-built "GPT-5.5 …"
            name = _slug_display(base)
    return name


def prune_slugs(slugs: list[str]) -> set[str]:
    """Drop obsolete model lines; keep newest (+ GPT previous generation).

    Rules:
    - auto: always keep
    - gpt* (OpenAI only): keep newest two minor versions of highest major
      (e.g. 5.5 + all 5.6* luna/sol/terra)
    - every other family (Claude Opus/Sonnet/Fable, Gemini, Composer, Grok,
      GLM, Kimi, …): keep ONLY the newest version line
    - if both `base` and `base-thinking` survive, drop `base` (keep thinking
      slug; display_name strips the Thinking label)
    - singleton / unknown: keep
    """
    # Map catalog slug -> (family, version, base_without_thinking)
    meta: dict[str, tuple[str, tuple[int, ...], str]] = {}
    for slug in slugs:
        thinking = slug.endswith("-thinking")
        base = slug[: -len("-thinking")] if thinking else slug
        family, ver, _kind = classify_base(base)
        meta[slug] = (family, ver, base)

    # Per-family versions present (by base, not thinking twin)
    family_versions: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for family, ver, _base in meta.values():
        family_versions[family].add(ver)

    keep_versions: dict[str, set[tuple[int, ...]]] = {}
    for family, versions in family_versions.items():
        if family == "auto" or family.startswith("other:"):
            keep_versions[family] = set(versions)
            continue
        ordered = sorted(versions, reverse=True)
        if family == "gpt":
            # Two generations only for OpenAI/GPT: newest + previous same-major.
            if not ordered:
                keep_versions[family] = set()
                continue
            newest = ordered[0]
            kept = {newest}
            for cand in ordered[1:]:
                if len(newest) >= 1 and len(cand) >= 1 and cand[0] == newest[0]:
                    kept.add(cand)
                    break
            # If no same-major previous, still allow one previous major line.
            if len(kept) == 1:
                for cand in ordered[1:]:
                    kept.add(cand)
                    break
            keep_versions[family] = kept
        else:
            # Hard prune: only the newest line per non-OpenAI family.
            keep_versions[family] = {ordered[0]} if ordered else set()

    kept: set[str] = set()
    for slug, (family, ver, _base) in meta.items():
        if ver in keep_versions.get(family, set()):
            kept.add(slug)

    # Prefer thinking twin when both exist for the same base.
    drop_non_thinking: set[str] = set()
    for slug in kept:
        if slug.endswith("-thinking"):
            continue
        twin = f"{slug}-thinking"
        if twin in kept:
            drop_non_thinking.add(slug)
    return kept - drop_non_thinking


def load_template() -> dict:
    base_instructions = (
        "You are Codex, a coding agent. Collaborate with the user until their goal is handled. "
        "Prefer precise, minimal changes. Use tools when needed. Report results clearly."
    )
    model_messages = {
        "instructions_template": (
            "You are Codex, a coding agent. Collaborate with the user until their goal is handled.\n\n"
            "{{ personality }}\n\n"
            "Prefer precise, minimal changes. Use tools when needed. Report results clearly."
        ),
        "instructions_variables": {
            "personality_default": "",
            "personality_friendly": "# Personality\nBe clear, warm, and collaborative.\n",
            "personality_pragmatic": "# Personality\nBe direct, pragmatic, and concise.\n",
        },
    }
    if _cache().exists():
        try:
            data = json.loads(_cache().read_text())
            for m in data.get("models") or []:
                # Prefer shorter instructions if our previous seed polluted cache
                bi = m.get("base_instructions") or ""
                if m.get("model_messages") and len(bi) > 500:
                    base_instructions = bi
                    model_messages = m["model_messages"]
                    break
        except Exception:
            pass
    return {
        "base_instructions": base_instructions,
        "model_messages": model_messages,
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "context_window": 272000,
        "max_context_window": 272000,
        "effective_context_window_percent": 90,
        "input_modalities": ["text", "image"],
        "supports_reasoning_summaries": True,
        # Codex only attaches a non-null reasoning payload for custom providers when this is true.
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": False,
        "supports_search_tool": False,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        # Match official OpenAI embedded catalog shape (Codex 0.144+)
        "default_service_tier": None,
        "service_tiers": [],
        "additional_speed_tiers": [],
        "availability_nux": None,
        "upgrade": None,
        "experimental_supported_tools": [],
        "include_skills_usage_instructions": True,
        "use_responses_lite": False,
    }


def build() -> tuple[list[dict], dict]:
    proc = subprocess.run([_agent(), "models"], text=True, capture_output=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    rows = parse_agent_models(text)
    if not rows:
        raise SystemExit("no models parsed from agent models")

    # group key: (catalog_slug, thinking)
    # catalog_slug = base or base+"-thinking"
    groups: dict[str, dict] = {}
    # effort map: catalog_slug -> {codex_effort -> {normal: id, fast: id|None}}
    effort_map: dict[str, dict] = {}

    for mid, name in rows:
        base, thinking, c_effort, fast = split_model(mid)
        slug = f"{base}-thinking" if thinking else base
        g = groups.setdefault(
            slug,
            {
                "base": base,
                "thinking": thinking,
                "sample_name": name,
                "variants": [],  # (mid, c_effort, fast, name)
                "has_fast": False,
            },
        )
        g["variants"].append((mid, c_effort, fast, name))
        if fast:
            g["has_fast"] = True
        # prefer non-effort-stripped names for display
        if not fast and (c_effort in (None, "high", "medium")):
            g["sample_name"] = name

    # Drop obsolete lines (keep newest; GPT keeps two generations).
    keep = prune_slugs(list(groups.keys()))
    pruned = sorted(set(groups) - keep)
    if pruned:
        print(f"pruned {len(pruned)} obsolete models: {', '.join(pruned)}")
    groups = {k: v for k, v in groups.items() if k in keep}

    template = load_template()
    models: list[dict] = []
    priority = 1

    # Stable ordering: auto, composer first, then alpha
    def sort_key(slug: str) -> tuple:
        pref = {"auto": 0, "composer-2.5": 1, "composer-2.5-thinking": 2}
        return (pref.get(slug, 50), slug)

    for slug in sorted(groups.keys(), key=sort_key):
        g = groups[slug]
        variants = g["variants"]

        # Build codex_effort -> agent ids
        by_codex: dict[str, dict[str, str | None]] = defaultdict(lambda: {"normal": None, "fast": None})
        bare_normal = None
        bare_fast = None
        cursor_efforts_present = set()

        for mid, c_effort, fast, _name in variants:
            if c_effort is None:
                if fast:
                    bare_fast = mid
                else:
                    bare_normal = mid
                continue
            cursor_efforts_present.add(c_effort)
            ce = TO_CODEX.get(c_effort)
            if not ce:
                continue
            slot = by_codex[ce]
            if fast:
                slot["fast"] = mid
            else:
                slot["normal"] = mid

        # If only bare ids exist (auto, gemini, composer), expose a single reasoning level
        if not by_codex and (bare_normal or bare_fast):
            # composer etc.
            default = "high"
            levels = [{"effort": default, "description": EFFORT_DESC[default]}]
            # Map high -> bare
            by_codex[default] = {"normal": bare_normal or bare_fast, "fast": bare_fast}
            # Also accept medium/low selecting same id so UI can still move? Better keep single level only.
        else:
            # If bare id exists alongside efforts, treat bare as medium (common for gpt-5.3-codex)
            if bare_normal or bare_fast:
                if "medium" not in by_codex:
                    by_codex["medium"] = {"normal": bare_normal, "fast": bare_fast}
                else:
                    if bare_normal and not by_codex["medium"]["normal"]:
                        by_codex["medium"]["normal"] = bare_normal
                    if bare_fast and not by_codex["medium"]["fast"]:
                        by_codex["medium"]["fast"] = bare_fast

            # Ensure each codex effort has at least a normal id (fall back to fast)
            for ce, slot in list(by_codex.items()):
                if not slot["normal"] and slot["fast"]:
                    slot["normal"] = slot["fast"]
                if not slot["normal"]:
                    del by_codex[ce]

            if not by_codex:
                # fallback: pick any variant
                mid = variants[0][0]
                by_codex["high"] = {"normal": mid, "fast": None}

            # order levels
            order = ["low", "medium", "high", "xhigh", "max", "ultra", "minimal"]
            levels = []
            for ce in order:
                if ce in by_codex:
                    levels.append({"effort": ce, "description": EFFORT_DESC.get(ce, ce)})
            # any leftovers
            for ce in by_codex:
                if ce not in {x["effort"] for x in levels}:
                    levels.append({"effort": ce, "description": EFFORT_DESC.get(ce, ce)})

        # default effort
        default = None
        for pref in ["high", "medium", "xhigh", "max", "low", "ultra", "minimal"]:
            if pref in by_codex:
                default = pref
                break
        if not default:
            default = levels[0]["effort"]

        display = nice_display(g["base"], g["thinking"], g["sample_name"])
        # thinking models: make clear in description
        if g["thinking"]:
            desc = f"Cursor agent `{g['base']}` (thinking). Reasoning level selects the Cursor effort variant."
        else:
            desc = f"Cursor agent `{g['base']}`. Reasoning level selects the Cursor effort variant."

        # Build entry: template first, then override catalog-facing fields so
        # additional_speed_tiers / service_tiers are never clobbered.
        entry = {
            **template,
            "slug": slug,
            "display_name": display,
            "description": desc,
            "default_reasoning_level": default,
            "supported_reasoning_levels": levels,
            "priority": priority,
            "default_service_tier": None,
            "additional_speed_tiers": [],
            "service_tiers": [],
        }
        # Official shape: both additional_speed_tiers=["fast"] and service_tiers
        # with id "priority" / name "Fast" (UI speed selector, not a separate model).
        if g["has_fast"]:
            entry["additional_speed_tiers"] = ["fast"]
            entry["service_tiers"] = [
                {
                    "id": "priority",
                    "name": "Fast",
                    "description": "1.5x speed, increased usage",
                }
            ]
        models.append(entry)
        effort_map[slug] = {
            "thinking": g["thinking"],
            "base": g["base"],
            "default_effort": default,
            "has_fast": g["has_fast"],
            "efforts": {
                ce: {"normal": slot["normal"], "fast": slot["fast"]}
                for ce, slot in by_codex.items()
            },
        }
        priority += 1

    return models, effort_map


def assert_clean_catalog(models: list[dict]) -> None:
    """Fail loudly if any entry would pollute the Codex model picker."""
    effort_words = re.compile(
        r"\b(None|Low|Medium|High|Extra[\s-]?High|XHigh|Max|Ultra|Minimal|Fast)\b",
        re.I,
    )
    suffix_re = re.compile(
        r"-(?:none|low|medium|high|extra-high|xhigh|max|ultra|minimal|fast)$"
    )
    bare_ver = re.compile(r"^[\d.]+$", re.I)
    thinking_label = re.compile(r"\bThinking\b", re.I)
    no_zdr_label = re.compile(r"NO\s+ZDR", re.I)
    errors: list[str] = []
    slugs_present = {m.get("slug") or "" for m in models}
    for m in models:
        slug = m.get("slug") or ""
        name = m.get("display_name") or ""
        if slug != "gpt-5.1-codex-max" and suffix_re.search(slug):
            errors.append(f"dirty slug: {slug}")
        if effort_words.search(name):
            errors.append(f"dirty display_name: {slug!r} -> {name!r}")
        if thinking_label.search(name):
            errors.append(f"Thinking label in display_name: {slug!r} -> {name!r}")
        if no_zdr_label.search(name):
            errors.append(f"NO ZDR in display_name: {slug!r} -> {name!r}")
        if bare_ver.match(name.strip()):
            errors.append(f"bare version display_name: {slug!r} -> {name!r}")
        # GPT product models must keep GPT- prefix (not bare / not Codex-only).
        base = slug[: -len("-thinking")] if slug.endswith("-thinking") else slug
        family, _ver, kind = classify_base(base)
        if family == "gpt" and kind == "gpt" and not re.match(r"^GPT[\s\-]", name):
            errors.append(f"missing GPT brand: {slug!r} -> {name!r}")
        if family == "gpt" and kind == "codex" and not name.upper().startswith("CODEX"):
            errors.append(f"missing Codex brand: {slug!r} -> {name!r}")
        # Non-thinking must not coexist with its thinking twin.
        if not slug.endswith("-thinking") and f"{slug}-thinking" in slugs_present:
            errors.append(f"non-thinking twin should be pruned: {slug}")
        if not m.get("supported_reasoning_levels"):
            errors.append(f"missing supported_reasoning_levels: {slug}")
    if errors:
        raise SystemExit("catalog validation failed:\n  " + "\n  ".join(errors[:20]))


def write_models_cache(models: list[dict], fetched_at: str) -> str:
    """Refresh ~/.codex/models_cache.json with a new etag so app-server picks it up."""
    payload_for_hash = json.dumps(
        [{"slug": m["slug"], "display_name": m["display_name"],
          "levels": m.get("supported_reasoning_levels"),
          "tiers": m.get("service_tiers"),
          "speed": m.get("additional_speed_tiers")} for m in models],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload_for_hash.encode()).hexdigest()[:12]
    etag = f"{CATALOG_ETAG_PREFIX}-{digest}"
    # Normalize fetched_at to Zulu like Codex cache
    ts = fetched_at.replace("+00:00", "Z")
    if ts.endswith("Z") is False and "+" not in ts:
        ts = ts + "Z" if "T" in ts else ts
    # Match installed CLI so OnlineIfUncached accepts this cache instead of
    # refetching chatgpt.com and clobbering the local grouped catalog.
    client_version = "0.144.3"
    try:
        proc = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=10)
        m = re.search(r"(\d+\.\d+\.\d+)", (proc.stdout or "") + (proc.stderr or ""))
        if m:
            client_version = m.group(1)
    except Exception:
        pass
    cache = {
        "fetched_at": ts if ts.endswith("Z") else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "etag": etag,
        "client_version": client_version,
        "models": models,
    }
    _cache().parent.mkdir(parents=True, exist_ok=True)
    _cache().write_text(json.dumps(cache, separators=(",", ":")) + "\n")
    return etag


def main() -> int:
    models, effort_map = build()
    assert_clean_catalog(models)
    fetched_at = datetime.now(timezone.utc).isoformat()
    catalog = {
        "fetched_at": fetched_at,
        "source": "cursor-agent-grouped",
        "models": models,
    }
    _out().parent.mkdir(parents=True, exist_ok=True)
    _out().write_text(json.dumps(catalog, indent=2) + "\n")
    _map_out().write_text(json.dumps(effort_map, indent=2) + "\n")
    etag = write_models_cache(models, fetched_at)
    print(f"wrote {_out()} ({len(models)} base models)")
    print(f"wrote {_map_out()}")
    print(f"wrote {_cache()} etag={etag}")
    # summary
    multi = [m for m in models if len(m["supported_reasoning_levels"]) > 1]
    print(f"multi-effort models: {len(multi)}")
    for m in models:
        eff = ",".join(x["effort"] for x in m["supported_reasoning_levels"])
        print(
            f"  {m['slug']}: {m['display_name']!r} [{eff}] "
            f"default={m['default_reasoning_level']} fast={m.get('additional_speed_tiers')}"
        )
    return 0


def sync() -> int:
    """CLI-friendly alias for main()."""
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
