#!/usr/bin/env python3
"""OpenAI-compatible local router that fronts the Cursor agent CLI for Codex.

Codex talks to http://127.0.0.1:18789/v1 (chat.completions + responses).

Default (ask + tool bridge):
  agent --print --mode ask …  →  plain text and/or <<<CODEX_TOOL_CALL>>> blocks
  Router maps blocks to Responses API function_call items; Codex executes tools.

Nested escape hatch (CURSOR_CODEX_ROUTER_NESTED_AGENT=1):
  agent --print --force … in the project cwd; Cursor owns tools (bypasses Codex sandbox).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import paths as P
from .agent_runner import AgentRunner, AgentRunnerConfig, FakeAgentRunner
from .effort_map import EffortMap, FileEffortMapStore, MemoryEffortMapStore
from .model_identity import TO_CODEX as CURSOR_TO_CODEX_EFFORT
from .model_identity import parse_agent_models, peel_agent_model
from .tool_bridge import (
    backend_system,
    compact_tools,
    format_tools_for_prompt,
)

HOST = P.host()
PORT = P.port()
STATE_DIR = P.state_dir()
LOG_PATH = P.log_path()
KEY_PATH = P.api_key_path()
AGENT_BIN = P.agent_bin()
DEFAULT_MODEL = P.default_model()
AGENT_TIMEOUT = P.agent_timeout()
MAX_PROMPT_CHARS = P.max_prompt_chars()
MODELS_CACHE_TTL = P.models_cache_ttl()
MAX_CONCURRENT = P.max_concurrent()
WORKSPACE = P.workspace_dir()

_models_cache: dict[str, Any] = {"ts": 0.0, "ids": []}
_log_lock = threading.Lock()
_api_key_cache: str | None = None
_agent_runner: AgentRunner | FakeAgentRunner | None = None
# Back-compat alias for tests / older imports
BACKEND_SYSTEM = backend_system(nested=False, bridge=False, has_tools=False)

SKIP_INPUT_TYPES = {
    "function_call",
    "custom_tool_call",
    "tool_use",
    "tool_call",
    "reasoning",
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "image_generation_call",
    "mcp_call",
    "mcp_list_tools",
    "mcp_approval_request",
}


def ensure_state() -> str:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not KEY_PATH.exists():
        import secrets

        KEY_PATH.write_text(secrets.token_urlsafe(32) + "\n")
        KEY_PATH.chmod(0o600)
    return KEY_PATH.read_text().strip()


def get_api_key() -> str:
    global _api_key_cache
    if _api_key_cache is None:
        _api_key_cache = ensure_state()
    return _api_key_cache


def log(msg: str, **extra: Any) -> None:
    line = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "msg": msg,
        **extra,
    }
    text = json.dumps(line, ensure_ascii=False)
    with _log_lock:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def list_models(force: bool = False) -> list[str]:
    now = time.time()
    if not force and _models_cache["ids"] and now - _models_cache["ts"] < MODELS_CACHE_TTL:
        return list(_models_cache["ids"])
    ids: list[str] = []
    try:
        proc = subprocess.run(
            [AGENT_BIN, "models"],
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ids = [mid for mid, _ in parse_agent_models(out)]
    except Exception as e:
        log("models_error", error=str(e))
    if not ids:
        ids = [DEFAULT_MODEL]
    seen: set[str] = set()
    uniq: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    _models_cache["ts"] = now
    _models_cache["ids"] = uniq
    return uniq


def catalog_model_ids() -> list[str]:
    """Prefer grouped catalog slugs for /v1/models (what Codex /model uses)."""
    emap = load_effort_map()
    if emap:
        # stable-ish: auto/composer first then alpha
        pref = {"auto": 0, "composer-2.5": 1, "composer-2.5-thinking": 2}
        return sorted(emap.keys(), key=lambda s: (pref.get(s, 50), s))
    return list_models()


_effort_store: FileEffortMapStore | MemoryEffortMapStore | None = None


def set_effort_map_store(store: FileEffortMapStore | MemoryEffortMapStore | None) -> None:
    """Test seam: inject MemoryEffortMapStore or reset to default file store."""
    global _effort_store
    _effort_store = store


def _get_effort_store() -> FileEffortMapStore | MemoryEffortMapStore:
    global _effort_store
    if _effort_store is None:
        _effort_store = FileEffortMapStore(P.effort_map_path())
    return _effort_store


def load_effort_map(force: bool = False) -> EffortMap:
    return _get_effort_store().load(force=force)


def extract_reasoning_effort(body: dict[str, Any] | None) -> str | None:
    if not body:
        return None
    r = body.get("reasoning")
    if isinstance(r, dict):
        for k in ("effort", "reasoning_effort"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
    for k in ("reasoning_effort", "model_reasoning_effort"):
        v = body.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return None


def extract_wants_fast(body: dict[str, Any] | None) -> bool:
    if not body:
        return False
    st = body.get("service_tier")
    if isinstance(st, str) and st.lower() in {"priority", "fast"}:
        return True
    r = body.get("reasoning")
    if isinstance(r, dict):
        st = r.get("service_tier") or r.get("speed")
        if isinstance(st, str) and st.lower() in {"priority", "fast"}:
            return True
    return False


MODEL_SHORTS = {
    "grok": "cursor-grok-4.5",
    "grok-4.5": "cursor-grok-4.5",
    "composer": "composer-2.5",
    "composer-2": "composer-2.5",
}


def normalize_codex_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    eff = effort.strip().lower()
    if eff in {"default", "auto"}:
        return None
    if eff in {"extra-high", "extra_high"}:
        return "xhigh"
    if eff == "none":
        return "low"
    if eff == "minimal":
        return "minimal"
    return eff


def resolve_model(model: str | None, body: dict[str, Any] | None = None) -> tuple[str, str]:
    """Map Codex model + reasoning.effort (+fast) -> (echo_slug, agent_model_id).

    Important: echo_slug must stay a catalog base id (e.g. cursor-grok-4.5), never a
    concrete effort/fast variant. Echoing variants makes Codex rewrite the session
    model and breaks the reasoning picker.
    """
    raw = model
    if not model or model in ("default", "openai"):
        model = DEFAULT_MODEL
    model = str(model).split("/", 1)[-1].strip()

    known = list_models()
    known_set = set(known)
    emap = load_effort_map()

    body_effort = normalize_codex_effort(extract_reasoning_effort(body))
    body_fast = extract_wants_fast(body)

    peeled_slug, peeled_cursor_effort, peeled_fast = peel_agent_model(model)
    peeled_slug = MODEL_SHORTS.get(peeled_slug, peeled_slug)

    # Prefer catalog slug if this is already one; otherwise use peeled base.
    if model in emap:
        slug = model
        suffix_effort = None
        wants_fast = body_fast
    else:
        slug = peeled_slug if peeled_slug in emap else MODEL_SHORTS.get(model, model)
        if slug not in emap and peeled_slug in emap:
            slug = peeled_slug
        suffix_effort = (
            CURSOR_TO_CODEX_EFFORT.get(peeled_cursor_effort, peeled_cursor_effort)
            if peeled_cursor_effort
            else None
        )
        wants_fast = body_fast or peeled_fast

    # Body reasoning.effort always wins over any effort baked into the model id.
    effort = body_effort or suffix_effort
    if effort:
        effort = normalize_codex_effort(effort)

    info = emap.get(slug)
    if not info:
        # Unknown catalog slug: fall back to agent ids without rewriting echo.
        echo = slug if slug in emap else (peeled_slug if peeled_slug else model)
        if model in known_set:
            agent = model
            if wants_fast and not agent.endswith("-fast") and (agent + "-fast") in known_set:
                agent = agent + "-fast"
            log("model_alias", requested=raw, echo=echo, resolved=agent, effort=effort)
            return echo, agent
        hits = [k for k in known if k == slug or k.startswith(slug + "-")]
        if hits:

            def score(k: str) -> tuple:
                return (0 if k.endswith("-fast") else 1, 2 if "-high" in k else 0, -len(k))

            hits.sort(key=score, reverse=True)
            agent = hits[0]
            log("model_alias", requested=raw, echo=echo, resolved=agent, effort=effort)
            return echo, agent
        return echo, slug

    resolved = emap.pick_agent(slug, effort, wants_fast)
    eff = effort or info.default_effort or "high"
    if not resolved:
        resolved = slug if slug in known_set else DEFAULT_MODEL

    log(
        "model_resolve",
        requested=raw,
        echo=slug,
        slug=slug,
        effort=eff,
        wants_fast=wants_fast,
        resolved=resolved,
    )
    return slug, resolved


def normalize_model(model: str | None, body: dict[str, Any] | None = None) -> str:
    """Backward-compatible helper: returns the concrete agent model id only."""
    _echo, agent = resolve_model(model, body)
    return agent


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                itype = item.get("type")
                if itype in ("tool_use", "function_call", "tool_call"):
                    continue
                if itype in ("text", "input_text", "output_text"):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                elif itype in ("tool_result", "function_call_output", "input_image"):
                    # keep tool results / note images as text stubs
                    if itype == "input_image":
                        parts.append("[image]")
                    else:
                        parts.append(
                            content_to_text(
                                item.get("output") or item.get("content") or item.get("text")
                            )
                        )
                elif "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(content_to_text(item["content"]))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return content_to_text(content.get("text") or content.get("content") or "")
    return str(content)


def _looks_like_tool_schema_blob(text: str) -> bool:
    if len(text) < 2000:
        return False
    markers = (
        '"parameters"',
        '"type": "function"',
        '"type":"function"',
        "tool_choice",
        "JSON schema",
    )
    hits = sum(1 for m in markers if m in text)
    return hits >= 2


def messages_to_prompt(
    messages: list[dict[str, Any]],
    *,
    include_backend_system: bool = True,
    system_preamble: str | None = None,
) -> str:
    chunks: list[str] = []
    if include_backend_system:
        preamble = system_preamble if system_preamble is not None else BACKEND_SYSTEM
        chunks.append(f"### SYSTEM\n{preamble}")
    for msg in messages:
        role = str(msg.get("role") or "user").upper()
        if role in {"TOOL", "FUNCTION"}:
            role = "TOOL"
        text = content_to_text(msg.get("content"))
        if not text:
            if msg.get("name"):
                text = (
                    f"[name={msg.get('name')}] "
                    f"{content_to_text(msg.get('arguments') or msg.get('content'))}"
                )
            else:
                continue
        if role == "SYSTEM" and _looks_like_tool_schema_blob(text):
            # Codex sometimes embeds tool schemas in system; drop the blob.
            log("prompt_strip_tool_schema", chars=len(text))
            continue
        # Cap individual tool dumps
        if role == "TOOL" and len(text) > 20000:
            text = text[:20000] + "\n\n[tool output truncated]"
        chunks.append(f"### {role}\n{text}")
    return "\n\n".join(chunks)


def responses_input_to_prompt(
    body: dict[str, Any],
    *,
    nested: bool | None = None,
    bridge: bool | None = None,
    cwd: Path | None = None,
) -> str:
    nested = P.nested_agent_enabled() if nested is None else nested
    bridge = P.tool_bridge_enabled() if bridge is None else bridge
    messages: list[dict[str, Any]] = []
    # Prefer Codex instructions; avoid duplicating huge tool catalogs.
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        if not _looks_like_tool_schema_blob(instructions):
            messages.append({"role": "system", "content": instructions})
        else:
            log("prompt_strip_instructions_tools", chars=len(instructions))

    inp = body.get("input")
    skipped = 0
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type") or "")
            if itype in SKIP_INPUT_TYPES:
                skipped += 1
                continue
            if itype in {"function_call_output", "custom_tool_call_output", "tool_result"}:
                out = item.get("output") or item.get("content") or item.get("text") or ""
                name = item.get("name") or item.get("call_id") or "tool"
                messages.append({"role": "tool", "content": f"[{name}] {content_to_text(out)}"})
                continue
            role = item.get("role") or "user"
            if role in ("message", "input_text"):
                role = item.get("role") or "user"
            content = item.get("content")
            if content is None:
                content = item.get("text")
            messages.append({"role": str(role), "content": content})
    elif isinstance(inp, dict):
        messages.append({"role": "user", "content": inp})

    if isinstance(body.get("messages"), list):
        messages.extend(body["messages"])

    tools_raw = body.get("tools")
    has_tools = bool(tools_raw)
    compact = compact_tools(tools_raw) if has_tools and bridge and not nested else []
    if has_tools and not compact and not nested:
        log(
            "prompt_ignore_tools",
            tool_count=len(tools_raw) if isinstance(tools_raw, list) else 1,
            skipped_input=skipped,
            bridge=False,
        )
    elif compact:
        log(
            "prompt_bridge_tools",
            tool_count=len(compact),
            skipped_input=skipped,
        )
    elif skipped:
        log("prompt_skip_input_items", skipped=skipped)

    if cwd is not None:
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"Project working directory (Codex cwd): {cwd}",
            },
        )

    preamble = backend_system(
        nested=nested, bridge=bridge, has_tools=bool(compact) or (nested and has_tools)
    )
    if compact:
        preamble = f"{preamble}\n\n{format_tools_for_prompt(compact)}"

    return messages_to_prompt(messages, system_preamble=preamble)


def chat_messages_to_prompt(
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    *,
    nested: bool | None = None,
    bridge: bool | None = None,
    cwd: Path | None = None,
) -> str:
    nested = P.nested_agent_enabled() if nested is None else nested
    bridge = P.tool_bridge_enabled() if bridge is None else bridge
    tools_raw = body.get("tools")
    compact = compact_tools(tools_raw) if tools_raw and bridge and not nested else []
    if tools_raw and not compact and not nested:
        log(
            "prompt_ignore_tools",
            tool_count=len(tools_raw) if isinstance(tools_raw, list) else 1,
            api="chat",
            bridge=False,
        )
    elif compact:
        log("prompt_bridge_tools", tool_count=len(compact), api="chat")

    msgs = list(messages)
    if cwd is not None:
        msgs = [
            {"role": "system", "content": f"Project working directory (Codex cwd): {cwd}"}
        ] + msgs

    preamble = backend_system(nested=nested, bridge=bridge, has_tools=bool(compact))
    if compact:
        preamble = f"{preamble}\n\n{format_tools_for_prompt(compact)}"
    return messages_to_prompt(msgs, system_preamble=preamble)


def set_agent_runner(runner: AgentRunner | FakeAgentRunner | None) -> None:
    """Test seam: inject FakeAgentRunner or reset to default AgentRunner."""
    global _agent_runner
    _agent_runner = runner


def get_agent_runner() -> AgentRunner | FakeAgentRunner:
    global _agent_runner
    if _agent_runner is None:
        _agent_runner = AgentRunner(
            AgentRunnerConfig(
                agent_bin=AGENT_BIN,
                workspace=WORKSPACE,
                timeout=float(AGENT_TIMEOUT),
                max_prompt_chars=MAX_PROMPT_CHARS,
                max_concurrent=MAX_CONCURRENT,
            ),
            log=log,
        )
    return _agent_runner


def run_agent_stream(
    model: str,
    prompt: str,
    *,
    on_text_delta: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    return get_agent_runner().run(
        model,
        prompt,
        on_text_delta=on_text_delta,
        should_cancel=should_cancel,
    )


def run_agent(model: str, prompt: str) -> dict[str, Any]:
    return run_agent_stream(model, prompt)


def usage_to_openai(usage: dict[str, Any]) -> dict[str, int]:
    prompt = int(
        usage.get("inputTokens") or usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    )
    completion = int(
        usage.get("outputTokens")
        or usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def chat_completion_response(model: str, text: str, usage: dict[str, Any]) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage_to_openai(usage),
    }


def responses_api_response(model: str, text: str, usage: dict[str, Any]) -> dict:
    rid = f"resp_{uuid.uuid4().hex[:24]}"
    u = usage_to_openai(usage)
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex[:20]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": u["prompt_tokens"],
            "output_tokens": u["completion_tokens"],
            "total_tokens": u["total_tokens"],
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "cursor-codex-router/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        log("http", message=fmt % args, client=self.client_address[0])

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _auth_ok(self) -> bool:
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            if token == get_api_key():
                return True
        xkey = self.headers.get("x-api-key") or self.headers.get("X-Api-Key")
        if xkey and xkey.strip() == get_api_key():
            return True
        return False

    def _send(self, code: int, body: Any, content_type: str = "application/json") -> None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _send_err(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._send(
            code,
            {
                "error": {
                    "message": message,
                    "type": err_type,
                    "code": code,
                }
            },
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/healthz", "/health", "/"):
            self._send(
                200,
                {
                    "status": "ok",
                    "service": "cursor-codex-router",
                    "agent": AGENT_BIN,
                    "default_model": DEFAULT_MODEL,
                    "workspace": str(WORKSPACE),
                    "max_concurrent": MAX_CONCURRENT,
                },
            )
            return
        if path in ("/v1/models", "/models"):
            if not self._auth_ok():
                self._send_err(401, "Invalid API key")
                return
            models = catalog_model_ids()
            now = int(time.time())
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": m,
                            "object": "model",
                            "created": now,
                            "owned_by": "cursor",
                        }
                        for m in models
                    ],
                },
            )
            return
        self._send_err(404, f"Not found: {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if not self._auth_ok():
            self._send_err(401, "Invalid API key")
            return
        try:
            body = self._read_json()
        except Exception:
            self._send_err(400, "Invalid JSON body")
            return

        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat(body)
                return
            if path in ("/v1/responses", "/responses"):
                self._handle_responses(body)
                return
            if path in ("/v1/completions", "/completions"):
                prompt = body.get("prompt") or ""
                if isinstance(prompt, list):
                    prompt = "\n".join(str(p) for p in prompt)
                echo_model, agent_model = resolve_model(body.get("model"), body)
                result = run_agent(
                    agent_model,
                    messages_to_prompt([{"role": "user", "content": prompt}]),
                )
                self._send(
                    200,
                    {
                        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": echo_model,
                        "choices": [
                            {
                                "text": result["text"],
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": usage_to_openai(result["usage"]),
                    },
                )
                return
            self._send_err(404, f"Not found: {path}")
        except RuntimeError as e:
            msg = str(e)
            if "busy" in msg:
                self._send_err(429, msg, err_type="server_error")
            else:
                log("handler_error", error=msg, trace=traceback.format_exc()[-2000:])
                self._send_err(500, msg, err_type="server_error")
        except Exception as e:
            log("handler_error", error=str(e), trace=traceback.format_exc()[-2000:])
            self._send_err(500, str(e), err_type="server_error")

    def _client_gone(self) -> bool:
        # Best-effort: after headers are sent, failed writes set this.
        return bool(getattr(self, "_client_disconnected", False))

    def _handle_chat(self, body: dict[str, Any]) -> None:
        echo_model, agent_model = resolve_model(body.get("model"), body)
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            self._send_err(400, "messages is required")
            return
        # Ignore tools — Codex / clients execute them.
        if body.get("tools"):
            log(
                "prompt_ignore_tools",
                tool_count=len(body["tools"]) if isinstance(body["tools"], list) else 1,
                api="chat",
            )
        prompt = messages_to_prompt(messages)
        stream = bool(body.get("stream"))
        if not stream:
            result = run_agent(agent_model, prompt)
            self._send(
                200,
                chat_completion_response(echo_model, result["text"], result["usage"]),
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def emit_chunk(delta: dict, finish: str | None = None) -> None:
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": echo_model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except Exception:
                self._client_disconnected = True  # type: ignore[attr-defined]
                raise

        try:
            emit_chunk({"role": "assistant", "content": ""})

            def on_delta(delta: str) -> None:
                emit_chunk({"content": delta})

            result = run_agent_stream(
                agent_model,
                prompt,
                on_text_delta=on_delta,
                should_cancel=self._client_gone,
            )
            emit_chunk({}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            _ = result
        except Exception as e:
            if not self._client_gone():
                try:
                    err = {"error": {"message": str(e), "type": "server_error"}}
                    self.wfile.write(f"data: {json.dumps(err)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            log("chat_stream_error", error=str(e))

    def _handle_responses(self, body: dict[str, Any]) -> None:
        echo_model, agent_model = resolve_model(body.get("model"), body)
        try:
            log(
                "responses_req",
                keys=sorted(body.keys()),
                stream=bool(body.get("stream")),
                model=body.get("model"),
                echo_model=echo_model,
                agent_model=agent_model,
                has_tools=bool(body.get("tools")),
                tool_count=len(body["tools"]) if isinstance(body.get("tools"), list) else 0,
                input_type=type(body.get("input")).__name__,
                reasoning=body.get("reasoning"),
                service_tier=body.get("service_tier"),
            )
        except Exception:
            pass

        prompt = responses_input_to_prompt(body)
        stream = bool(body.get("stream"))

        if not stream:
            result = run_agent(agent_model, prompt)
            self._send(
                200,
                responses_api_response(echo_model, result["text"], result["usage"]),
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        msg_id = f"msg_{uuid.uuid4().hex[:20]}"
        rid = f"resp_{uuid.uuid4().hex[:24]}"
        seq = 0
        created_at = int(time.time())

        def emit(event_type: str, payload: dict) -> None:
            nonlocal seq
            seq += 1
            payload = {**payload, "type": event_type, "sequence_number": seq}
            try:
                self.wfile.write(f"event: {event_type}\n".encode())
                self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            except Exception:
                self._client_disconnected = True  # type: ignore[attr-defined]
                raise

        stub = {
            "id": rid,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": echo_model,
            "output": [],
        }
        try:
            emit("response.created", {"response": stub})
            emit("response.in_progress", {"response": stub})
            emit(
                "response.output_item.added",
                {
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
            emit(
                "response.content_part.added",
                {
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )

            def on_delta(delta: str) -> None:
                emit(
                    "response.output_text.delta",
                    {
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                    },
                )

            result = run_agent_stream(
                agent_model,
                prompt,
                on_text_delta=on_delta,
                should_cancel=self._client_gone,
            )
            text = result["text"]
            resp = responses_api_response(echo_model, text, result["usage"])
            resp["id"] = rid
            resp["output"][0]["id"] = msg_id

            emit(
                "response.output_text.done",
                {
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
            )
            emit(
                "response.content_part.done",
                {
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                },
            )
            emit(
                "response.output_item.done",
                {"output_index": 0, "item": resp["output"][0]},
            )
            emit("response.completed", {"response": resp})
        except Exception as e:
            log("responses_stream_error", error=str(e), trace=traceback.format_exc()[-1500:])
            if not self._client_gone():
                try:
                    emit(
                        "response.failed",
                        {
                            "response": {
                                **stub,
                                "status": "failed",
                                "error": {"message": str(e), "type": "server_error"},
                            }
                        },
                    )
                except Exception:
                    pass


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    log(
        "starting",
        host=HOST,
        port=PORT,
        agent=AGENT_BIN,
        default_model=DEFAULT_MODEL,
        workspace=str(WORKSPACE),
        max_concurrent=MAX_CONCURRENT,
    )
    try:
        ms = list_models(force=True)
        log("models_loaded", count=len(ms), sample=ms[:8])
        load_effort_map(force=True)
    except Exception as e:
        log("models_warm_fail", error=str(e))

    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"cursor-codex-router listening on http://{HOST}:{PORT}/v1 "
        f"(agent={AGENT_BIN}, workspace={WORKSPACE}, models={len(_models_cache['ids'])})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        log("stopped")


if __name__ == "__main__":
    main()
