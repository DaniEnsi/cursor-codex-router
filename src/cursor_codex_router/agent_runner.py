"""Deep agent-runner module: spawn Cursor `agent`, stream text, cancel, concurrency.

HTTP Handler talks to this seam only (model, prompt, deltas, cancel). File/process
details stay behind AgentRunner; FakeAgentRunner is the second adapter for tests.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentRunnerConfig:
    agent_bin: str
    workspace: Path
    timeout: float
    max_prompt_chars: int
    max_concurrent: int


def _assistant_delta_text(obj: dict[str, Any]) -> str:
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in (None, "text", "output_text"):
                parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return ""


def _kill_proc(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=3)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


class AgentRunner:
    """Production adapter: subprocess `agent --print --mode ask …`."""

    def __init__(
        self,
        config: AgentRunnerConfig,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.config = config
        self._popen = popen
        self._log = log or (lambda *_a, **_k: None)
        self._slots = threading.BoundedSemaphore(config.max_concurrent)

    def run(
        self,
        model: str,
        prompt: str,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        if len(prompt) > cfg.max_prompt_chars:
            prompt = prompt[: cfg.max_prompt_chars] + "\n\n[truncated]"

        if not self._slots.acquire(blocking=False):
            raise RuntimeError(f"router busy: {cfg.max_concurrent} agent runs already in flight")

        cmd = [
            cfg.agent_bin,
            "--print",
            "--output-format",
            "stream-json",
            "--stream-partial-output",
            "--mode",
            "ask",
            "--model",
            model,
            "--trust",
            "--workspace",
            str(cfg.workspace),
            "-p",
            prompt,
        ]
        t0 = time.time()
        self._log(
            "agent_start",
            model=model,
            prompt_chars=len(prompt),
            workspace=str(cfg.workspace),
        )
        proc: Any | None = None
        text_parts: list[str] = []
        final_text = ""
        usage: dict[str, Any] = {}
        session_id = None
        request_id = None
        stderr_chunks: list[str] = []

        try:
            proc = self._popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
                cwd=str(cfg.workspace),
                start_new_session=True,
            )
            assert proc.stdout is not None
            assert proc.stderr is not None

            def _read_stderr() -> None:
                try:
                    while True:
                        chunk = proc.stderr.readline()
                        if not chunk:
                            break
                        stderr_chunks.append(chunk)
                except Exception:
                    pass

            err_thread = threading.Thread(target=_read_stderr, daemon=True)
            err_thread.start()

            deadline = t0 + cfg.timeout
            while True:
                if should_cancel and should_cancel():
                    _kill_proc(proc)
                    raise RuntimeError("client disconnected; agent cancelled")
                if time.time() > deadline:
                    _kill_proc(proc)
                    raise RuntimeError(f"agent timed out after {cfg.timeout}s")

                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue

                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue

                otype = obj.get("type")
                if otype == "assistant":
                    delta = _assistant_delta_text(obj)
                    if delta:
                        joined = "".join(text_parts)
                        if joined and delta == joined:
                            continue
                        if joined and delta.startswith(joined) and len(delta) > len(joined):
                            delta = delta[len(joined) :]
                        if not delta:
                            continue
                        text_parts.append(delta)
                        if on_text_delta:
                            on_text_delta(delta)
                elif otype == "result":
                    if obj.get("is_error"):
                        err = obj.get("result") or obj.get("error") or "agent returned is_error"
                        raise RuntimeError(str(err))
                    final_text = str(obj.get("result") or "")
                    usage = obj.get("usage") or {}
                    session_id = obj.get("session_id")
                    request_id = obj.get("request_id")

            err_thread.join(timeout=2)
            code = proc.wait(timeout=5)
            duration_ms = int((time.time() - t0) * 1000)
            stderr = "".join(stderr_chunks)

            if code != 0 and not final_text and not text_parts:
                self._log(
                    "agent_fail",
                    model=model,
                    code=code,
                    duration_ms=duration_ms,
                    stderr=stderr[-2000:],
                )
                raise RuntimeError(f"agent exited {code}: {(stderr)[-1500:]}")

            text = final_text if final_text else "".join(text_parts)
            self._log(
                "agent_ok",
                model=model,
                duration_ms=duration_ms,
                out_chars=len(text),
                usage=usage,
            )
            return {
                "text": text,
                "usage": usage,
                "session_id": session_id,
                "request_id": request_id,
                "duration_ms": duration_ms,
            }
        finally:
            if proc is not None and proc.poll() is None:
                _kill_proc(proc)
            self._slots.release()


class FakeAgentRunner:
    """In-memory adapter — second adapter that makes the agent seam real."""

    def __init__(
        self,
        handler: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._handler = handler or (lambda model, prompt, **_kw: {"text": "", "usage": {}})

    def run(
        self,
        model: str,
        prompt: str,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((model, prompt))
        if should_cancel and should_cancel():
            raise RuntimeError("client disconnected; agent cancelled")
        result = self._handler(
            model,
            prompt,
            on_text_delta=on_text_delta,
            should_cancel=should_cancel,
        )
        text = str(result.get("text") or "")
        if on_text_delta and text:
            on_text_delta(text)
        return result
