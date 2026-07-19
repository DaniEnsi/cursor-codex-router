"""Seam tests for the agent runner module."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from cursor_codex_router.agent_runner import (
    AgentRunner,
    AgentRunnerConfig,
    FakeAgentRunner,
)


def _config(tmp_path: Path, **overrides: Any) -> AgentRunnerConfig:
    base = AgentRunnerConfig(
        agent_bin="agent",
        workspace=tmp_path / "ws",
        timeout=5.0,
        max_prompt_chars=1000,
        max_concurrent=2,
    )
    return AgentRunnerConfig(**{**base.__dict__, **overrides})


def test_fake_agent_runner_records_calls_and_returns() -> None:
    fake = FakeAgentRunner(
        handler=lambda model, prompt, **kw: {
            "text": f"echo:{model}:{prompt[:4]}",
            "usage": {"inputTokens": 1, "outputTokens": 2},
        }
    )
    deltas: list[str] = []
    result = fake.run(
        "cursor-grok-4.5-high",
        "hello world",
        on_text_delta=deltas.append,
    )
    assert result["text"] == "echo:cursor-grok-4.5-high:hell"
    assert deltas == ["echo:cursor-grok-4.5-high:hell"]
    assert fake.calls == [("cursor-grok-4.5-high", "hello world")]


def test_agent_runner_busy_when_slots_exhausted(tmp_path: Path) -> None:
    runner = AgentRunner(_config(tmp_path, max_concurrent=1))
    assert runner._slots.acquire(blocking=False)
    with pytest.raises(RuntimeError, match="router busy"):
        runner.run("auto", "hi")
    runner._slots.release()


def test_agent_runner_parses_stream_json(tmp_path: Path) -> None:
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hel"}]},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "lo"}]},
            }
        ),
        json.dumps(
            {
                "type": "result",
                "result": "Hello",
                "usage": {"inputTokens": 3, "outputTokens": 2},
                "session_id": "s1",
                "request_id": "r1",
            }
        ),
    ]

    class FakeStdout:
        def __init__(self, payload: str) -> None:
            self._lines = payload.splitlines(keepends=True)
            self._i = 0

        def readline(self) -> str:
            if self._i >= len(self._lines):
                return ""
            line = self._lines[self._i]
            self._i += 1
            return line

    class FakeProc:
        def __init__(self) -> None:
            self.pid = 4242
            self.stdout = FakeStdout("\n".join(lines) + "\n")
            self.stderr = io.StringIO("")
            self._code = 0

        def poll(self) -> int | None:
            # Exhausted stdout ⇒ process finished
            if self.stdout._i >= len(self.stdout._lines):
                return self._code
            return None

        def wait(self, timeout: float | None = None) -> int:
            return self._code

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    def fake_popen(*_a: Any, **_k: Any) -> FakeProc:
        return FakeProc()

    (tmp_path / "ws").mkdir()
    runner = AgentRunner(_config(tmp_path), popen=fake_popen)
    deltas: list[str] = []
    result = runner.run("auto", "hi", on_text_delta=deltas.append)
    assert result["text"] == "Hello"
    assert result["usage"]["inputTokens"] == 3
    assert result["session_id"] == "s1"
    assert "".join(deltas) == "Hello"


def test_agent_runner_truncates_long_prompt(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    class FakeStdout:
        def __init__(self) -> None:
            self._done = False
            self._payload = json.dumps({"type": "result", "result": "ok", "usage": {}}) + "\n"

        def readline(self) -> str:
            if self._done:
                return ""
            self._done = True
            return self._payload

    class FakeProc:
        def __init__(self) -> None:
            self.pid = 1
            self.stdout = FakeStdout()
            self.stderr = io.StringIO("")

        def poll(self) -> int | None:
            return 0 if self.stdout._done else None

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    def fake_popen(cmd: list[str], **_k: Any) -> FakeProc:
        seen["prompt"] = cmd[cmd.index("-p") + 1]
        return FakeProc()

    (tmp_path / "ws").mkdir()
    runner = AgentRunner(_config(tmp_path, max_prompt_chars=10), popen=fake_popen)
    runner.run("auto", "0123456789ABCDEF")
    assert seen["prompt"].startswith("0123456789")
    assert seen["prompt"].endswith("[truncated]")
