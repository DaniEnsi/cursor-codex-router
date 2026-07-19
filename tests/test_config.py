"""Seam tests for live Config (replaces import-time path freeze)."""

from __future__ import annotations

from pathlib import Path

from cursor_codex_router.config import Config, get_config, set_config


def test_from_env_reads_live_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_HOST", "10.0.0.2")
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_PORT", "9999")
    monkeypatch.setenv("CURSOR_AGENT_BIN", "/bin/agent-test")
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_DEFAULT_MODEL", "composer-2.5")
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_MAX_CONCURRENT", "7")
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_NESTED_AGENT", "1")
    monkeypatch.setenv("CURSOR_CODEX_ROUTER_TOOL_BRIDGE", "0")

    set_config(None)
    cfg = Config.from_env()
    assert cfg.host == "10.0.0.2"
    assert cfg.port == 9999
    assert cfg.state_dir == tmp_path / "state"
    assert cfg.workspace == tmp_path / "ws"
    assert cfg.agent_bin == "/bin/agent-test"
    assert cfg.default_model == "composer-2.5"
    assert cfg.max_concurrent == 7
    assert cfg.nested_agent is True
    assert cfg.tool_bridge is False
    assert cfg.api_key_path == tmp_path / "state" / "api_key"
    assert cfg.effort_map_path == tmp_path / "state" / "model_effort_map.json"
    assert cfg.base_url == "http://10.0.0.2:9999/v1"


def test_set_config_is_live_seam(tmp_path: Path) -> None:
    cfg = Config(
        host="127.0.0.1",
        port=1,
        state_dir=tmp_path / "s",
        workspace=tmp_path / "w",
        agent_bin="agent",
        default_model="auto",
        agent_timeout=10,
        max_prompt_chars=100,
        models_cache_ttl=30,
        max_concurrent=2,
        nested_agent=False,
        tool_bridge=True,
    )
    set_config(cfg)
    assert get_config() is cfg
    assert get_config().workspace == tmp_path / "w"
    set_config(None)
