"""Shared paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def state_dir() -> Path:
    return Path(
        env(
            "CURSOR_CODEX_ROUTER_STATE",
            str(Path.home() / ".local/share/cursor-codex-router"),
        )
    )


def workspace_dir() -> Path:
    return Path(
        env(
            "CURSOR_CODEX_ROUTER_WORKSPACE",
            "/tmp/cursor-codex-router-ws",
        )
    )


def agent_bin() -> str:
    return env("CURSOR_AGENT_BIN", str(Path.home() / ".local/bin/agent"))


def host() -> str:
    return env("CURSOR_CODEX_ROUTER_HOST", "127.0.0.1")


def port() -> int:
    return int(env("CURSOR_CODEX_ROUTER_PORT", "18789"))


def default_model() -> str:
    return env("CURSOR_CODEX_ROUTER_DEFAULT_MODEL", "auto")


def agent_timeout() -> int:
    return int(env("CURSOR_CODEX_ROUTER_TIMEOUT", "600"))


def max_prompt_chars() -> int:
    return int(env("CURSOR_CODEX_ROUTER_MAX_PROMPT", "200000"))


def models_cache_ttl() -> int:
    return int(env("CURSOR_CODEX_ROUTER_MODELS_TTL", "300"))


def max_concurrent() -> int:
    return int(env("CURSOR_CODEX_ROUTER_MAX_CONCURRENT", "3"))


def api_key_path() -> Path:
    return state_dir() / "api_key"


def log_path() -> Path:
    return state_dir() / "router.log"


def service_log_path() -> Path:
    return state_dir() / "service.log"


def pid_path() -> Path:
    return state_dir() / "router.pid"


def catalog_path() -> Path:
    return state_dir() / "model_catalog.json"


def effort_map_path() -> Path:
    return state_dir() / "model_effort_map.json"


def codex_dir() -> Path:
    return Path.home() / ".codex"


def codex_config_path() -> Path:
    return codex_dir() / "config.toml"


def codex_auth_path() -> Path:
    return codex_dir() / "auth.json"


def codex_models_cache_path() -> Path:
    return codex_dir() / "models_cache.json"


def systemd_user_dir() -> Path:
    return Path.home() / ".config/systemd/user"


def systemd_unit_path() -> Path:
    return systemd_user_dir() / "cursor-codex-router.service"


def base_url() -> str:
    return f"http://{host()}:{port()}/v1"
