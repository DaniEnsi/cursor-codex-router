"""Live path/env getters — thin adapters over :mod:`config`.

Kept for existing call sites (CLI, catalog). Prefer ``get_config()`` for new code.
"""

from __future__ import annotations

from pathlib import Path

from .config import get_config


def env(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def state_dir() -> Path:
    return get_config().state_dir


def workspace_dir() -> Path:
    return get_config().workspace


def agent_bin() -> str:
    return get_config().agent_bin


def host() -> str:
    return get_config().host


def port() -> int:
    return get_config().port


def default_model() -> str:
    return get_config().default_model


def agent_timeout() -> int:
    return get_config().agent_timeout


def max_prompt_chars() -> int:
    return get_config().max_prompt_chars


def models_cache_ttl() -> int:
    return get_config().models_cache_ttl


def max_concurrent() -> int:
    return get_config().max_concurrent


def nested_agent_enabled() -> bool:
    return get_config().nested_agent


def tool_bridge_enabled() -> bool:
    return get_config().tool_bridge


def api_key_path() -> Path:
    return get_config().api_key_path


def log_path() -> Path:
    return get_config().log_path


def service_log_path() -> Path:
    return get_config().service_log_path


def pid_path() -> Path:
    return get_config().pid_path


def catalog_path() -> Path:
    return get_config().catalog_path


def effort_map_path() -> Path:
    return get_config().effort_map_path


def codex_dir() -> Path:
    return get_config().codex_dir


def codex_config_path() -> Path:
    return get_config().codex_config_path


def codex_auth_path() -> Path:
    return get_config().codex_auth_path


def codex_models_cache_path() -> Path:
    return get_config().codex_models_cache_path


def systemd_user_dir() -> Path:
    return get_config().systemd_user_dir


def systemd_unit_path() -> Path:
    return get_config().systemd_unit_path


def base_url() -> str:
    return get_config().base_url
