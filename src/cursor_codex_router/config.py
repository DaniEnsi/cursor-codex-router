"""Live router Config — env, paths, and runtime knobs behind one interface.

Callers and tests use get_config() / set_config(); nothing freezes at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(environ: dict[str, str], name: str, default: str) -> str:
    return environ.get(name, default)


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _falsey(raw: str) -> bool:
    return raw.strip().lower() in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    state_dir: Path
    workspace: Path
    agent_bin: str
    default_model: str
    agent_timeout: int
    max_prompt_chars: int
    models_cache_ttl: int
    max_concurrent: int
    nested_agent: bool
    tool_bridge: bool
    home: Path = Path.home()

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Config:
        env = environ if environ is not None else os.environ
        home = Path.home()
        return cls(
            host=_env(env, "CURSOR_CODEX_ROUTER_HOST", "127.0.0.1"),
            port=int(_env(env, "CURSOR_CODEX_ROUTER_PORT", "18789")),
            state_dir=Path(
                _env(
                    env,
                    "CURSOR_CODEX_ROUTER_STATE",
                    str(home / ".local/share/cursor-codex-router"),
                )
            ),
            workspace=Path(
                _env(env, "CURSOR_CODEX_ROUTER_WORKSPACE", "/tmp/cursor-codex-router-ws")
            ),
            agent_bin=_env(env, "CURSOR_AGENT_BIN", str(home / ".local/bin/agent")),
            default_model=_env(env, "CURSOR_CODEX_ROUTER_DEFAULT_MODEL", "auto"),
            agent_timeout=int(_env(env, "CURSOR_CODEX_ROUTER_TIMEOUT", "600")),
            max_prompt_chars=int(_env(env, "CURSOR_CODEX_ROUTER_MAX_PROMPT", "200000")),
            models_cache_ttl=int(_env(env, "CURSOR_CODEX_ROUTER_MODELS_TTL", "300")),
            max_concurrent=int(_env(env, "CURSOR_CODEX_ROUTER_MAX_CONCURRENT", "3")),
            nested_agent=_truthy(_env(env, "CURSOR_CODEX_ROUTER_NESTED_AGENT", "")),
            tool_bridge=not _falsey(_env(env, "CURSOR_CODEX_ROUTER_TOOL_BRIDGE", "1")),
            home=home,
        )

    @property
    def api_key_path(self) -> Path:
        return self.state_dir / "api_key"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "router.log"

    @property
    def service_log_path(self) -> Path:
        return self.state_dir / "service.log"

    @property
    def pid_path(self) -> Path:
        return self.state_dir / "router.pid"

    @property
    def catalog_path(self) -> Path:
        return self.state_dir / "model_catalog.json"

    @property
    def effort_map_path(self) -> Path:
        return self.state_dir / "model_effort_map.json"

    @property
    def codex_dir(self) -> Path:
        return self.home / ".codex"

    @property
    def codex_config_path(self) -> Path:
        return self.codex_dir / "config.toml"

    @property
    def codex_auth_path(self) -> Path:
        return self.codex_dir / "auth.json"

    @property
    def codex_models_cache_path(self) -> Path:
        return self.codex_dir / "models_cache.json"

    @property
    def systemd_user_dir(self) -> Path:
        return self.home / ".config/systemd/user"

    @property
    def systemd_unit_path(self) -> Path:
        return self.systemd_user_dir / "cursor-codex-router.service"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(cfg: Config | None) -> None:
    """Inject Config for tests, or pass None to reload from the environment."""
    global _config
    _config = cfg
