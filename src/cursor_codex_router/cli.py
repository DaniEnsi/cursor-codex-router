"""CLI for cursor-codex-router: setup, start/stop, status, sync."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__
from . import paths as P

UNIT_NAME = "cursor-codex-router.service"

SYSTEMD_UNIT = textwrap.dedent(
    """\
    [Unit]
    Description=Cursor→Codex OpenAI-compatible local router
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    Environment=HOME=%h
    Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
    Environment=CURSOR_CODEX_ROUTER_HOST=127.0.0.1
    Environment=CURSOR_CODEX_ROUTER_PORT=18789
    Environment=CURSOR_CODEX_ROUTER_DEFAULT_MODEL=auto
    WorkingDirectory=%h
    ExecStart={exec_start}
    Restart=on-failure
    RestartSec=3
    StandardOutput=append:{service_log}
    StandardError=append:{service_log}

    [Install]
    WantedBy=default.target
    """
)

CODEX_PROVIDER_SNIPPET = textwrap.dedent(
    """\
    # --- cursor-codex-router (managed) ---
    # Refresh catalog: cursor-codex-router sync && cursor-codex-router restart
    # Then fully quit & reopen Codex (model_catalog_json is startup-only).

    model_provider = "cursor"
    model = "auto"
    model_reasoning_effort = "high"
    model_reasoning_summary = "none"
    model_supports_reasoning_summaries = true
    model_catalog_json = "{catalog}"
    forced_login_method = "api"
    cli_auth_credentials_store = "file"

    [model_providers.cursor]
    name = "Cursor (local router)"
    base_url = "{base_url}"
    wire_api = "responses"
    requires_openai_auth = true
    request_max_retries = 2
    stream_max_retries = 3
    stream_idle_timeout_ms = 600000
    # --- end cursor-codex-router ---
    """
)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def _have_systemd() -> bool:
    if not _which("systemctl"):
        return False
    try:
        r = _run(["systemctl", "--user", "is-system-running"], check=False, capture=True)
        # "running", "degraded", "offline" etc. — presence of user bus matters more
        return r.returncode in (0, 1) or bool(r.stdout or r.stderr)
    except Exception:
        return False


def _resolve_exec_start() -> str:
    """Prefer installed console script, else python -m."""
    exe = _which("cursor-codex-router")
    if exe:
        return f"{exe} serve"
    return f"{sys.executable} -m cursor_codex_router serve"


def cmd_serve(_args: argparse.Namespace) -> int:
    from .router import main as router_main

    router_main()
    return 0


def cmd_sync(_args: argparse.Namespace) -> int:
    from .catalog import sync

    return sync()


def _ensure_api_key() -> str:
    from .router import ensure_state

    return ensure_state()


def _write_codex_auth(api_key: str) -> None:
    path = P.codex_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    # Codex file auth: OPENAI_API_KEY style for custom providers
    data["OPENAI_API_KEY"] = api_key
    # Some Codex builds also read tokens.access_token / api_key
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    tokens = dict(tokens)
    tokens.setdefault("account_id", "cursor-local")
    data["tokens"] = tokens
    if "api_key" not in data:
        data["api_key"] = api_key
    path.write_text(json.dumps(data, indent=2) + "\n")
    path.chmod(0o600)


def _configure_codex(api_key: str, *, merge: bool = True) -> None:
    """Write/update ~/.codex/config.toml provider block and auth."""
    _write_codex_auth(api_key)
    cfg = P.codex_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    snippet = CODEX_PROVIDER_SNIPPET.format(
        catalog=str(P.catalog_path()),
        base_url=P.base_url(),
    )
    if not cfg.exists():
        cfg.write_text(snippet)
        print(f"wrote {cfg}")
        return

    text = cfg.read_text()
    marker_start = "# --- cursor-codex-router (managed) ---"
    marker_end = "# --- end cursor-codex-router ---"
    if marker_start in text and marker_end in text:
        before, rest = text.split(marker_start, 1)
        _, after = rest.split(marker_end, 1)
        text = before.rstrip() + "\n\n" + snippet + after.lstrip("\n")
        cfg.write_text(text if text.endswith("\n") else text + "\n")
        print(f"updated managed block in {cfg}")
        return

    if merge and 'model_providers.cursor' in text:
        print(
            f"note: {cfg} already has [model_providers.cursor]; "
            "leaving it alone. Re-run with --force-codex-config to replace."
        )
        return

    # Append managed block; set top-level provider if missing
    if "model_provider" not in text:
        cfg.write_text(snippet + "\n" + text)
    else:
        cfg.write_text(text.rstrip() + "\n\n" + snippet)
    print(f"appended provider config to {cfg}")


def _install_systemd() -> None:
    unit_dir = P.systemd_user_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = P.systemd_unit_path()
    body = SYSTEMD_UNIT.format(
        exec_start=_resolve_exec_start(),
        service_log=str(P.service_log_path()),
    )
    # Expand %h in Environment PATH is handled by systemd; ExecStart must be absolute.
    # Replace %h in our template only for paths we control in append logs — systemd
    # expands %h in unit files, so leave %h as-is.
    unit.write_text(body)
    print(f"wrote {unit}")
    _run(["systemctl", "--user", "daemon-reload"], check=False)
    _run(["systemctl", "--user", "enable", UNIT_NAME], check=False)


def _pidfile_alive() -> int | None:
    path = P.pid_path()
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        path.unlink(missing_ok=True)
        return None
    return pid


def _health() -> dict | None:
    url = f"http://{P.host()}:{P.port()}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _start_systemd() -> int:
    r = _run(["systemctl", "--user", "start", UNIT_NAME], check=False, capture=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        print(f"systemd start failed: {err}", file=sys.stderr)
        return 1
    return _wait_healthy()


def _start_pidfile() -> int:
    if _pidfile_alive():
        print("already running (pidfile)")
        return 0
    P.state_dir().mkdir(parents=True, exist_ok=True)
    P.workspace_dir().mkdir(parents=True, exist_ok=True)
    log = P.service_log_path()
    cmd = _resolve_exec_start().split()
    # Detach
    with log.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=lf,
            start_new_session=True,
            cwd=str(Path.home()),
            env=os.environ.copy(),
        )
    P.pid_path().write_text(str(proc.pid) + "\n")
    return _wait_healthy()


def _wait_healthy(timeout: float = 8.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health():
            print(f"router healthy at {P.base_url()}")
            return 0
        time.sleep(0.25)
    print("router did not become healthy in time", file=sys.stderr)
    return 1


def _stop_systemd() -> int:
    _run(["systemctl", "--user", "stop", UNIT_NAME], check=False)
    return 0


def _stop_pidfile() -> int:
    pid = _pidfile_alive()
    if not pid:
        print("not running")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"kill failed: {e}", file=sys.stderr)
        return 1
    for _ in range(40):
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except OSError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    P.pid_path().unlink(missing_ok=True)
    print("stopped")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    _ensure_api_key()
    if args.foreground:
        return cmd_serve(args)
    if _have_systemd() and P.systemd_unit_path().exists() and not args.no_systemd:
        return _start_systemd()
    return _start_pidfile()


def cmd_stop(args: argparse.Namespace) -> int:
    if _have_systemd() and P.systemd_unit_path().exists() and not args.no_systemd:
        return _stop_systemd()
    return _stop_pidfile()


def cmd_restart(args: argparse.Namespace) -> int:
    cmd_stop(args)
    time.sleep(0.5)
    return cmd_start(args)


def cmd_status(_args: argparse.Namespace) -> int:
    print(f"cursor-codex-router {__version__}")
    print(f"  base_url:   {P.base_url()}")
    print(f"  state_dir:  {P.state_dir()}")
    print(f"  agent_bin:  {P.agent_bin()}")
    agent = Path(P.agent_bin())
    print(f"  agent_ok:   {agent.is_file() and os.access(agent, os.X_OK)}")
    print(f"  api_key:    {'yes' if P.api_key_path().exists() else 'missing'}")
    print(f"  catalog:    {'yes' if P.catalog_path().exists() else 'missing'}")
    print(f"  effort_map: {'yes' if P.effort_map_path().exists() else 'missing'}")

    health = _health()
    if health:
        print(f"  health:     ok  ({health.get('service')})")
    else:
        print("  health:     down")

    if _have_systemd() and P.systemd_unit_path().exists():
        r = _run(
            ["systemctl", "--user", "is-active", UNIT_NAME],
            check=False,
            capture=True,
        )
        print(f"  systemd:    {(r.stdout or '').strip() or 'unknown'}")
    else:
        pid = _pidfile_alive()
        print(f"  pidfile:    {pid if pid else 'none'}")

    if P.log_path().exists():
        print(f"  log:        {P.log_path()}")
    return 0 if health else 1


def cmd_setup(args: argparse.Namespace) -> int:
    print(f"setting up cursor-codex-router {__version__}")

    agent = Path(P.agent_bin())
    if not agent.is_file():
        # try PATH
        found = _which("agent")
        if found:
            os.environ["CURSOR_AGENT_BIN"] = found
            agent = Path(found)
        else:
            print(
                "error: Cursor agent CLI not found.\n"
                "  Install Cursor CLI / `agent`, or set CURSOR_AGENT_BIN.",
                file=sys.stderr,
            )
            return 1
    print(f"  agent: {agent}")

    api_key = _ensure_api_key()
    print(f"  state: {P.state_dir()}")
    print(f"  api_key: {P.api_key_path()} (chmod 600)")

    # Sync catalog
    print("  syncing model catalog…")
    from .catalog import sync

    rc = sync()
    if rc != 0:
        print("catalog sync failed", file=sys.stderr)
        return rc

    if not args.skip_codex_config:
        _configure_codex(api_key, merge=not args.force_codex_config)
    else:
        print("  skipped Codex config (--skip-codex-config)")

    use_systemd = _have_systemd() and not args.no_systemd
    if use_systemd:
        _install_systemd()
        print("  starting systemd user service…")
        rc = _start_systemd()
    else:
        print("  starting background process (pidfile)…")
        rc = _start_pidfile()

    if rc != 0:
        return rc

    print()
    print("setup complete.")
    print(f"  OpenAI base URL: {P.base_url()}")
    print(f"  API key file:    {P.api_key_path()}")
    print("  Commands: cursor-codex-router status | stop | restart | sync")
    print("  Fully quit & reopen Codex so it reloads model_catalog_json.")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    if _have_systemd() and P.systemd_unit_path().exists():
        _run(["systemctl", "--user", "stop", UNIT_NAME], check=False)
        _run(["systemctl", "--user", "disable", UNIT_NAME], check=False)
        P.systemd_unit_path().unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"], check=False)
        print(f"removed {P.systemd_unit_path()}")
    else:
        _stop_pidfile()

    if args.purge_state:
        if P.state_dir().exists():
            shutil.rmtree(P.state_dir())
            print(f"removed {P.state_dir()}")
    else:
        print(f"left state dir intact: {P.state_dir()} (use --purge-state to delete)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cursor-codex-router",
        description="OpenAI-compatible local router that fronts the Cursor agent CLI for Codex.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Install, configure Codex, sync catalog, start in background")
    setup.add_argument("--skip-codex-config", action="store_true", help="Do not write ~/.codex/config.toml")
    setup.add_argument(
        "--force-codex-config",
        action="store_true",
        help="Replace existing [model_providers.cursor] with managed block",
    )
    setup.add_argument("--no-systemd", action="store_true", help="Use pidfile instead of systemd")
    setup.set_defaults(func=cmd_setup)

    start = sub.add_parser("start", help="Start router in background")
    start.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    start.add_argument("--no-systemd", action="store_true")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="Stop background router")
    stop.add_argument("--no-systemd", action="store_true")
    stop.set_defaults(func=cmd_stop)

    restart = sub.add_parser("restart", help="Restart background router")
    restart.add_argument("--no-systemd", action="store_true")
    restart.set_defaults(func=cmd_restart)

    status = sub.add_parser("status", help="Show router status")
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync", help="Rebuild model catalog from `agent models`")
    sync.set_defaults(func=cmd_sync)

    serve = sub.add_parser("serve", help="Run HTTP server in foreground (systemd ExecStart)")
    serve.set_defaults(func=cmd_serve)

    uninstall = sub.add_parser("uninstall", help="Stop service and remove systemd unit")
    uninstall.add_argument("--purge-state", action="store_true", help="Also delete state directory")
    uninstall.set_defaults(func=cmd_uninstall)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
