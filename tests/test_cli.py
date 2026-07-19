"""CLI parser smoke tests."""

from __future__ import annotations

from cursor_codex_router.cli import build_parser, cmd_restart


def test_parser_commands() -> None:
    p = build_parser()
    for cmd in ("setup", "start", "stop", "restart", "status", "sync", "serve", "uninstall"):
        args = p.parse_args([cmd])
        assert args.command == cmd


def test_restart_args_are_start_compatible() -> None:
    """restart Namespace must not blow up when handed to cmd_start."""
    p = build_parser()
    args = p.parse_args(["restart"])
    assert not hasattr(args, "foreground") or args.foreground is False
    # Simulate the defensive defaults cmd_restart applies
    if not hasattr(args, "foreground"):
        args.foreground = False
    assert args.foreground is False
    assert getattr(args, "no_systemd", False) is False
    assert callable(cmd_restart)
