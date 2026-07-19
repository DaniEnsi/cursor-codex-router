"""CLI parser smoke tests."""

from __future__ import annotations

from cursor_codex_router.cli import build_parser


def test_parser_commands() -> None:
    p = build_parser()
    for cmd in ("setup", "start", "stop", "restart", "status", "sync", "serve", "uninstall"):
        args = p.parse_args([cmd])
        assert args.command == cmd
