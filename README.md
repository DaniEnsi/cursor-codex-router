# cursor-codex-router

OpenAI-compatible local proxy that lets **OpenAI Codex** (CLI / Desktop) use **Cursor Agent** models.

Codex talks to `http://127.0.0.1:18789/v1`. Each completion is fulfilled by:

```bash
agent --print --mode ask --output-format stream-json --stream-partial-output \
  --model <id> --trust --workspace <tmp> -p <prompt>
```

Codex owns tools and sandbox; this router only returns assistant text.

## Requirements

- Python 3.10+
- [Cursor Agent CLI](https://cursor.com) (`agent` on your `PATH`, or set `CURSOR_AGENT_BIN`)
- OpenAI Codex CLI or Desktop
- Linux recommended (systemd user service for background). macOS works via pidfile mode.

## Install

```bash
# recommended
uv tool install git+https://github.com/DaniEnsi/cursor-codex-router.git

# or with pip
pip install git+https://github.com/DaniEnsi/cursor-codex-router.git

# or from a clone
uv tool install -e .
# pip install -e .
```

Then run setup (starts the router in the background):

```bash
cursor-codex-router setup
```

This will:

1. Create `~/.local/share/cursor-codex-router/` and a local API key
2. Sync the grouped model catalog from `agent models`
3. Write Codex provider config + auth (`~/.codex/config.toml`, `~/.codex/auth.json`)
4. Install a systemd user unit (when available) and **start the router in the background**

Then **fully quit and reopen Codex** so it reloads `model_catalog_json`.

### Useful flags

```bash
cursor-codex-router setup --skip-codex-config   # don't touch ~/.codex
cursor-codex-router setup --no-systemd          # background via pidfile instead
cursor-codex-router setup --force-codex-config  # rewrite managed provider block
```

## Commands

| Command | Description |
|---------|-------------|
| `setup` | Install, configure, sync catalog, start in background |
| `start` | Start in background (`--foreground` / `-f` for foreground) |
| `stop` | Stop background router |
| `restart` | Restart |
| `status` | Health, systemd/pidfile, paths |
| `sync` | Rebuild model catalog from `agent models` |
| `serve` | Foreground HTTP server (used by systemd) |
| `uninstall` | Stop + remove unit (`--purge-state` deletes state dir) |

```bash
cursor-codex-router status
cursor-codex-router sync && cursor-codex-router restart
```

## Codex config

`setup` writes something like:

```toml
model_provider = "cursor"
model = "auto"
model_catalog_json = "~/.local/share/cursor-codex-router/model_catalog.json"

[model_providers.cursor]
name = "Cursor (local router)"
base_url = "http://127.0.0.1:18789/v1"
wire_api = "responses"
requires_openai_auth = true
```

Auth uses the generated key in `~/.local/share/cursor-codex-router/api_key` (also written into `~/.codex/auth.json`).

## API

| Method | Path | Auth |
|--------|------|------|
| GET | `/healthz` | no |
| GET | `/v1/models` | Bearer / `x-api-key` |
| POST | `/v1/chat/completions` | yes (SSE supported) |
| POST | `/v1/responses` | yes (SSE supported) |
| POST | `/v1/completions` | yes |

Bind defaults to **localhost only** (`127.0.0.1:18789`). Do not expose this on a public interface — it shells out to your Cursor agent.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `CURSOR_CODEX_ROUTER_HOST` | `127.0.0.1` | Bind host |
| `CURSOR_CODEX_ROUTER_PORT` | `18789` | Bind port |
| `CURSOR_CODEX_ROUTER_STATE` | `~/.local/share/cursor-codex-router` | State dir |
| `CURSOR_CODEX_ROUTER_WORKSPACE` | `/tmp/cursor-codex-router-ws` | Agent workspace |
| `CURSOR_CODEX_ROUTER_DEFAULT_MODEL` | `auto` | Fallback model |
| `CURSOR_CODEX_ROUTER_TIMEOUT` | `600` | Agent timeout (seconds) |
| `CURSOR_CODEX_ROUTER_MAX_CONCURRENT` | `3` | Concurrent agent runs |
| `CURSOR_AGENT_BIN` | `~/.local/bin/agent` | Agent binary |

## Model catalog

Cursor exposes many effort-suffixed ids (`…-high`, `…-fast`, …). The sync step groups them into base slugs with `supported_reasoning_levels` so Codex’s `/model` picker shows reasoning + Fast selectors instead of a flat list of variants.

```bash
cursor-codex-router sync
cursor-codex-router restart
# then fully quit & reopen Codex
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Security

- State files (`api_key`, logs) stay under your home directory and are gitignored.
- Never commit `api_key`, `chatgpt_auth.json`, or response dumps.
- Keep the router on loopback unless you know what you are doing.

## License

MIT
