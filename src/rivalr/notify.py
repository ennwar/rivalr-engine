"""Telegram notifications for the scheduled runner - DEDICATED rivalr bot.

Config lives in rivalr-engine/.env (gitignored), with rivalr-specific
variable names so the trading bot's credentials can never be picked up
by accident:

    RIVALR_TELEGRAM_BOT_TOKEN=...
    RIVALR_TELEGRAM_CHAT_ID=...

Policy:
  - ONLY these two keys, ONLY from this repo's .env. No fallback to
    os.environ, no fallback to other variable names or files.
  - require_config() raises loudly when they're missing - the snapshot
    runner calls it at startup and screams (alert file + ERROR log) but
    still writes the ledger, which outranks everything.
  - telegram_send() itself never raises mid-pipeline.

Test after configuring:  uv run python -m rivalr.notify --test
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

log = logging.getLogger("rivalr.notify")

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
API = "https://api.telegram.org/bot{token}/{method}"

TOKEN_KEY = "RIVALR_TELEGRAM_BOT_TOKEN"
CHAT_KEY = "RIVALR_TELEGRAM_CHAT_ID"


def _load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """.env file first; RIVALR_-prefixed process env fills gaps (Railway
    injects config that way). The prefix rule means no other project's
    credentials can ever be picked up."""
    import os

    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    for k, v in os.environ.items():
        if k.startswith("RIVALR_") and not env.get(k):
            env[k] = v
    return env


def require_config(path: Path = ENV_FILE) -> tuple[str, str]:
    """(token, chat_id) or a loud RuntimeError. Never falls back to any
    other variable name, file, or the process environment."""
    env = _load_env(path)
    token = env.get(TOKEN_KEY, "")
    chat_id = env.get(CHAT_KEY, "")
    missing = [k for k, v in ((TOKEN_KEY, token), (CHAT_KEY, chat_id)) if not v]
    if missing:
        raise RuntimeError(
            f"rivalr telegram not configured: {', '.join(missing)} missing "
            f"from {path} - notifications will NOT be sent. Create the bot "
            f"via BotFather, /start it, and put the values in {path.name}."
        )
    return token, chat_id


def telegram_send(text: str) -> bool:
    """Best-effort Telegram message via the rivalr bot. Never raises."""
    try:
        token, chat_id = require_config()
    except RuntimeError as exc:
        log.error("%s", exc)
        return False
    try:
        resp = requests.post(
            API.format(token=token, method="sendMessage"),
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        )
        ok = resp.status_code == 200 and resp.json().get("ok", False)
        if not ok:
            log.warning("telegram send failed: %s %s",
                        resp.status_code, resp.text[:200])
        return ok
    except Exception as exc:
        log.warning("telegram send failed: %r", exc)
        return False


def main() -> int:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    require_config()  # loud failure by design
    text = "rivalr bot test: dedicated bot is live."
    if len(sys.argv) > 1 and sys.argv[1] != "--test":
        text = " ".join(sys.argv[1:])
    ok = telegram_send(text)
    print("delivered:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
