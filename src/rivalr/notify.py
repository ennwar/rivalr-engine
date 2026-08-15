"""Telegram notifications for the scheduled runner.

Token/chat id live in rivalr-engine/.env (gitignored), same variable
names as the trading-platform bot:

    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...

telegram_send() never raises and never blocks a snapshot: missing
config or network failure logs a warning and returns False.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

log = logging.getLogger("rivalr.notify")

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
API = "https://api.telegram.org/bot{token}/{method}"


def _load_env(path: Path = ENV_FILE) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def telegram_send(text: str) -> bool:
    """Best-effort Telegram message. Returns True on confirmed delivery."""
    env = _load_env()
    token = env.get("TELEGRAM_BOT_TOKEN")
    chat_id = env.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("telegram not configured (.env missing TELEGRAM_BOT_TOKEN"
                    "/TELEGRAM_CHAT_ID) - notification skipped")
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
