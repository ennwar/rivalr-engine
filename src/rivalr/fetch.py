"""FPL API client with a disk cache.

All endpoints are public and unauthenticated. Responses are cached on disk
under data/cache/ with per-endpoint TTLs; every cache miss is logged.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("rivalr.fetch")

BASE_URL = "https://fantasy.premierleague.com/api"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

HOUR = 3600

# TTL per endpoint pattern (seconds). First regex match wins; keep the more
# specific patterns first. Near a deadline the standings TTL is what matters:
# 1h means rival squads are at most one hour stale.
_TTL_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^bootstrap-static/$"), 6 * HOUR),
    (re.compile(r"^leagues-classic/\d+/standings/"), 1 * HOUR),
    (re.compile(r"^fixtures/"), 6 * HOUR),
    (re.compile(r"^element-summary/\d+/$"), 6 * HOUR),
    (re.compile(r"^entry/\d+/event/\d+/picks/$"), 1 * HOUR),
    (re.compile(r"^entry/\d+/transfers/$"), 1 * HOUR),
    (re.compile(r"^entry/\d+/history/$"), 1 * HOUR),
    (re.compile(r"^entry/\d+/$"), 1 * HOUR),
    (re.compile(r"^event/\d+/live/$"), 1 * HOUR),
]
_DEFAULT_TTL = 1 * HOUR


def _ttl_for(path: str) -> int:
    for pattern, ttl in _TTL_RULES:
        if pattern.match(path):
            return ttl
    return _DEFAULT_TTL


class FPLClient:
    """Thin FPL API client with disk cache and 429 backoff."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        session: requests.Session | None = None,
        max_retries: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.max_retries = max_retries
        # In-memory layer over the disk cache: bootstrap-static is ~10MB of
        # JSON and gets hit once per player during feature building.
        self._mem: dict[str, Any] = {}

    # -- cache -------------------------------------------------------------

    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        key = path.strip("/").replace("/", "_")
        if params:
            key += "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, cache_file: Path, ttl: int) -> Any | None:
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("cache corrupt, discarding: %s", cache_file.name)
            return None
        if time.time() - payload["fetched_at"] > ttl:
            return None
        return payload["data"]

    def _write_cache(self, cache_file: Path, data: Any) -> None:
        payload = {"fetched_at": time.time(), "data": data}
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

    # -- http --------------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None, force: bool = False) -> Any:
        """GET an API path (relative to /api/), serving from cache when fresh."""
        path = path.lstrip("/")
        ttl = _ttl_for(path)
        cache_file = self._cache_path(path, params)
        mem_key = str(cache_file)
        if not force:
            if mem_key in self._mem:
                return self._mem[mem_key]
            cached = self._read_cache(cache_file, ttl)
            if cached is not None:
                self._mem[mem_key] = cached
                return cached

        log.info("cache miss: %s params=%s (ttl=%ss)", path, params or {}, ttl)
        data = self._request(f"{BASE_URL}/{path}", params)
        self._write_cache(cache_file, data)
        self._mem[mem_key] = data
        return data

    def _request(self, url: str, params: dict[str, Any] | None) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                log.warning(
                    "429 from %s, backing off %.1fs (attempt %d/%d)",
                    url, wait, attempt, self.max_retries,
                )
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"still rate-limited after {self.max_retries} attempts: {url}")

    # -- endpoint helpers --------------------------------------------------

    def bootstrap(self) -> dict:
        return self.get("bootstrap-static/")

    def fixtures(self, event: int | None = None) -> list[dict]:
        params = {"event": event} if event else None
        return self.get("fixtures/", params=params)

    def entry(self, team_id: int) -> dict:
        return self.get(f"entry/{team_id}/")

    def entry_picks(self, team_id: int, gw: int) -> dict:
        return self.get(f"entry/{team_id}/event/{gw}/picks/")

    def entry_history(self, team_id: int) -> dict:
        return self.get(f"entry/{team_id}/history/")

    def entry_transfers(self, team_id: int) -> list[dict]:
        return self.get(f"entry/{team_id}/transfers/")

    def league_standings(self, league_id: int, page: int = 1) -> dict:
        return self.get(
            f"leagues-classic/{league_id}/standings/",
            params={"page_standings": page},
        )

    def element_summary(self, player_id: int) -> dict:
        return self.get(f"element-summary/{player_id}/")

    def event_live(self, gw: int) -> dict:
        return self.get(f"event/{gw}/live/")

    def cache_age(self, path: str = "bootstrap-static/") -> float | None:
        """Seconds since the cached copy of a path was fetched, or None."""
        cache_file = self._cache_path(path.lstrip("/"), None)
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return time.time() - payload["fetched_at"]
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    # -- convenience -------------------------------------------------------

    def current_gw(self) -> int:
        """The current (or next, in pre-season) gameweek id."""
        events = self.bootstrap()["events"]
        for ev in events:
            if ev["is_current"]:
                return ev["id"]
        for ev in events:
            if ev["is_next"]:
                return ev["id"]
        raise RuntimeError("no current or next gameweek in bootstrap-static")

    def next_gw(self) -> int:
        events = self.bootstrap()["events"]
        for ev in events:
            if ev["is_next"]:
                return ev["id"]
        raise RuntimeError("no next gameweek (season over?)")
