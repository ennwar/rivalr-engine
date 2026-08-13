"""Understat client behind an interface, with disk cache and loud failures.

Understat moved its data out of embedded <script> blobs into JSON XHR
endpoints (verified 2026-08):

  GET /getLeagueData/EPL/{season}   {teams, players, dates}
  GET /getPlayerData/{player_id}    {player, matches, groups, shots, ...}

Both require an `X-Requested-With: XMLHttpRequest` header or they 404.
`season` is the start year (2025 = the 2025/26 season).

Cross-season windows: OpenFPL's rolling features look back over previous
gameweeks; at the start of a season those are last season's matches, so
team histories and player matches merge the previous season with the
current one, chronologically. Promoted teams simply have less history.

Failure policy: league-level failures raise after ERROR logging (the
caller decides how to degrade); a single player failing returns [] so one
missing player can't kill a batch run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("rivalr.understat")

BASE = "https://understat.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
LEAGUE_TTL = 6 * 3600
PLAYER_TTL = 24 * 3600
PREV_SEASON_TTL = 30 * 24 * 3600  # finished seasons don't change
POLITE_DELAY = 0.3  # seconds between live requests; cache hits are free

# FPL bootstrap team name -> Understat team title, where they differ.
FPL_TO_UNDERSTAT_TEAM = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Sheffield Utd": "Sheffield United",
    "West Brom": "West Bromwich Albion",
    "Brighton": "Brighton",
    "West Ham": "West Ham",
}


class Understat:
    def __init__(self, season: int, cache_dir: str | Path = "data/cache") -> None:
        """season: start year, e.g. 2026 for the 2026/27 season."""
        self.season = season
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.session.headers["X-Requested-With"] = "XMLHttpRequest"
        self._last_request = 0.0

    # -- fetch + cache -----------------------------------------------------

    def _get_json(self, path: str, key: str, ttl: int):
        cache_file = self.cache_dir / f"understat_{key}.json"
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        url = f"{BASE}/{path}"
        log.info("understat cache miss: %s", url)
        wait = POLITE_DELAY - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self.session.get(url, timeout=30)
            self._last_request = time.time()
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.error("UNDERSTAT FETCH FAILED: %s (%s)", url, exc)
            raise
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def league_data(self, season: int) -> dict:
        ttl = LEAGUE_TTL if season == self.season else PREV_SEASON_TTL
        return self._get_json(f"getLeagueData/EPL/{season}", f"epl_{season}", ttl)

    # -- datasets ----------------------------------------------------------

    @staticmethod
    def _teams_as_dict(raw) -> dict[str, list[dict]]:
        """{title: history}; tolerates the empty-season list shape."""
        if not raw:
            return {}
        values = raw.values() if isinstance(raw, dict) else raw
        return {t["title"]: t.get("history", []) for t in values}

    def teams_data(self) -> dict[str, list[dict]]:
        """{team title: match history, previous season + current, oldest
        first}. Raises if neither season is fetchable."""
        prev = current = None
        try:
            current = self._teams_as_dict(self.league_data(self.season)["teams"])
        except Exception:
            log.error("understat: current season %d league data unavailable", self.season)
        try:
            prev = self._teams_as_dict(self.league_data(self.season - 1)["teams"])
        except Exception:
            log.error("understat: previous season %d league data unavailable",
                      self.season - 1)
        if current is None and prev is None:
            raise RuntimeError("understat: no league data for either season")
        merged: dict[str, list[dict]] = {}
        for title, hist in (prev or {}).items():
            merged[title] = list(hist)
        for title, hist in (current or {}).items():
            merged.setdefault(title, []).extend(hist)
        if not any(merged.values()):
            raise RuntimeError("understat: league data present but empty")
        return merged

    def players_data(self) -> list[dict]:
        """Player index (id, player_name, team_title). Current season if it
        has players, else previous season (pre-season / GW1)."""
        for season in (self.season, self.season - 1):
            try:
                players = self.league_data(season).get("players", [])
            except Exception:
                continue
            if players:
                if season != self.season:
                    log.warning(
                        "understat: using season %d player index (season %d empty)",
                        season, self.season,
                    )
                return players
        raise RuntimeError("understat: no player index for either season")

    def player_matches(self, understat_id: int | str) -> list[dict]:
        """Per-match rows for one player over previous + current season,
        oldest first. Returns [] on failure so a batch run survives."""
        try:
            data = self._get_json(
                f"getPlayerData/{understat_id}", f"player_{understat_id}", PLAYER_TTL
            )
        except Exception:
            return []
        keep = {str(self.season), str(self.season - 1)}
        matches = [m for m in data.get("matches", []) if m.get("season") in keep]
        matches.sort(key=lambda m: m.get("date", ""))
        return matches

    # -- name matching -----------------------------------------------------

    @staticmethod
    def _norm(name: str) -> str:
        import unicodedata

        stripped = "".join(
            c for c in unicodedata.normalize("NFKD", name)
            if not unicodedata.combining(c)
        )
        return stripped.lower().strip()

    def map_fpl_players(self, elements: list[dict], teams: list[dict]) -> dict[int, str]:
        """FPL element id -> understat player id, by normalised name + team.

        Unmatched players are summarised at WARNING; their player-level
        Understat features will be NaN (handled downstream)."""
        import difflib

        upl = self.players_data()
        team_by_fpl_id = {t["id"]: t["name"] for t in teams}

        by_team: dict[str, dict[str, str]] = {}
        all_names: dict[str, str] = {}
        for p in upl:
            for title in p["team_title"].split(","):
                by_team.setdefault(title, {})[self._norm(p["player_name"])] = p["id"]
            all_names[self._norm(p["player_name"])] = p["id"]

        mapping: dict[int, str] = {}
        unmatched = 0
        for el in elements:
            fpl_team = team_by_fpl_id.get(el["team"], "")
            u_team = FPL_TO_UNDERSTAT_TEAM.get(fpl_team, fpl_team)
            candidates = by_team.get(u_team, all_names)
            full = self._norm(f"{el['first_name']} {el['second_name']}")
            web = self._norm(el["web_name"])

            hit = candidates.get(full) or all_names.get(full)
            if not hit:
                close = difflib.get_close_matches(full, list(candidates), n=1, cutoff=0.75)
                if not close:
                    close = difflib.get_close_matches(
                        web, list(candidates), n=1, cutoff=0.85
                    )
                if close:
                    hit = candidates[close[0]]
            if hit:
                mapping[int(el["id"])] = hit
            else:
                unmatched += 1
                log.debug("understat: no match for %s (%s)", full, fpl_team)
        if unmatched:
            log.warning(
                "understat: %d/%d FPL players unmatched - their player-level "
                "Understat features will be empty", unmatched, len(elements),
            )
        return mapping
