"""Single source of truth for gameweek state.

FPL's flags mean DIFFERENT things and must not be conflated:

  events[].is_current    the gameweek whose deadline most recently passed
                         (it may still have fixtures to play!)
  events[].is_next       the gameweek you can currently make transfers for
  events[].finished      event-level: all fixtures done (lags full-time)
  events[].data_checked  points finalised - the only safe "complete"
  fixtures[].finished    UNRELIABLE mid-gameweek: verified 2026-08-31,
                         played fixtures kept finished=False for days
  fixtures[].started     flips at kickoff - the reliable "has been played
                         (at least begun)" signal

Rules enforced here:
  - next_gw() self-heals stale caches: if the cached is_next deadline has
    already passed, the bootstrap is force-refreshed (this is how a
    planner could otherwise solve transfers for a gameweek already in
    progress).
  - is_complete() requires event-level finished AND data_checked. A GW
    with unplayed fixtures is never complete.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger("rivalr.gameweek")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def state(client) -> dict:
    """{current, next, next_deadline} with a stale-cache self-heal."""
    events = client.bootstrap()["events"]
    now = datetime.now(timezone.utc)

    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt is None or _parse(nxt["deadline_time"]) <= now:
        # Cached bootstrap straddled a deadline: is_next is stale and any
        # solve keyed on it would plan transfers for a gameweek already
        # underway. Refresh and re-read.
        log.warning(
            "bootstrap is_next is stale (deadline passed) - forcing refresh"
        )
        client.get("bootstrap-static/", force=True)
        events = client.bootstrap()["events"]
        nxt = next((e for e in events if e.get("is_next")), None)
        if nxt is None or _parse(nxt["deadline_time"]) <= now:
            # API itself slow to flip (rare): advance manually.
            future = [e for e in events if _parse(e["deadline_time"]) > now]
            if not future:
                raise RuntimeError("no upcoming gameweek deadline")
            nxt = min(future, key=lambda e: e["id"])
            log.warning("API is_next still stale; advanced to gw%d", nxt["id"])

    cur = next((e for e in events if e.get("is_current")), None)
    return {
        "current": cur["id"] if cur else None,
        "next": nxt["id"],
        "next_deadline": _parse(nxt["deadline_time"]),
    }


def next_gw(client) -> int:
    return state(client)["next"]


def next_deadline(client) -> tuple[int, datetime]:
    s = state(client)
    return s["next"], s["next_deadline"]


def current_gw(client) -> int:
    s = state(client)
    return s["current"] if s["current"] is not None else s["next"]


def is_complete(client, gw: int) -> bool:
    """True only when the event is finished AND data_checked. A gameweek
    with fixtures still to play is NEVER complete, regardless of what
    per-fixture flags claim."""
    for e in client.bootstrap()["events"]:
        if e["id"] == gw:
            return bool(e.get("finished")) and bool(e.get("data_checked"))
    return False


def fixture_played(fixture: dict) -> bool:
    """Has this fixture been played (at least kicked off)? Uses started,
    not the unreliable per-fixture finished flag."""
    return bool(fixture.get("started"))
