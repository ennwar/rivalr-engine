"""Expected minutes estimator (v0).

P(start) from rolling starts, minutes trend, the FPL availability flag and
fixture congestion. Projections are multiplied by an expected-minutes factor.

v0 limitations, by design:
  - purely statistical; no press-conference / team-news signal
  - congestion penalty is a blunt discount for heavy schedules
A news-based v1 plugs in via `news_adjustment()` below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .fetch import FPLClient

log = logging.getLogger("rivalr.minutes")

ROLLING_WINDOW = 6       # matches used for the start-rate baseline
TREND_WINDOW = 3         # recent matches for the minutes trend
CAMEO_MINUTES = 18.0     # assumed minutes for a sub appearance
CONGESTION_PENALTY = 0.9  # P(start) multiplier when 2+ games in last 7 days
STATUS_UNAVAILABLE = {"i", "s", "n"}  # injured / suspended / not in squad


def news_adjustment(player_id: int) -> float | None:
    """Hook for a news-based v1: return a P(start) override in [0, 1],
    or None to keep the statistical estimate. v0 always returns None."""
    return None


@dataclass
class MinutesEstimate:
    player_id: int
    p_start: float
    expected_minutes: float
    factor: float          # multiply per-GW projections by this
    flags: list[str]       # human-readable risk flags for the report


def _availability_multiplier(element: dict) -> tuple[float, list[str]]:
    """From bootstrap-static element: status + chance_of_playing."""
    flags: list[str] = []
    status = element.get("status", "a")
    chance = element.get("chance_of_playing_next_round")
    if status in STATUS_UNAVAILABLE:
        flags.append(f"unavailable (status={status})")
        return 0.0, flags
    if status == "d":
        pct = (chance if chance is not None else 50) / 100.0
        flags.append(f"doubtful ({int(pct * 100)}%)")
        return pct, flags
    return 1.0, flags


def estimate_minutes(
    client: FPLClient,
    player_id: int,
    reference_time: datetime | None = None,
) -> MinutesEstimate:
    """v0 expected-minutes estimate for one player."""
    bootstrap = client.bootstrap()
    element = next(el for el in bootstrap["elements"] if el["id"] == player_id)
    history = client.element_summary(player_id).get("history", [])
    played = [h for h in history if h["minutes"] > 0 or h["starts"] > 0]

    flags: list[str] = []

    recent = history[-ROLLING_WINDOW:]
    if not recent:
        # No matches yet this season (GW1 or new signing): fall back to a
        # prior from last season's total minutes (history_past).
        past = client.element_summary(player_id).get("history_past", [])
        last_minutes = past[-1]["minutes"] if past else None
        if last_minutes is None:
            p_start = 0.6
            flags.append("no history at all, default P(start)=0.6")
        elif last_minutes >= 2400:
            p_start = 0.85
            flags.append(f"pre-season prior: {last_minutes} mins last season")
        elif last_minutes >= 1200:
            p_start = 0.65
            flags.append(f"pre-season prior: {last_minutes} mins last season")
        else:
            p_start = 0.45
            flags.append(f"pre-season prior: only {last_minutes} mins last season")
        avg_start_minutes = 80.0
    else:
        start_rate = sum(h["starts"] for h in recent) / len(recent)

        trend = history[-TREND_WINDOW:]
        trend_minutes = sum(h["minutes"] for h in trend) / max(len(trend), 1)
        baseline_minutes = sum(h["minutes"] for h in recent) / len(recent)
        # Blend: mostly the rolling start rate, nudged by whether recent
        # minutes are rising or falling relative to the window average.
        nudge = 0.0
        if baseline_minutes > 0:
            nudge = 0.15 * (trend_minutes - baseline_minutes) / 90.0
        p_start = min(1.0, max(0.0, start_rate + nudge))

        started = [h for h in recent if h["starts"]]
        avg_start_minutes = (
            sum(h["minutes"] for h in started) / len(started) if started else 75.0
        )
        if start_rate < 0.5 and played:
            flags.append(f"rotation risk (started {int(start_rate * len(recent))}/{len(recent)})")

    # Fixture congestion: 2+ club games in the 7 days before reference_time.
    now = reference_time or datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    recent_games = 0
    for h in history:
        kickoff = h.get("kickoff_time")
        if kickoff and week_ago <= datetime.fromisoformat(kickoff.replace("Z", "+00:00")) <= now:
            recent_games += 1
    if recent_games >= 2:
        p_start *= CONGESTION_PENALTY
        flags.append(f"congestion ({recent_games} games in 7d)")

    avail, avail_flags = _availability_multiplier(element)
    flags.extend(avail_flags)
    p_start *= avail

    override = news_adjustment(player_id)
    if override is not None:
        p_start = override
        flags.append("news override")

    p_cameo = (1.0 - p_start) * 0.5 if avail > 0 else 0.0
    xmins = p_start * avg_start_minutes + p_cameo * CAMEO_MINUTES

    return MinutesEstimate(
        player_id=player_id,
        p_start=round(p_start, 3),
        expected_minutes=round(xmins, 1),
        factor=round(min(1.0, xmins / 90.0), 3),
        flags=flags,
    )


def apply_minutes(
    projections: dict[int, list[float]],
    estimates: dict[int, MinutesEstimate],
) -> dict[int, list[float]]:
    """Scale per-GW projections by each player's expected-minutes factor."""
    return {
        pid: [round(x * estimates[pid].factor, 3) if pid in estimates else x for x in xs]
        for pid, xs in projections.items()
    }
