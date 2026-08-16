"""Fixture ticker: 20 clubs x N gameweeks, FPL FDR plus our own
difficulty overlay derived from opponent rolling xG/xGA (Understat).

Two overlays because a fixture that is good for defenders is not
necessarily good for attackers:

  our_def  difficulty for this club's DEFENDERS/GK = the opponent's
           attacking output (rolling xG per match), quintiled 1-5
  our_att  difficulty for this club's ATTACKERS = the opponent's
           defensive solidity (rolling xGA per match, inverted),
           quintiled 1-5

1 = easiest, 5 = hardest (FDR convention). Blanks = empty fixture
list for a GW; doubles = two entries.
"""

from __future__ import annotations

import logging

from .fetch import FPLClient
from .understat import FPL_TO_UNDERSTAT_TEAM, Understat

log = logging.getLogger("rivalr.fixtures")

ROLL = 6  # matches in the rolling opponent-strength window


def _quintile(value: float, sorted_values: list[float]) -> int:
    """1..5 rank of value within the league distribution."""
    below = sum(1 for v in sorted_values if v < value)
    return min(5, max(1, 1 + (below * 5) // max(len(sorted_values), 1)))


def fixture_grid(client: FPLClient, horizon: int = 8) -> dict:
    bootstrap = client.bootstrap()
    teams = bootstrap["teams"]
    next_gw = client.next_gw()
    gws = list(range(next_gw, min(39, next_gw + horizon)))

    # Rolling attack (xG) and defence (xGA) per club from Understat,
    # cross-season histories already merged by teams_data().
    year = int(bootstrap["events"][0]["deadline_time"][:4])
    xg_rate: dict[int, float] = {}
    xga_rate: dict[int, float] = {}
    try:
        hist = Understat(season=year, cache_dir=client.cache_dir).teams_data()
        for t in teams:
            title = FPL_TO_UNDERSTAT_TEAM.get(t["name"], t["name"])
            tail = hist.get(title, [])[-ROLL:]
            if tail:
                xg_rate[t["id"]] = sum(float(m["xG"]) for m in tail) / len(tail)
                xga_rate[t["id"]] = sum(float(m["xGA"]) for m in tail) / len(tail)
    except Exception:
        log.exception("understat unavailable for fixture grid - FDR only")

    xg_sorted = sorted(xg_rate.values())
    xga_sorted = sorted(xga_rate.values())

    def our_difficulty(opp_id: int) -> tuple[int | None, int | None]:
        if opp_id not in xg_rate:
            return None, None
        # defenders fear opponent attack; attackers fear a low-xGA defence
        d = _quintile(xg_rate[opp_id], xg_sorted)
        a = _quintile(-xga_rate[opp_id], sorted(-v for v in xga_sorted))
        return d, a

    grid: dict[int, dict[int, list[dict]]] = {t["id"]: {gw: [] for gw in gws}
                                              for t in teams}
    short = {t["id"]: t["short_name"] for t in teams}
    for f in client.fixtures():
        gw = f.get("event")
        if gw not in gws or f.get("finished"):
            continue
        for tid, opp, home, fdr in (
            (f["team_h"], f["team_a"], True, f.get("team_h_difficulty")),
            (f["team_a"], f["team_h"], False, f.get("team_a_difficulty")),
        ):
            d, a = our_difficulty(opp)
            grid[tid][gw].append({
                "opponent": short.get(opp, "?"),
                "home": home,
                "fdr": fdr,
                "our_def": d,
                "our_att": a,
            })

    clubs = []
    for t in sorted(teams, key=lambda t: t["name"]):
        fixtures = {str(gw): grid[t["id"]][gw] for gw in gws}
        clubs.append({
            "team_id": t["id"],
            "name": t["name"],
            "short": t["short_name"],
            "fixtures": fixtures,
        })
    return {
        "gameweeks": gws,
        "clubs": clubs,
        "overlays": ["fdr", "our_def", "our_att"],
        "understat_available": bool(xg_rate),
    }
