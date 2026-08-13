"""Mini-league intelligence: rivals, mini-league effective ownership,
shield/sword classification, head-to-head squad comparisons.

This is the differentiator module. Everything downstream (optimiser modes,
report) keys off the output of build_rivals_report().

Definitions
-----------
Mini-league effective ownership (EO) of a player:

    EO = (n_owned + n_captained + n_triple_captained) / league_size

where n_owned counts managers with the player anywhere in their 15,
n_captained counts managers captaining him, and n_triple_captained counts
managers captaining him with the Triple Captain chip active. A TC'd captain
therefore contributes 3 to the numerator: owned + captained + tc.

SHIELD  high mini-league EO — owning him is rank-neutral, NOT owning him is
        a bet against the league.
SWORD   low mini-league EO but high projection — a differential that moves
        rank if it hauls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fetch import FPLClient

log = logging.getLogger("rivalr.rivals")

# Chip inventory per manager for the season. Adjust if FPL changes the rules
# (e.g. the 2024/25 second-half chip set). Names match the API's chip names.
CHIP_INVENTORY: dict[str, int] = {
    "wildcard": 2,   # one per half-season
    "freehit": 1,
    "bboost": 1,
    "3xc": 1,
}

SHIELD_EO_THRESHOLD = 0.5   # >= 50% mini-league EO
SWORD_EO_THRESHOLD = 0.15   # <= 15% mini-league EO
SWORD_MIN_PROJECTION = 4.5  # xPts next GW to qualify as a sword

MAX_RIVALS = 50  # safety cap: this is a mini-league tool, not a 100k scraper


@dataclass
class ManagerState:
    entry_id: int
    name: str
    team_name: str
    rank: int
    total_points: int
    squad: list[int] = field(default_factory=list)      # all 15 element ids
    starters: list[int] = field(default_factory=list)   # positions 1-11
    bench: list[int] = field(default_factory=list)      # positions 12-15
    captain: int | None = None
    vice_captain: int | None = None
    active_chip: str | None = None
    chips_used: list[dict] = field(default_factory=list)  # [{name, event}]
    bank: float = 0.0
    team_value: float = 0.0
    transfers: list[dict] = field(default_factory=list)

    @property
    def chips_left(self) -> dict[str, int]:
        left = dict(CHIP_INVENTORY)
        for chip in self.chips_used:
            name = chip["name"]
            if name in left and left[name] > 0:
                left[name] -= 1
        return left


def fetch_league_entries(client: FPLClient, league_id: int) -> tuple[str, list[dict]]:
    """All entries in a classic league, following pagination."""
    entries: list[dict] = []
    page = 1
    league_name = ""
    while True:
        data = client.league_standings(league_id, page=page)
        league_name = data["league"]["name"]
        standings = data["standings"]
        entries.extend(standings["results"])
        if not standings.get("has_next") or len(entries) >= MAX_RIVALS:
            break
        page += 1
    if len(entries) > MAX_RIVALS:
        log.warning(
            "league %s has more than %d entries; truncating to top %d",
            league_id, MAX_RIVALS, MAX_RIVALS,
        )
        entries = entries[:MAX_RIVALS]
    return league_name, entries


def fetch_manager_state(client: FPLClient, row: dict, gw: int) -> ManagerState:
    """Full state for one manager: squad, chips, bank, transfer history."""
    entry_id = row["entry"]
    state = ManagerState(
        entry_id=entry_id,
        name=row.get("player_name", str(entry_id)),
        team_name=row.get("entry_name", ""),
        rank=row.get("rank", 0),
        total_points=row.get("total", 0),
    )
    try:
        picks = client.entry_picks(entry_id, gw)
    except Exception:
        log.error("could not fetch picks for entry %s gw %s", entry_id, gw)
        return state

    state.active_chip = picks.get("active_chip")
    for p in picks["picks"]:
        state.squad.append(p["element"])
        if p["position"] <= 11:
            state.starters.append(p["element"])
        else:
            state.bench.append(p["element"])
        if p["is_captain"]:
            state.captain = p["element"]
        if p["is_vice_captain"]:
            state.vice_captain = p["element"]

    entry_meta = client.entry(entry_id)
    state.bank = entry_meta.get("last_deadline_bank", 0) / 10.0
    state.team_value = entry_meta.get("last_deadline_value", 0) / 10.0

    history = client.entry_history(entry_id)
    state.chips_used = [
        {"name": c["name"], "event": c["event"]} for c in history.get("chips", [])
    ]
    state.transfers = client.entry_transfers(entry_id)
    return state


# -- EO and classification -------------------------------------------------


def mini_league_eo(managers: list[ManagerState]) -> dict[int, float]:
    """Mini-league effective ownership per player (see module docstring)."""
    n = len(managers)
    if n == 0:
        return {}
    counts: dict[int, float] = {}
    for m in managers:
        for element in m.squad:
            counts[element] = counts.get(element, 0) + 1
        if m.captain is not None:
            counts[m.captain] = counts.get(m.captain, 0) + 1
            if m.active_chip == "3xc":
                counts[m.captain] = counts.get(m.captain, 0) + 1
    return {element: count / n for element, count in counts.items()}


def classify_pool(
    eo: dict[int, float],
    projections: dict[int, float],
    shield_threshold: float = SHIELD_EO_THRESHOLD,
    sword_eo_max: float = SWORD_EO_THRESHOLD,
    sword_min_projection: float = SWORD_MIN_PROJECTION,
) -> dict[int, str]:
    """SHIELD / SWORD / NEUTRAL for every player in the league pool plus any
    projected player outside it (swords are often *not* owned by anyone yet)."""
    labels: dict[int, str] = {}
    pool = set(eo) | set(projections)
    for element in pool:
        player_eo = eo.get(element, 0.0)
        proj = projections.get(element, 0.0)
        if player_eo >= shield_threshold:
            labels[element] = "SHIELD"
        elif player_eo <= sword_eo_max and proj >= sword_min_projection:
            labels[element] = "SWORD"
        else:
            labels[element] = "NEUTRAL"
    return labels


def compare_squads(me: ManagerState, rival: ManagerState) -> dict[str, Any]:
    mine, theirs = set(me.squad), set(rival.squad)
    shared = mine & theirs
    return {
        "overlap_pct": round(100 * len(shared) / 15, 1) if theirs else None,
        "shared": sorted(shared),
        "their_differentials": sorted(theirs - mine),
        "my_differentials": sorted(mine - theirs),
    }


# -- top-level report ------------------------------------------------------


def build_rivals_report(
    client: FPLClient,
    team_id: int,
    league_id: int,
    projections: dict[int, float] | None = None,
    gw: int | None = None,
) -> dict[str, Any]:
    """The full mini-league intelligence report as a JSON-serialisable dict.

    projections: element_id -> next-GW xPts. Falls back to the FPL site's
    own ep_next if none supplied (weaker, but keeps the module standalone).
    """
    gw = gw or client.current_gw()
    league_name, rows = fetch_league_entries(client, league_id)

    managers = [fetch_manager_state(client, row, gw) for row in rows]
    by_id = {m.entry_id: m for m in managers}
    if team_id not in by_id:
        raise ValueError(f"team {team_id} not found in league {league_id}")
    me = by_id[team_id]

    if projections is None:
        bootstrap = client.bootstrap()
        projections = {
            el["id"]: float(el.get("ep_next") or 0.0) for el in bootstrap["elements"]
        }
        log.warning("no model projections supplied; using FPL ep_next as fallback")

    eo = mini_league_eo(managers)
    labels = classify_pool(eo, projections)

    my_squad = set(me.squad)
    missing_shields = sorted(
        el for el, label in labels.items() if label == "SHIELD" and el not in my_squad
    )
    available_swords = sorted(
        (el for el, label in labels.items() if label == "SWORD" and el not in my_squad),
        key=lambda el: -projections.get(el, 0.0),
    )

    rivals = []
    for m in managers:
        if m.entry_id == team_id:
            continue
        rivals.append(
            {
                "entry_id": m.entry_id,
                "name": m.name,
                "team_name": m.team_name,
                "rank": m.rank,
                "total_points": m.total_points,
                "gap": m.total_points - me.total_points,  # +ve: they lead me
                "captain": m.captain,
                "active_chip": m.active_chip,
                "chips_used": m.chips_used,
                "chips_left": m.chips_left,
                "bank": m.bank,
                "team_value": m.team_value,
                "transfers_this_season": len(m.transfers),
                **compare_squads(me, m),
            }
        )
    rivals.sort(key=lambda r: r["rank"])

    return {
        "gw": gw,
        "league_id": league_id,
        "league_name": league_name,
        "league_size": len(managers),
        "my_entry_id": team_id,
        "my_rank": me.rank,
        "my_total_points": me.total_points,
        "my_squad": me.squad,
        "my_captain": me.captain,
        "my_chips_left": me.chips_left,
        "my_bank": me.bank,
        "effective_ownership": {str(k): round(v, 4) for k, v in eo.items()},
        "classification": {str(k): v for k, v in labels.items()},
        "missing_shields": missing_shields,
        "available_swords": available_swords[:20],
        "rivals": rivals,
    }


def write_rivals_report(report: dict, path: str | Path = "rivals_report.json") -> Path:
    out = Path(path)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    return out
