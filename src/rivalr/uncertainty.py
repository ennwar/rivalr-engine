"""Manager-change uncertainty: flag players whose projections rest on
last season's tactical assumptions.

The projection features are rolling means over 2025-26 + early 2026-27
data. Where a club changed manager in summer 2026, that history encodes
the OLD manager's system (pressing intensity, full-back roles, set-piece
duties, rotation habits) - so projections for those players carry
elevated uncertainty until the new regime has produced enough of its own
matches. Projections are NOT adjusted; the flag is surfaced alongside
LOW_CONF so the reader knows what a recommendation rests on.

A club stops being flagged once it has SETTLED_AFTER_MATCHES finished
2026-27 fixtures (same 5-match convention as the cold-start blends).

Source for the change list: premierleague.com "Manager line-up complete
for 2026/27 season" (fetched 2026-08-15). Mid-season sackings should be
added here by hand - keys are FPL bootstrap team names, verbatim.

Deliberately NOT flagged:
  Man Utd   Carrick in post since 13 Jan 2026 - half of 2025-26 is his
  Everton   Moyes since Jan 2025
  Brentford Andrews since summer 2025
"""

from __future__ import annotations

import logging

from .fetch import FPLClient

log = logging.getLogger("rivalr.uncertainty")

SETTLED_AFTER_MATCHES = 5

# FPL team name -> (incoming manager, outgoing manager), summer 2026.
MANAGER_CHANGES: dict[str, tuple[str, str]] = {
    "Bournemouth": ("Marco Rose", "Andoni Iraola"),
    "Chelsea": ("Xabi Alonso", "Liam Rosenior (interim) / Enzo Maresca"),
    "Crystal Palace": ("Pierre Sage", "Oliver Glasner"),
    "Fulham": ("Alvaro Arbeloa", "Marco Silva"),
    "Ipswich Town": ("Gary O'Neil", "Kieran McKenna"),
    "Liverpool": ("Andoni Iraola", "Arne Slot"),
    "Man City": ("Enzo Maresca", "Pep Guardiola"),
    "Newcastle": ("Matthias Jaissle", "Eddie Howe"),
    "Nott'm Forest": ("Oliver Glasner", "Sean Dyche"),
    "Spurs": ("Roberto De Zerbi", "Thomas Frank"),
}


def team_flags(client: FPLClient) -> dict[int, dict]:
    """{team_id: {new, out, matches_played, active}} for changed clubs.
    `active` goes False once the club has SETTLED_AFTER_MATCHES finished
    2026-27 fixtures."""
    bootstrap = client.bootstrap()
    by_name = {t["name"]: t["id"] for t in bootstrap["teams"]}

    played: dict[int, int] = {}
    for f in client.fixtures():
        if f.get("finished"):
            for tid in (f["team_h"], f["team_a"]):
                played[tid] = played.get(tid, 0) + 1

    flags: dict[int, dict] = {}
    for name, (new, out) in MANAGER_CHANGES.items():
        tid = by_name.get(name)
        if tid is None:
            log.warning("uncertainty: team %r not in bootstrap - config stale?", name)
            continue
        n = played.get(tid, 0)
        flags[tid] = {
            "team": name,
            "new": new,
            "out": out,
            "matches_played": n,
            "active": n < SETTLED_AFTER_MATCHES,
        }
    active = [f["team"] for f in flags.values() if f["active"]]
    if active:
        log.info(
            "manager-change uncertainty active for %d clubs (< %d matches): %s",
            len(active), SETTLED_AFTER_MATCHES, ", ".join(sorted(active)),
        )
    return flags


def player_flags(client: FPLClient) -> dict[int, dict]:
    """{player_id: team flag info} for every player at a flagged club."""
    flags = team_flags(client)
    active = {tid: f for tid, f in flags.items() if f["active"]}
    if not active:
        return {}
    return {
        el["id"]: active[el["team"]]
        for el in client.bootstrap()["elements"]
        if el["team"] in active
    }
