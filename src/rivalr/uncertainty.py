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


def transferred_players(client: FPLClient) -> dict[int, dict]:
    """Players whose 2026-27 club differs from their 2025-26 club.

    Their per-90 rates and rolling form were measured in a DIFFERENT
    system, so they carry elevated uncertainty (NEW_CLUB) until they
    have SETTLED_AFTER_MATCHES played matches for the new club - by
    which point the cold-start blends run on new-club data anyway.

    Club identity always comes from live bootstrap-static; the 2025-26
    prior (vaastav, joined by permanent player code) supplies only the
    previous club name for display."""
    import csv
    from pathlib import Path

    bootstrap = client.bootstrap()
    cur_team = {t["id"]: t["name"] for t in bootstrap["teams"]}
    cache = Path(client.cache_dir)
    players_csv = cache / "vaastav_2025-26_players_raw.csv"
    teams_csv = cache / "vaastav_2025-26_teams.csv"
    try:
        for path, name in ((players_csv, "players_raw.csv"), (teams_csv, "teams.csv")):
            if not path.exists():
                import urllib.request
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/vaastav/"
                    f"Fantasy-Premier-League/master/data/2025-26/{name}", path,
                )
        prev_team_name = {
            int(t["id"]): t["name"]
            for t in csv.DictReader(teams_csv.open(encoding="utf-8"))
        }
        prev_by_code = {
            p["code"]: prev_team_name.get(int(p["team"]), "?")
            for p in csv.DictReader(players_csv.open(encoding="utf-8"))
        }
    except Exception:
        log.error("transferred_players: 2025-26 prior unavailable", exc_info=True)
        return {}

    out: dict[int, dict] = {}
    for el in bootstrap["elements"]:
        prev = prev_by_code.get(str(el.get("code")))
        cur = cur_team[el["team"]]
        if prev is None or prev == cur:
            continue
        played = sum(
            1 for h in client.element_summary(el["id"]).get("history", [])
            if h["minutes"] > 0
        )
        if played < SETTLED_AFTER_MATCHES:
            out[el["id"]] = {"from": prev, "to": cur, "matches_played": played}
    if out:
        log.info("NEW_CLUB uncertainty active for %d transferred players", len(out))
    return out


def player_flags(client: FPLClient) -> dict[int, dict]:
    """{player_id: {kinds: [MGR_CHG, NEW_CLUB], ...info}}.

    MGR_CHG keys off the player's CURRENT club (live bootstrap-static);
    NEW_CLUB marks summer movers whose prior-season rates come from a
    different system. A player can carry both (e.g. a mover joining a
    club that also changed manager)."""
    flags = team_flags(client)
    active = {tid: f for tid, f in flags.items() if f["active"]}
    moved = transferred_players(client)

    out: dict[int, dict] = {}
    for el in client.bootstrap()["elements"]:
        pid = el["id"]
        kinds = []
        info: dict = {}
        if el["team"] in active:
            kinds.append("MGR_CHG")
            info.update(active[el["team"]])
        if pid in moved:
            kinds.append("NEW_CLUB")
            info["transfer"] = moved[pid]
        if kinds:
            info["kinds"] = kinds
            out[pid] = info
    return out
