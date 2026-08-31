"""The Model: an autonomous FPL manager built from the ledger.

It drafted its own 100m squad at GW1 (the ledger's recommended draft),
applies each snapshot's recommended transfers exactly, banks free
transfers by its own sequence, and never sees the user's squad.

Reconstruction rules (no hindsight anywhere):
  - XI/bench: chosen by the SNAPSHOT's own pre-deadline projections -
    the model follows its own advice, formation-legal (1 GK, 3-5 DEF,
    2-5 MID, 1-3 FWD).
  - Captain: the recommendation's captain; if he plays 0 minutes the
    vice (next-highest-projected starter) doubles instead.
  - Auto-subs: FPL-style approximation - starters with 0 minutes swap
    for the highest-projected bench player that keeps the formation
    legal (GK for GK).
  - Hits: transfers beyond its own banked FTs cost -4 each
    (1 FT/week from GW2, bank capped at 5, chips would exempt).
  - Chips: it plays whatever its recommendations say (none so far).
  - Budget: every squad was solver-validated within budget at
    recommendation time; prices are not re-simulated and the UI says so.

The WORKER computes one row per settled gameweek from the ledger
snapshots on its volume and stores them in Postgres (model_team table);
the API serves them read-only.
"""

from __future__ import annotations

import json
import logging

from . import ledger
from .fetch import FPLClient

log = logging.getLogger("rivalr.modelteam")

MODEL_NAME = "The Model"

# (DEF, MID, FWD) legal formations, XI = 1 GK + these 10
FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (4, 5, 1),
    (5, 3, 2), (5, 4, 1), (5, 2, 3), (3, 3, 4),
]
FORMATIONS = [f for f in FORMATIONS if sum(f) == 10 and f[2] <= 3]


def _best_xi(squad: list[int], proj: dict[int, float],
             etype: dict[int, int]) -> list[int]:
    """Highest-projected legal XI from a 15-man squad."""
    by_pos = {1: [], 2: [], 3: [], 4: []}
    for pid in squad:
        by_pos.get(etype.get(pid, 0), []).append(pid)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: -proj.get(p, 0.0))

    best, best_score = None, -1.0
    gk = by_pos[1][:1]
    for d, m, f in FORMATIONS:
        if len(by_pos[2]) < d or len(by_pos[3]) < m or len(by_pos[4]) < f or not gk:
            continue
        xi = gk + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f]
        score = sum(proj.get(p, 0.0) for p in xi)
        if score > best_score:
            best, best_score = xi, score
    return best or squad[:11]


def _auto_subs(xi: list[int], squad: list[int], minutes: dict[int, int],
               proj: dict[int, float], etype: dict[int, int]) -> list[int]:
    """Swap 0-minute starters for the best-projected bench players that
    keep the XI legal."""
    bench = [p for p in squad if p not in xi]
    bench.sort(key=lambda p: -proj.get(p, 0.0))
    final = list(xi)
    for i, pid in enumerate(list(final)):
        if minutes.get(pid, 0) > 0:
            continue
        for b in bench:
            if minutes.get(b, 0) == 0:
                continue
            trial = list(final)
            trial[i] = b
            counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for p in trial:
                counts[etype.get(p, 0)] = counts.get(etype.get(p, 0), 0) + 1
            if (counts[1] == 1 and 3 <= counts[2] <= 5
                    and 2 <= counts[3] <= 5 and 1 <= counts[4] <= 3):
                final[i] = b
                bench.remove(b)
                break
    return final


def build_gw(client: FPLClient, gw: int, prev: dict | None) -> dict | None:
    """One settled gameweek of the model team, from the ledger snapshot
    on this machine's volume. Returns None if no snapshot exists."""
    try:
        path = ledger._latest_ledger_for(gw, ledger.LEDGER_DIR)
    except FileNotFoundError:
        return None
    snap = json.loads(path.read_text(encoding="utf-8"))
    rec = snap.get("recommendation") or {}
    proj = {int(k): (v[0] if v else 0.0)
            for k, v in (snap.get("projections") or {}).items()}

    bootstrap = client.bootstrap()
    etype = {el["id"]: el["element_type"] for el in bootstrap["elements"]}
    names = {el["id"]: el["web_name"] for el in bootstrap["elements"]}
    stats = {el["id"]: el["stats"]
             for el in client.event_live(gw)["elements"]}
    minutes = {pid: s.get("minutes", 0) for pid, s in stats.items()}
    pts = {pid: s.get("total_points", 0) for pid, s in stats.items()}

    t_in = rec.get("transfers_in") or []
    t_out = rec.get("transfers_out") or []
    if prev is None:
        squad = list(t_in)[:15]
        n_moves, ft, hits, ft_after = 0, 0, 0, 1
        if len(squad) != 15:
            log.warning("model draft gw%d has %d players", gw, len(squad))
    else:
        squad = [p for p in prev["squad"] if p not in t_out] + [
            p for p in t_in if p not in prev["squad"]
        ]
        ft = prev.get("ft_after", 1)
        n_moves = len(t_in)
        hits = max(0, n_moves - ft)
        ft_after = min(5, max(ft - n_moves, 0) + 1)

    xi0 = _best_xi(squad, proj, etype)
    xi = _auto_subs(xi0, squad, minutes, proj, etype)
    captain = rec.get("captain")
    if captain not in xi or minutes.get(captain, 0) == 0:
        outfield = sorted((p for p in xi if minutes.get(p, 0) > 0),
                          key=lambda p: -proj.get(p, 0.0))
        captain = outfield[0] if outfield else (xi[0] if xi else None)

    gw_points = sum(pts.get(p, 0) for p in xi)
    if captain is not None:
        gw_points += pts.get(captain, 0)
    gw_points -= 4 * hits

    total = (prev.get("total", 0) if prev else 0) + gw_points
    return {
        "gw": gw,
        "squad": squad,
        "xi": xi,
        "bench": [p for p in squad if p not in xi],
        "captain": captain,
        "transfers": {"in": t_in if prev is not None else [],
                      "out": t_out},
        "drafted": prev is None,
        "hits": hits,
        "ft_after": ft_after,
        "chip": None,
        "points": gw_points,
        "total": total,
        "players": [
            {"id": p, "name": names.get(p, f"#{p}"),
             "points": pts.get(p, 0), "in_xi": p in xi,
             "captain": p == captain}
            for p in squad
        ],
    }


def sync(client: FPLClient, store) -> None:
    """Worker-side: compute any settled gameweeks not yet in Postgres."""
    settled = sorted(
        ev["id"] for ev in client.bootstrap()["events"]
        if ev.get("finished") and ev.get("data_checked")
    )
    rows = {r["gw"]: r for r in store.model_rows()}
    prev = None
    for gw in settled:
        if gw in rows:
            prev = rows[gw]
            continue
        row = build_gw(client, gw, prev)
        if row is None:
            log.info("model team: no snapshot for gw%d, skipping", gw)
            prev = None if prev is None else prev
            continue
        store.put_model_gw(gw, row)
        log.info("model team gw%d: %d pts (total %d, hits %d)",
                 gw, row["points"], row["total"], row["hits"])
        prev = row
