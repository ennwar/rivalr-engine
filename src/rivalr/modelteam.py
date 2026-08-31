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


def _snapshot(gw: int) -> dict | None:
    try:
        path = ledger._latest_ledger_for(gw, ledger.LEDGER_DIR)
    except FileNotFoundError:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def decide_gw(client: FPLClient, gw: int, prev_decision: dict | None) -> dict | None:
    """The model's OWN pre-deadline decision for one gameweek: a fresh
    solve seeded with ITS squad, ITS bank, ITS free-transfer count, on
    the snapshot's pre-deadline projections. Never reads the human's
    squad or recommendations (except GW1, where the recommendation WAS
    a from-scratch draft solved for an empty squad).

    Price approximation, stated openly: buys and sells use current
    now_cost; price-change profits/losses are not simulated.
    """
    snap = _snapshot(gw)
    if snap is None:
        return None
    projections = {
        int(k): v for k, v in (snap.get("projections") or {}).items() if v
    }
    bootstrap = client.bootstrap()
    price = {el["id"]: el["now_cost"] for el in bootstrap["elements"]}

    if prev_decision is None:
        # GW1: the ledger's draft was solved from an empty squad - it is
        # genuinely the model's own, and my entry played no part in it.
        rec = snap.get("recommendation") or {}
        squad = list(rec.get("transfers_in") or [])[:15]
        if len(squad) != 15:
            log.warning("model gw%d draft has %d players", gw, len(squad))
            return None
        bank = round(100.0 - sum(price.get(p, 0) for p in squad) / 10.0, 1)
        return {
            "gw": gw, "squad": squad, "transfers": {"in": [], "out": []},
            "captain": rec.get("captain"), "hits": 0,
            "bank": max(bank, 0.0), "ft_after": 1, "drafted": True,
        }

    from . import optimise

    ft = prev_decision.get("ft_after", 1)
    my_data = {
        "picks": [
            {"element": p, "selling_price": price.get(p, 40),
             "purchase_price": price.get(p, 40), "element_type": 0}
            for p in prev_decision["squad"]
        ],
        "chips": [],
        "transfers": {"bank": int(prev_decision.get("bank", 0.0) * 10),
                      "limit": ft, "made": 0},
    }
    stub_rep = {"effective_ownership": {}, "classification": {},
                "rivals": [], "small_league": True, "league_size": 0}
    horizon = min(5, 39 - gw)
    plans = optimise.solve_all_modes(
        client=client, team_id=0, projections=projections,
        rivals_report=stub_rep, target_id=None, horizon=horizon,
        my_data_override=my_data,
        solver_options={"override_next_gw": gw,
                        "weekly_hit_limit": 0, "hit_limit": 0},
    )
    plan = plans.get("points")
    if plan is None:
        log.error("model gw%d solve failed - carrying squad unchanged", gw)
        return {
            "gw": gw, "squad": prev_decision["squad"],
            "transfers": {"in": [], "out": []},
            "captain": prev_decision.get("captain"), "hits": 0,
            "bank": prev_decision.get("bank", 0.0),
            "ft_after": min(5, ft + 1), "solve_failed": True,
        }
    t_in = plan.get("transfers_in") or []
    t_out = plan.get("transfers_out") or []
    squad = [p for p in prev_decision["squad"] if p not in t_out] + [
        p for p in t_in if p not in prev_decision["squad"]
    ]
    spent = sum(price.get(p, 0) for p in t_in) - sum(
        price.get(p, 0) for p in t_out
    )
    n_moves = len(t_in)
    hits = max(0, n_moves - ft)
    return {
        "gw": gw, "squad": squad,
        "transfers": {"in": t_in, "out": t_out},
        "captain": plan.get("captain"), "hits": hits,
        "bank": round(max(prev_decision.get("bank", 0.0) - spent / 10.0, 0.0), 1),
        "ft_after": min(5, max(ft - n_moves, 0) + 1),
    }


def build_gw(client: FPLClient, gw: int, prev: dict | None,
             decision: dict | None = None) -> dict | None:
    """One settled gameweek of the model team, scored from ITS OWN
    pre-deadline decision (not the human's recommendation)."""
    snap = _snapshot(gw)
    if snap is None or decision is None:
        return None
    proj = {int(k): (v[0] if v else 0.0)
            for k, v in (snap.get("projections") or {}).items()}

    bootstrap = client.bootstrap()
    etype = {el["id"]: el["element_type"] for el in bootstrap["elements"]}
    names = {el["id"]: el["web_name"] for el in bootstrap["elements"]}
    stats = {el["id"]: el["stats"]
             for el in client.event_live(gw)["elements"]}
    minutes = {pid: s.get("minutes", 0) for pid, s in stats.items()}
    pts = {pid: s.get("total_points", 0) for pid, s in stats.items()}

    squad = decision["squad"]
    t_in = decision["transfers"]["in"]
    t_out = decision["transfers"]["out"]
    hits = decision.get("hits", 0)
    ft_after = decision.get("ft_after", 1)

    xi0 = _best_xi(squad, proj, etype)
    xi = _auto_subs(xi0, squad, minutes, proj, etype)
    captain = decision.get("captain")
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
        "transfers": {"in": t_in, "out": t_out},
        "drafted": decision.get("drafted", False),
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
    """Worker-side. Two passes:

    1. DECIDE: every gameweek with a snapshot gets a model decision -
       a fresh solve from the model's own squad on that snapshot's
       pre-deadline projections (sequential, so each builds on the
       previous decision). Normally this runs pre-deadline; a missing
       one is backfilled from the snapshot, which is hindsight-free
       because the projections were frozen before the deadline.
    2. SETTLE: settled gameweeks get scored from their decision.
    """
    events = client.bootstrap()["events"]
    snapshot_gws = []
    for ev in sorted(events, key=lambda e: e["id"]):
        if _snapshot(ev["id"]) is not None:
            snapshot_gws.append(ev["id"])

    decisions = {d["gw"]: d for d in store.model_decisions()}
    prev_dec = None
    for gw in snapshot_gws:
        # side-sync: the snapshot's first-GW projections into pg so the
        # web service can show "his remaining fixture is worth ~X"
        try:
            if not store.gw_projections(gw):
                snap = _snapshot(gw)
                store.put_gw_projections(gw, {
                    k: round(v[0], 2)
                    for k, v in (snap.get("projections") or {}).items() if v
                })
        except Exception:
            log.warning("gw_projections sync failed for gw%d", gw,
                        exc_info=True)
        if gw in decisions:
            prev_dec = decisions[gw]
            continue
        dec = decide_gw(client, gw, prev_dec)
        if dec is None:
            continue
        store.put_model_decision(gw, dec)
        decisions[gw] = dec
        log.info("model decision gw%d: %d in / %d out, captain %s, "
                 "bank %.1f, ft_after %d",
                 gw, len(dec["transfers"]["in"]), len(dec["transfers"]["out"]),
                 dec.get("captain"), dec.get("bank", 0.0), dec["ft_after"])
        prev_dec = dec

    settled = sorted(
        ev["id"] for ev in events
        if ev.get("finished") and ev.get("data_checked")
    )
    rows = {r["gw"]: r for r in store.model_rows()}
    prev = None
    for gw in settled:
        if gw in rows:
            prev = rows[gw]
            continue
        row = build_gw(client, gw, prev, decision=decisions.get(gw))
        if row is None:
            log.info("model team: no snapshot/decision for gw%d, skipping", gw)
            continue
        store.put_model_gw(gw, row)
        log.info("model team gw%d: %d pts (total %d, hits %d)",
                 gw, row["points"], row["total"], row["hits"])
        prev = row
