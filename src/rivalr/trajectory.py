"""5-week projected points trajectory: me vs each rival, under three
transfer SCENARIOS (not predictions).

This is a new VIEW over data we already compute - the same per-GW
projections (OpenFPL + form + venue + DefCon, minutes-adjusted) and the
same MILP solver used for the planner. Nothing here is new modelling.

Building blocks per entity, each a cumulative path anchored at the
entity's CURRENT total points:
  - standstill: current squad projected forward, no transfers (best XI +
    captain each GW)
  - optimal:    the solver's plan for that squad/bank/free-transfers

The frontend composes the three scenarios from these:
  1. me.standstill + rivals.standstill
  2. me.optimal   + rivals.standstill
  3. me.optimal   + rivals.optimal   (rivals shown as a band, not a line)

HONESTY: the optimal-rival case assumes a rival plays perfectly, which
they won't. It is a worst-case bound, never a forecast of behaviour -
real outcomes sit between a rival's stand-still and optimal paths.
"""

from __future__ import annotations

import logging

from . import defcon, minutes, model, modelteam, optimise, rivals
from .briefdata import league_usable
from .fetch import FPLClient

log = logging.getLogger("rivalr.trajectory")

HORIZON = 5
MAX_OPTIMAL_RIVAL_SOLVES = 8  # bound cost; extras get standstill only

STAND_STILL_NOTE = (
    "These are SCENARIOS, not predictions - we can't know what a rival "
    "will actually do. Rivals rarely play optimally: real outcomes "
    "usually sit between the 'stand still' and 'optimal' lines. The "
    "optimal-rival case is a worst-case bound (their squad solved by the "
    "same optimiser we use for yours), shown as a shaded band - not a "
    "forecast of how they'll play."
)

_STUB_REP = {"effective_ownership": {}, "classification": {}, "rivals": [],
             "small_league": True, "league_size": 0}


def _gw_points(squad: list[int], proj_i: dict[int, float],
               etype: dict[int, int]) -> float:
    """Best-XI points for one GW from a squad, captain (top XI) doubled.
    A projection, so every picked player is assumed to play - no
    auto-subs (those settle actuals, which we don't have here)."""
    if not squad:
        return 0.0
    xi = modelteam._best_xi(squad, proj_i, etype)
    if not xi:
        return 0.0
    cap = max(xi, key=lambda p: proj_i.get(p, 0.0))
    return sum(proj_i.get(p, 0.0) for p in xi) + proj_i.get(cap, 0.0)


def _standstill_cum(squad: list[int], start: float,
                    final: dict[int, list[float]],
                    etype: dict[int, int]) -> list[float]:
    cum, total = [], start
    for i in range(HORIZON):
        proj_i = {p: (final.get(p) or [0.0] * HORIZON)[i] for p in squad}
        total += round(_gw_points(squad, proj_i, etype), 2)
        cum.append(round(total, 1))
    return cum


def _optimal_cum(client: FPLClient, team_id: int, start: float,
                 final: dict[int, list[float]]) -> list[float] | None:
    """Cumulative path when this team plays its solver-optimal plan.
    Reuses solve_all_modes with the shared projections (no re-modelling).
    None if the solve fails (caller falls back to standstill)."""
    try:
        plans = optimise.solve_all_modes(
            client=client, team_id=team_id, projections=final,
            rivals_report=_STUB_REP, target_id=None, horizon=HORIZON,
            solver_options={"weekly_hit_limit": 0, "hit_limit": 0},
        )
        plan = plans.get("points")
        if not plan or not plan.get("weeks"):
            return None
    except Exception:
        log.warning("optimal solve failed for team %s", team_id, exc_info=True)
        return None
    cum, total = [], start
    for wk in plan["weeks"][:HORIZON]:
        total += float(wk.get("xp") or 0.0)
        cum.append(round(total, 1))
    while len(cum) < HORIZON:  # short plan: hold flat
        cum.append(cum[-1] if cum else round(start, 1))
    return cum


def build_trajectory_json(
    client: FPLClient, team_id: int, league_id: int | None = None,
) -> dict:
    bootstrap = client.bootstrap()
    elements = {el["id"]: el for el in bootstrap["elements"]}
    etype = {pid: el["element_type"] for pid, el in elements.items()}
    gw = client.next_gw()
    cur_gw = client.current_gw()
    gws = list(range(gw, gw + HORIZON))

    # -- shared projections (the exact brief pipeline) --------------------
    raw = model.project_all(client, horizon=HORIZON)
    est = {pid: minutes.estimate_minutes(client, pid) for pid in raw}
    base = minutes.apply_minutes(raw, est)
    try:
        dc = defcon.DefConModel(client).corrections(list(base), est, horizon=HORIZON)
    except Exception:
        log.exception("defcon failed in trajectory")
        dc = {}
    final = {
        pid: [round(x + (dc.get(pid) or [0.0] * len(xs))[i], 3)
              for i, x in enumerate(xs)]
        for pid, xs in base.items()
    }

    rivals_on, league_note = league_usable(client, team_id, league_id)

    # -- me (always) ------------------------------------------------------
    me_state = rivals.fetch_my_state(client, team_id, cur_gw)
    me_total = float(me_state.total_points or 0)
    me = {
        "name": (me_state.name or "You").split()[0] or "You",
        "is_me": True,
        "start_total": round(me_total, 1),
        "standstill": _standstill_cum(me_state.squad, me_total, final, etype),
        "optimal": (_optimal_cum(client, team_id, me_total, final)
                    or _standstill_cum(me_state.squad, me_total, final, etype)),
    }

    entities = [me]
    capped = False
    if rivals_on:
        _, rows = rivals.fetch_league_entries(client, league_id)
        rival_rows = [r for r in rows if r.get("entry") != team_id]
        # Optimal solves are the expensive part: bound how many run.
        solve_ids = {r["entry"] for r in rival_rows[:MAX_OPTIMAL_RIVAL_SOLVES]}
        capped = len(rival_rows) > MAX_OPTIMAL_RIVAL_SOLVES
        for r in rival_rows:
            st = rivals.fetch_manager_state(client, r, cur_gw)
            total = float(st.total_points or 0)
            standstill = _standstill_cum(st.squad, total, final, etype)
            optimal = None
            if r["entry"] in solve_ids:
                optimal = _optimal_cum(client, r["entry"], total, final)
            entities.append({
                "name": (st.name or str(r["entry"])).split()[0],
                "is_me": False,
                "start_total": round(total, 1),
                "standstill": standstill,
                # Fall back to standstill when uncomputed, but flag it so
                # the band isn't drawn as if it were a real solve.
                "optimal": optimal or standstill,
                "optimal_computed": optimal is not None,
            })

    return {
        "gameweeks": gws,
        "rivals_available": rivals_on,
        "league_note": league_note if not rivals_on else None,
        "anchored_at_current_totals": True,
        "entities": entities,
        "rivals_capped": capped,
        "scenarios": {
            "1": {"label": "I stand still, rivals stand still",
                  "me": "standstill", "rivals": "standstill"},
            "2": {"label": "I make the recommended moves, rivals stand still",
                  "me": "optimal", "rivals": "standstill"},
            "3": {"label": "I move, rivals also move optimally",
                  "me": "optimal", "rivals": "optimal"},
        },
        "honesty_note": STAND_STILL_NOTE,
    }
