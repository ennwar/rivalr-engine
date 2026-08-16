"""Machine-readable gameweek brief (JSON) for the web API.

Reuses exactly the same components as the text brief (report.py) - the
CLI path is untouched. Rival intelligence is wrapped: a rivals failure
degrades to rivals=null plus a warning, never an exception.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import defcon, minutes, model, optimise, rivals, uncertainty
from .fetch import FPLClient

log = logging.getLogger("rivalr.briefdata")

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _player(
    pid: int,
    elements: dict[int, dict],
    teams: dict[int, str],
    final: dict[int, list[float]],
    base: dict[int, list[float]],
    dc: dict[int, list[float]],
    flags: dict[int, list[str]],
) -> dict:
    el = elements.get(pid, {})
    f = final.get(pid) or [0.0]
    b = base.get(pid) or [0.0]
    d = dc.get(pid) or [0.0]
    return {
        "id": pid,
        "name": el.get("web_name", f"#{pid}"),
        "club": teams.get(el.get("team"), "?"),
        "position": POS.get(el.get("element_type"), "?"),
        "price": el.get("now_cost", 0) / 10.0,
        "projection": round(f[0], 2),
        "base": round(b[0], 2),
        "defcon": round(d[0], 2),
        "flags": flags.get(pid, []),
        "status": el.get("status"),
        "news": el.get("news") or "",
        "chance_of_playing": el.get("chance_of_playing_next_round"),
    }


def build_plan_json(
    client: FPLClient,
    team_id: int,
    league_id: int,
    horizon: int = 5,
    locked: list[int] | None = None,
    banned: list[int] | None = None,
) -> dict:
    """Week-by-week transfer plan (points mode) with lock-in/lock-out
    constraints passed straight to the MILP."""
    bootstrap = client.bootstrap()
    elements = {el["id"]: el for el in bootstrap["elements"]}
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    gw = client.next_gw()

    raw = model.project_all(client, horizon=horizon)
    est = {pid: minutes.estimate_minutes(client, pid) for pid in raw}
    base = minutes.apply_minutes(raw, est)
    try:
        dc_corr = defcon.DefConModel(client).corrections(
            list(base), est, horizon=horizon
        )
    except Exception:
        log.exception("defcon layer failed in planner")
        dc_corr = {}
    final = {
        pid: [
            round(x + (dc_corr.get(pid) or [0.0] * len(xs))[i], 3)
            for i, x in enumerate(xs)
        ]
        for pid, xs in base.items()
    }

    stub_rep = {"effective_ownership": {}, "classification": {},
                "rivals": [], "small_league": True, "league_size": 0}
    plans = optimise.solve_all_modes(
        client=client,
        team_id=team_id,
        projections=final,
        rivals_report=stub_rep,
        target_id=None,
        horizon=horizon,
        xmins={pid: e.expected_minutes for pid, e in est.items()},
        solver_options={
            "locked": locked or [],
            "banned": banned or [],
        },
    )
    plan = plans.get("points")
    if plan is None:
        raise RuntimeError(
            "solver found no feasible plan - locks may be contradictory "
            "(e.g. too many locked players for the budget or formation)"
        )

    def mini(pid: int | None) -> dict | None:
        if pid is None:
            return None
        el = elements.get(pid, {})
        return {
            "id": pid,
            "name": el.get("web_name", f"#{pid}"),
            "club": teams.get(el.get("team"), "?"),
            "position": POS.get(el.get("element_type"), "?"),
            "price": el.get("now_cost", 0) / 10.0,
            "projection": round((final.get(pid) or [0.0])[0], 2),
        }

    weeks = []
    for w in plan.get("weeks", []):
        pairs = list(zip(w["transfers_out"], w["transfers_in"]))
        pairs += [(None, p) for p in w["transfers_in"][len(w["transfers_out"]):]]
        weeks.append({
            "gw": w["gw"],
            "transfers": [
                {"out": mini(o), "in": mini(i)} for o, i in pairs
            ],
            "banked": w["banked"],
            "free_transfers": w["free_transfers"],
            "hits": w["hits"],
            "itb": w["itb"],
            "chip": w["chip"],
            "captain": mini(w["captain"]),
            "squad": [mini(p) for p in sorted(
                w["squad"], key=lambda p: elements.get(p, {}).get("element_type", 9)
            )],
            "xp": w["xp"],
            "cum_xp": w["cum_xp"],
        })

    return {
        "gameweek": gw,
        "horizon": horizon,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "locked": locked or [],
        "banned": banned or [],
        "total_xp": plan.get("expected_points"),
        "weeks": weeks,
    }


def build_brief_json(
    client: FPLClient,
    team_id: int,
    league_id: int,
    mode: str = "points",
    target_id: int | None = None,
    horizon: int = 5,
) -> dict:
    warnings: list[str] = []
    bootstrap = client.bootstrap()
    elements = {el["id"]: el for el in bootstrap["elements"]}
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    gw = client.next_gw()
    deadline = next(
        ev["deadline_time"] for ev in bootstrap["events"] if ev["id"] == gw
    )

    # -- projections: base -> defcon -> final -----------------------------
    raw = model.project_all(client, horizon=horizon)
    est = {pid: minutes.estimate_minutes(client, pid) for pid in raw}
    base = minutes.apply_minutes(raw, est)
    try:
        dc_corr = defcon.DefConModel(client).corrections(
            list(base), est, horizon=horizon
        )
    except Exception:
        log.exception("defcon layer failed")
        warnings.append("DefCon layer failed; projections are base-only")
        dc_corr = {}
    final = {
        pid: [
            round(x + (dc_corr.get(pid) or [0.0] * len(xs))[i], 3)
            for i, x in enumerate(xs)
        ]
        for pid, xs in base.items()
    }
    next_gw_proj = {pid: xs[0] for pid, xs in final.items() if xs}

    # -- flags -------------------------------------------------------------
    flags: dict[int, list[str]] = {}
    for pid, xs in base.items():
        if xs and model.is_low_confidence(
            xs[0], est[pid].factor if pid in est else 1.0
        ):
            flags.setdefault(pid, []).append("LOW_CONF")
    try:
        for pid, info in uncertainty.player_flags(client).items():
            flags.setdefault(pid, []).extend(info.get("kinds", []))
    except Exception:
        log.exception("uncertainty flags failed")
        warnings.append("uncertainty flags unavailable")

    # -- rivals (wrapped: never a 500) ------------------------------------
    rep = None
    rivals_block = None
    try:
        rep = rivals.build_rivals_report(
            client, team_id, league_id, projections=next_gw_proj
        )
        labels = {int(k): v for k, v in rep["classification"].items()}
        rivals_block = []
        for r in rep["rivals"]:
            their = sorted(rivals.rival_squad(r))
            rivals_block.append({
                "entry_id": r["entry_id"],
                "name": r["name"],
                "team_name": r["team_name"],
                "rank": r["rank"],
                "points": r["total_points"],
                "chips_left": r["chips_left"],
                "overlap_pct": r["overlap_pct"],
                "differentials": [
                    _player(p, elements, teams, final, base, dc_corr, flags)
                    for p in r["their_differentials"]
                ],
                "shields": [
                    _player(p, elements, teams, final, base, dc_corr, flags)
                    for p in their if labels.get(p) == "SHIELD"
                ],
                "swords": [
                    _player(p, elements, teams, final, base, dc_corr, flags)
                    for p in their if labels.get(p) == "SWORD"
                ],
            })
    except Exception as exc:
        log.exception("rivals block failed")
        warnings.append(f"rival intelligence unavailable: {exc!r}")
        rep = {
            "effective_ownership": {}, "classification": {}, "rivals": [],
            "small_league": True, "league_size": 0,
            "my_squad": [], "my_captain": None,
        }

    # -- solve -------------------------------------------------------------
    plans = optimise.solve_all_modes(
        client=client,
        team_id=team_id,
        projections=final,
        rivals_report=rep,
        target_id=target_id,
        horizon=horizon,
        xmins={pid: e.expected_minutes for pid, e in est.items()},
        requested_mode=mode,
    )
    chosen = plans.get(mode) or plans.get("points")
    if chosen is None:
        warnings.append("solver failed for all modes; no transfer plan")

    # -- assemble ----------------------------------------------------------
    squad = [
        _player(pid, elements, teams, final, base, dc_corr, flags)
        for pid in sorted(
            rep.get("my_squad", []), key=lambda p: -next_gw_proj.get(p, 0)
        )
    ]
    if not squad:
        warnings.append(
            "current squad not visible (pre-season or picks unavailable); "
            "transfers contain the recommended draft"
        )

    transfers = []
    if chosen:
        ins = chosen.get("transfers_in", [])
        outs = chosen.get("transfers_out", [])
        rival_names = {
            r["entry_id"]: r["name"].split()[0]
            for r in (rep.get("rivals") or [])
        }
        proj_sum = {pid: sum(xs) for pid, xs in final.items()}
        pairs = list(zip(outs, ins)) + [(None, p) for p in ins[len(outs):]]
        for pid_out, pid_in in pairs:
            swings = {}
            for r in rep.get("rivals") or []:
                swings[rival_names[r["entry_id"]]] = round(
                    rivals.pairwise_transfer_gain(
                        pid_in, pid_out, rivals.rival_squad(r), proj_sum
                    ), 2,
                )
            gain = proj_sum.get(pid_in, 0.0) - (
                proj_sum.get(pid_out, 0.0) if pid_out else 0.0
            )
            transfers.append({
                "out": _player(pid_out, elements, teams, final, base,
                               dc_corr, flags) if pid_out else None,
                "in": _player(pid_in, elements, teams, final, base,
                              dc_corr, flags),
                "net_gain": round(gain, 2),
                "hits": chosen.get("hits", 0),
                "swings": swings,
                "flags": flags.get(pid_in, []),
            })

    captain = None
    if chosen and chosen.get("captain"):
        cap = chosen["captain"]
        reasoning = (
            f"highest projected scorer in the XI "
            f"({next_gw_proj.get(cap, 0):.2f} xPts)"
        )
        target = next(
            (r for r in (rep.get("rivals") or [])
             if target_id and r["entry_id"] == target_id), None,
        )
        if target and target.get("captain"):
            if target["captain"] == cap:
                reasoning += "; same pick as target - captaincy neutralised"
            else:
                reasoning += (
                    f"; differs from target's captain - swing in play"
                )
        captain = {
            "player": _player(cap, elements, teams, final, base, dc_corr, flags),
            "reasoning": reasoning,
        }

    n_lc = sum(1 for t in transfers if "LOW_CONF" in t["flags"])
    if n_lc:
        warnings.append(f"{n_lc} incoming picks rest on LOW_CONF projections")

    return {
        "gameweek": gw,
        "deadline": deadline,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "target_id": target_id,
        "expected_points_horizon": chosen.get("expected_points") if chosen else None,
        "squad": squad,
        "captain": captain,
        "transfers": transfers,
        "rivals": rivals_block,
        "warnings": warnings,
    }
