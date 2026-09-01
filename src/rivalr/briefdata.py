"""Machine-readable gameweek brief (JSON) for the web API.

Reuses exactly the same components as the text brief (report.py) - the
CLI path is untouched. Rival intelligence is wrapped: a rivals failure
degrades to rivals=null plus a warning, never an exception.
"""

from __future__ import annotations

import logging
import re
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


def build_for_mode(
    client: FPLClient, team_id: int, league_id: int, mode: str,
    target: int | None = None,
) -> dict:
    """Dispatch a cached-pair mode string to the RIGHT builder. Plan
    modes ('plan:h5', 'plan:h5:hits') must never be rebuilt with the
    brief builder - that poisons the plan cache with brief-shaped
    payloads (this happened; see the pre-warm dispatcher)."""
    if mode.startswith("plan"):
        m = re.search(r"h(\d+)", mode)
        return build_plan_json(
            client, team_id, league_id,
            horizon=int(m.group(1)) if m else 5,
            allow_hits=":hits" in mode,
        )
    return build_brief_json(client, team_id, league_id, mode=mode,
                            target_id=target)


def build_plan_json(
    client: FPLClient,
    team_id: int,
    league_id: int,
    horizon: int = 5,
    locked: list[int] | None = None,
    banned: list[int] | None = None,
    allow_hits: bool = False,
) -> dict:
    """Week-by-week transfer plan (points mode) with lock-in/lock-out
    constraints passed straight to the MILP.

    Hits are OPT-IN: by default the solver may not take any -4s. When
    allowed, every hit week carries raw/penalty/net so the cost is never
    hidden."""
    from . import gameweek

    bootstrap = client.bootstrap()
    elements = {el["id"]: el for el in bootstrap["elements"]}
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    gw = client.next_gw()

    # Mid-gameweek context: the plan starts at the NEXT deadline, but if
    # the previous GW is still being played the UI must say so, and mark
    # outgoing players whose fixture hasn't happened yet.
    live_state = None
    teams_to_play: set[int] = set()
    cur = gameweek.state(client)["current"]
    if cur is not None and cur != gw and not gameweek.is_complete(client, cur):
        for f in client.fixtures():
            if f.get("event") == cur and not gameweek.fixture_played(f):
                teams_to_play.update((f["team_h"], f["team_a"]))
        live_state = {"gw": cur, "in_progress": True,
                      "teams_to_play": len(teams_to_play)}

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
            "weekly_hit_limit": 2 if allow_hits else 0,
            "hit_limit": None if allow_hits else 0,
        },
    )
    plan = plans.get("points")
    if plan is None:
        raise RuntimeError(
            "solver found no feasible plan - locks may be contradictory "
            "(e.g. too many locked players for the budget or formation)"
        )

    # Pre-deadline projections for the in-progress gameweek, so a sale of
    # a still-to-play player can show what his remaining fixture is worth
    # (points that accrue to the owner regardless of the sale).
    cur_gw_proj: dict[str, float] = {}
    if live_state is not None:
        try:
            from .store import make_store

            cur_gw_proj = make_store().gw_projections(live_state["gw"]) or {}
        except Exception:
            log.warning("current-gw projections unavailable", exc_info=True)

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
            "still_to_play": el.get("team") in teams_to_play or None,
            "pending_fixture_xpts": (
                cur_gw_proj.get(str(pid))
                if el.get("team") in teams_to_play else None
            ),
        }

    proj_sum_from = lambda pid, i: sum((final.get(pid) or [])[i:]) if pid else 0.0

    weeks = []
    for w in plan.get("weeks", []):
        pairs = list(zip(w["transfers_out"], w["transfers_in"]))
        pairs += [(None, p) for p in w["transfers_in"][len(w["transfers_out"]):]]
        # Hit justification: gain measured over the REMAINING horizon
        # from this week, minus the -4s. A hit only ever appears when
        # allow_hits is on, and never without its net shown.
        wk_idx = w["gw"] - gw
        raw_gain = round(sum(
            proj_sum_from(i_, wk_idx) - proj_sum_from(o_, wk_idx)
            for o_, i_ in pairs
        ), 1)
        penalty = 4 * w["hits"]
        weeks.append({
            "raw_gain": raw_gain,
            "hit_penalty": penalty,
            "net_gain": round(raw_gain - penalty, 1),
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

    try:
        ft_now = rivals.free_transfers(client, team_id, gw)
    except Exception:
        log.warning("free-transfer count unavailable", exc_info=True)
        ft_now = None

    return {
        "gameweek": gw,
        "horizon": horizon,
        "live": live_state,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "locked": locked or [],
        "banned": banned or [],
        "allow_hits": allow_hits,
        "free_transfers_now": ft_now,
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
    from . import gameweek

    warnings: list[str] = []
    bootstrap = client.bootstrap()
    elements = {el["id"]: el for el in bootstrap["elements"]}
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    gw = client.next_gw()
    deadline = next(
        ev["deadline_time"] for ev in bootstrap["events"] if ev["id"] == gw
    )

    # Mid-gameweek awareness: if the previous GW is still being played,
    # say so, and know which teams have not yet kicked off.
    live_state = None
    teams_to_play: set[int] = set()
    cur = gameweek.state(client)["current"]
    if cur is not None and cur != gw and not gameweek.is_complete(client, cur):
        for f in client.fixtures():
            if f.get("event") == cur and not gameweek.fixture_played(f):
                teams_to_play.update((f["team_h"], f["team_a"]))
        live_state = {"gw": cur, "in_progress": True,
                      "teams_to_play": len(teams_to_play)}

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
        n_horizon = min(5, horizon)
        rivals_block = []
        for r in rep["rivals"]:
            their = sorted(rivals.rival_squad(r))

            # -- chip war: BB/TC threat over the next 5 GWs from their
            # ACTUAL squad. Projections already sum double-gameweek
            # fixtures, so the argmax lands on their best (often double)
            # window naturally.
            bench_ids = r.get("bench_players", [])
            bb = tc = None
            if their:
                bb_by_gw = [
                    sum((final.get(p) or [0.0] * n_horizon)[i]
                        for p in bench_ids)
                    for i in range(n_horizon)
                ] if bench_ids else []
                if bb_by_gw and max(bb_by_gw) > 0:
                    i = bb_by_gw.index(max(bb_by_gw))
                    bb = {"best_gw": gw + i, "swing": round(bb_by_gw[i], 1)}
                tc_by_gw = []
                for i in range(n_horizon):
                    best_pid, best_val = None, 0.0
                    for p in their:
                        v = (final.get(p) or [0.0] * n_horizon)[i]
                        if v > best_val:
                            best_pid, best_val = p, v
                    tc_by_gw.append((best_val, best_pid))
                if tc_by_gw and max(v for v, _ in tc_by_gw) > 0:
                    i = max(range(n_horizon), key=lambda j: tc_by_gw[j][0])
                    val, pid = tc_by_gw[i]
                    tc = {
                        "best_gw": gw + i,
                        "swing": round(val, 1),
                        "player": elements.get(pid, {}).get("web_name", "?"),
                    }
            first_half_used = {
                c["name"] for c in r.get("chips_used", [])
                if c.get("event") and c["event"] <= rivals.FIRST_SET_EXPIRY_GW
            }
            chip_war = {
                "active_now": r.get("active_chip"),
                "chips_used": r.get("chips_used", []),
                "first_set_left": [
                    c for c in rivals.CHIP_SET if c not in first_half_used
                ] if gw <= rivals.FIRST_SET_EXPIRY_GW else [],
                "expiry_gw": rivals.FIRST_SET_EXPIRY_GW,
                "gws_to_expiry": max(0, rivals.FIRST_SET_EXPIRY_GW - gw + 1),
                "bench_boost": bb,
                "triple_captain": tc,
            }

            rivals_block.append({
                "chip_war": chip_war,
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
        # The brief is the recommendation surface: no -4s by default,
        # same as the planner. Hits live behind the planner's toggle.
        solver_options={"weekly_hit_limit": 0, "hit_limit": 0},
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
    if live_state is not None:
        my_to_play = [
            elements[pid]["web_name"] for pid in rep.get("my_squad", [])
            if pid in elements and elements[pid]["team"] in teams_to_play
        ]
        live_state["my_players_to_play"] = len(my_to_play)
        live_state["my_players_to_play_names"] = my_to_play
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

        # Captain board: top 5 XI candidates with their fixture context,
        # so a near-tie is visible instead of hidden behind one name.
        short = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
        fx_next = [f for f in client.fixtures() if f.get("event") == gw]

        def _fixture_info(team_id: int) -> list[dict]:
            out = []
            for f in fx_next:
                if f["team_h"] == team_id:
                    out.append({"opponent": short.get(f["team_a"], "?"),
                                "venue": "H",
                                "difficulty": f.get("team_h_difficulty")})
                elif f["team_a"] == team_id:
                    out.append({"opponent": short.get(f["team_h"], "?"),
                                "venue": "A",
                                "difficulty": f.get("team_a_difficulty")})
            return out

        ranked = sorted(
            chosen.get("xi") or [], key=lambda p: -next_gw_proj.get(p, 0.0),
        )[:5]
        board = []
        for i, pid in enumerate(ranked):
            el = elements.get(pid, {})
            nxt = next_gw_proj.get(ranked[i + 1], 0.0) if i + 1 < len(ranked) else None
            board.append({
                "id": pid,
                "name": el.get("web_name", f"#{pid}"),
                "projection": round(next_gw_proj.get(pid, 0.0), 2),
                "fixtures": _fixture_info(el.get("team")),
                "margin_over_next": (
                    round(next_gw_proj.get(pid, 0.0) - nxt, 2)
                    if nxt is not None else None
                ),
            })
        captain["board"] = board
        captain["close_call"] = (
            len(board) >= 2 and (board[0]["margin_over_next"] or 0) < 1.0
        )

    n_lc = sum(1 for t in transfers if "LOW_CONF" in t["flags"])
    if n_lc:
        warnings.append(f"{n_lc} incoming picks rest on LOW_CONF projections")

    # The autonomous model team as a fifth league entry (settled GWs
    # only, so mid-gameweek it's labelled "through GW n").
    model_standing = None
    try:
        from .store import make_store

        rows = make_store().model_rows()
        if rows:
            total = rows[-1]["total"]
            others = [(rep.get("my_total_points") or 0, "me")] + [
                (r["total_points"], r["name"]) for r in (rep.get("rivals") or [])
            ]
            model_standing = {
                "name": "The Model",
                "points": total,
                "rank_in_league": 1 + sum(1 for t, _ in others if t > total),
                "through_gw": rows[-1]["gw"],
                "hits_taken": sum(r.get("hits", 0) for r in rows),
            }
    except Exception:
        log.warning("model standing unavailable", exc_info=True)

    # -- the answer, up front: what to do, why, cost of doing nothing ------
    try:
        ft_now = rivals.free_transfers(client, team_id, gw)
    except Exception:
        ft_now = None
    action = None
    if chosen:
        cap_name = (
            elements.get(chosen.get("captain"), {}).get("web_name", "?")
            if chosen.get("captain") else "?"
        )
        real_moves = [t for t in transfers if t["out"] is not None]
        if not real_moves:
            if squad:
                headline = (
                    f"Bank your free transfer and captain {cap_name}."
                )
                why = (
                    "No available move beats holding this week; banking "
                    f"takes you to {min((ft_now or 0) + 1, 5)} free transfers "
                    "next week."
                )
                do_nothing = (
                    "Doing nothing IS this week's recommendation - just set "
                    f"{cap_name} as captain."
                )
            else:
                headline = f"Enter the recommended draft and captain {cap_name}."
                why = "Pre-season/no visible squad: the draft below is the plan."
                do_nothing = "No squad entered means zero points - enter a team."
        else:
            moves_txt = "; ".join(
                f"{t['out']['name']} → {t['in']['name']}" for t in real_moves
            )
            top = max(real_moves, key=lambda t: t["net_gain"])
            gain_total = round(sum(t["net_gain"] for t in real_moves), 1)
            hits = chosen.get("hits", 0)
            n = len(real_moves)
            headline = (
                f"Make {n} transfer{'s' if n > 1 else ''} ({moves_txt}) "
                f"and captain {cap_name}."
            )
            outs_still_to_play = [
                t["out"]["name"] for t in real_moves
                if t["out"] and elements.get(t["out"]["id"], {}).get("team")
                in teams_to_play
            ]
            why = (
                f"{top['in']['name']} projects {top['net_gain']:+.1f} over "
                f"{top['out']['name']} across the horizon"
                + (f"; total {gain_total:+.1f} xPts from the moves" if n > 1 else "")
                + (f", after {hits} hit(s) costing {4 * hits}" if hits else
                   f", using {'a free transfer' if n == 1 else 'free transfers'}")
                + "."
            )
            if outs_still_to_play and live_state:
                why += (
                    f" Note: {', '.join(outs_still_to_play)} still play(s) in "
                    f"GW{live_state['gw']} - transfers only take effect from "
                    f"GW{gw}, so you lose nothing from their remaining fixture."
                )
            do_nothing = (
                f"Doing nothing keeps "
                f"{min((ft_now or 1) + 1, 5)} free transfers for next week "
                f"but gives up ~{gain_total:.1f} projected points over the "
                f"horizon."
            )
        action = {"headline": headline, "why": why, "do_nothing": do_nothing}

    return {
        "action": action,
        "model_team": model_standing,
        "live": live_state,
        "free_transfers_now": ft_now,
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
