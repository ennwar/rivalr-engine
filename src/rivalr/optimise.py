"""Transfer optimisation: wraps the FPL-Optimization-Tools HiGHS MILP,
fed with OUR projections, under three objective modes.

Integration (verified against vendor source):
  - vendor root goes on sys.path (their modules import `paths`, `utils`,
    `dev.*` from the repo root)
  - projections are written as data/<datasource>.csv inside the vendor repo
    in the fplreview format its readers accept:
        ID,Name,Pos,Value,Team,{gw}_Pts,{gw}_xMins,...
    (Pos in G/D/M/F; ID = FPL element id; prices/teams are re-derived by
    the solver from bootstrap-static, so only ID/Pos/points/mins matter)
  - dev.solver.generate_team_json(team_id, options) builds the squad state
    from the public API; prep_data + solve_multi_period_fpl do the rest.

Standard constraints all live in their model: 15-squad 2/5/5/3, max 3 per
club, budget, valid XI, captain in XI, -4 hits, up to 5 banked FTs.

THREE OBJECTIVE MODES - implemented as reweightings of the Pts columns the
solver maximises. The solver itself is untouched; after solving, expected
points are recomputed from the RAW projections so the side-by-side
comparison is honest.

  points  raw xPts (what every competitor does)
  chase   boost players the target rival does NOT own, plus a variance
          bonus scaled by (1 - mini-league EO): differentials that can
          move rank
  defend  boost players the chasing rival owns and shields, penalise
          low-EO differentials: minimise divergence, cover their squad
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from . import vendors
from .fetch import FPLClient

log = logging.getLogger("rivalr.optimise")

POS_LETTER = {1: "G", 2: "D", 3: "M", 4: "F"}

# Mode weights (multiplicative on projected points).
CHASE_TARGET_BOOST = 0.15   # target doesn't own the player
CHASE_EO_BONUS = 0.10       # x (1 - min(EO, 1)): low mini-league EO
DEFEND_OWNED_BOOST = 0.15   # chasing rival owns the player
DEFEND_SHIELD_BOOST = 0.10  # classified SHIELD in the league pool
DEFEND_DIFF_PENALTY = 0.10  # low-EO differential the rival doesn't own

SOLVER_OVERRIDES = {
    "preseason": False,
    "team_data": "id",
    "weekly_hit_limit": 2,   # allow hits; -4 each is in the objective
    "secs": 120,
    "verbose": False,
    "single_solve": True,
    "randomized": False,
    # The vendor's pool-filter defaults assume fplreview-style optimistic
    # xMins; ours are conservative, so relax them or the cheap enablers
    # get filtered out and the squad becomes infeasible.
    "xmin_lb": 100,
    "ev_per_price_cutoff": 20,
    "keep_top_ev_percent": 15,
}

# Fallback when a solve is infeasible: open the pool filters completely.
RELAXED_FILTERS = {"xmin_lb": 0, "ev_per_price_cutoff": 0, "keep_top_ev_percent": 100}


# -- squad validation (also used by the test-suite, no solver needed) ------

SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}  # element_type -> required count


def validate_squad(
    squad: list[dict],
    budget: float = 100.0,
) -> list[str]:
    """Check a 15-man squad. Each entry: {id, element_type, team, cost}.
    Returns a list of violations (empty = valid)."""
    violations = []
    if len(squad) != 15:
        violations.append(f"squad size {len(squad)} != 15")
    counts: dict[int, int] = {}
    clubs: dict[int, int] = {}
    for p in squad:
        counts[p["element_type"]] = counts.get(p["element_type"], 0) + 1
        clubs[p["team"]] = clubs.get(p["team"], 0) + 1
    for etype, need in SQUAD_QUOTA.items():
        if counts.get(etype, 0) != need:
            violations.append(
                f"position {etype}: {counts.get(etype, 0)} != {need}"
            )
    for team, n in clubs.items():
        if n > 3:
            violations.append(f"club {team}: {n} players > 3")
    total = sum(p["cost"] for p in squad)
    if total > budget + 1e-6:
        violations.append(f"cost {total:.1f} > budget {budget:.1f}")
    return violations


def validate_xi(xi: list[dict], captain_id: int | None = None) -> list[str]:
    """Check a starting XI (subset of squad dicts) + captain membership."""
    violations = []
    if len(xi) != 11:
        violations.append(f"XI size {len(xi)} != 11")
    counts: dict[int, int] = {}
    for p in xi:
        counts[p["element_type"]] = counts.get(p["element_type"], 0) + 1
    if counts.get(1, 0) != 1:
        violations.append(f"goalkeepers in XI: {counts.get(1, 0)} != 1")
    if counts.get(2, 0) < 3:
        violations.append(f"defenders in XI: {counts.get(2, 0)} < 3")
    if counts.get(4, 0) < 1:
        violations.append(f"forwards in XI: {counts.get(4, 0)} < 1")
    if captain_id is not None and captain_id not in [p["id"] for p in xi]:
        violations.append(f"captain {captain_id} not in XI")
    return violations


# -- mode weighting --------------------------------------------------------


def mode_weight(
    pid: int,
    mode: str,
    target_squad: set[int],
    eo: dict[int, float],
    labels: dict[int, str],
    small_league: bool = False,
) -> float:
    """small_league: EO is too quantised to weight by (see rivals.py), so
    only the direct target-ownership terms apply."""
    e = min(eo.get(pid, 0.0), 1.0)
    if mode == "chase":
        w = 1.0 if small_league else 1.0 + CHASE_EO_BONUS * (1.0 - e)
        if pid not in target_squad:
            w += CHASE_TARGET_BOOST
        return w
    if mode == "defend":
        w = 1.0
        if pid in target_squad:
            w += DEFEND_OWNED_BOOST
        if small_league:
            return w
        if labels.get(pid) == "SHIELD":
            w += DEFEND_SHIELD_BOOST
        if e <= 0.15 and pid not in target_squad:
            w -= DEFEND_DIFF_PENALTY
        return w
    return 1.0


# -- solver integration ----------------------------------------------------


def _write_projection_csv(
    vendor_data_dir,
    datasource: str,
    elements: dict[int, dict],
    projections: dict[int, list[float]],
    xmins: dict[int, float],
    next_gw: int,
    horizon: int,
    weight_fn,
) -> None:
    rows = []
    gws = list(range(next_gw, min(39, next_gw + horizon)))
    for pid, xs in projections.items():
        el = elements.get(pid)
        if el is None or el["element_type"] not in POS_LETTER:
            continue
        row: dict[str, Any] = {
            "ID": pid,
            "Name": el["web_name"],
            "Pos": POS_LETTER[el["element_type"]],
            "Value": el["now_cost"] / 10.0,
            "Team": el["team"],
        }
        w = weight_fn(pid)
        for i, gw in enumerate(gws):
            pts = xs[i] if i < len(xs) else 0.0
            row[f"{gw}_Pts"] = round(pts * w, 3)
            row[f"{gw}_xMins"] = round(xmins.get(pid, 0.0), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = vendor_data_dir / f"{datasource}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("wrote %d-player projection file %s", len(df), out.name)


def _extract_plan(solution, next_gw: int, horizon: int,
                  projections: dict[int, list[float]]) -> dict:
    picks: pd.DataFrame = solution["picks"]
    week1 = picks[picks["week"] == next_gw]

    transfers_in = week1[week1["transfer_in"] > 0.5]["id"].astype(int).tolist()
    transfers_out = week1[week1["transfer_out"] > 0.5]["id"].astype(int).tolist()
    xi = week1[week1["lineup"] > 0.5]["id"].astype(int).tolist()
    bench_rows = week1[week1["bench"] >= 0].sort_values("bench")
    bench = bench_rows["id"].astype(int).tolist()
    cap_rows = week1[week1["captain"] > 0.5]
    captain = int(cap_rows["id"].iloc[0]) if len(cap_rows) else None

    # Honest expected points: recompute from RAW projections over the
    # horizon using the solved lineup/captain multipliers, so chase/defend
    # plans are comparable with the pure-points plan.
    raw_xp = 0.0
    hits = 0
    for _, r in picks.iterrows():
        gw_idx = int(r["week"]) - next_gw
        if 0 <= gw_idx < horizon and r["multiplier"] > 0:
            xs = projections.get(int(r["id"]), [])
            if gw_idx < len(xs):
                raw_xp += xs[gw_idx] * r["multiplier"]
    stats = solution.get("statistics", {})
    hits = sum(s.get("pt", 0) for s in stats.values())

    # Week-by-week plan over the whole horizon (planner view).
    weeks = []
    cum_xp = 0.0
    for gw_i in range(next_gw, next_gw + horizon):
        wk = picks[picks["week"] == gw_i]
        if wk.empty and gw_i != next_gw:
            continue
        stats_w = stats.get(gw_i, stats.get(str(gw_i), {}))
        w_in = wk[wk["transfer_in"] > 0.5]["id"].astype(int).tolist()
        w_out = wk[wk["transfer_out"] > 0.5]["id"].astype(int).tolist()
        w_squad = wk[wk["squad"] > 0.5]["id"].astype(int).tolist()
        w_xi = wk[wk["lineup"] > 0.5]["id"].astype(int).tolist()
        w_cap = wk[wk["captain"] > 0.5]["id"].astype(int).tolist()
        w_xp = 0.0
        for _, r in wk.iterrows():
            idx = int(r["week"]) - next_gw
            if r["multiplier"] > 0 and 0 <= idx < horizon:
                xs = projections.get(int(r["id"]), [])
                if idx < len(xs):
                    w_xp += xs[idx] * r["multiplier"]
        w_hits = int(stats_w.get("pt", 0))
        cum_xp += w_xp - 4 * w_hits
        weeks.append({
            "gw": gw_i,
            "transfers_in": w_in,
            "transfers_out": w_out,
            "banked": len(w_in) == 0,
            "free_transfers": stats_w.get("ft"),
            "hits": w_hits,
            "itb": stats_w.get("itb"),
            "chip": stats_w.get("chip") or None,
            "squad": w_squad,
            "xi": w_xi,
            "captain": w_cap[0] if w_cap else None,
            "xp": round(w_xp, 2),
            "cum_xp": round(cum_xp, 2),
        })

    return {
        "transfers_in": transfers_in,
        "transfers_out": transfers_out,
        "xi": xi,
        "bench": bench,
        "captain": captain,
        "expected_points": round(raw_xp - 4 * hits, 2),
        "hits": int(hits),
        "weeks": weeks,
        "solver_objective": round(solution.get("score", 0.0), 3),
        "summary": solution.get("summary", ""),
    }


def solve_all_modes(
    client: FPLClient,
    team_id: int,
    projections: dict[int, list[float]],
    rivals_report: dict,
    target_id: int | None,
    horizon: int = 5,
    xmins: dict[int, float] | None = None,
    requested_mode: str = "points",
    solver_options: dict | None = None,
) -> dict[str, dict | None]:
    """Solve points / chase / defend side by side.

    chase & defend require a target rival (entry id present in the rivals
    report); without one they come back None.
    """
    vendor = vendors.require_optimizer()
    from dev.solver import generate_team_json, prep_data, solve_multi_period_fpl  # noqa: E402

    vendor_data = vendor / "data"
    base_options = json.loads(
        (vendor_data / "comprehensive_settings.json").read_text(encoding="utf-8")
    )

    bootstrap = client.bootstrap()
    elements = {el["id"]: el for el in bootstrap["elements"]}
    next_gw = client.next_gw()

    if xmins is None:
        # crude default: 90 * chance-of-playing-ish; callers should pass
        # minutes.py estimates instead.
        xmins = {
            pid: 90.0 if elements[pid].get("status") == "a" else 45.0
            for pid in projections if pid in elements
        }

    eo = {int(k): v for k, v in rivals_report["effective_ownership"].items()}
    labels = {int(k): v for k, v in rivals_report["classification"].items()}
    small_league = rivals_report.get("small_league", False)
    if small_league:
        log.info(
            "small league (%s entries): EO weighting off, pairwise swings on",
            rivals_report.get("league_size", "?"),
        )
    target_squad: set[int] = set()
    target = None
    if target_id is not None:
        target = next(
            (r for r in rivals_report["rivals"] if r["entry_id"] == target_id), None
        )
        if target is None:
            log.error("target %s not in rivals report; chase/defend skipped", target_id)
        else:
            target_squad = set(target["shared"]) | set(target["their_differentials"])

    modes = ["points"] + (["chase", "defend"] if target else [])
    if target is None and requested_mode in ("chase", "defend"):
        # No target: the mode still solves, degraded to its league-wide
        # terms only (chase: low-EO variance bonus; defend: shield bonus).
        log.warning(
            "%s mode without a target: using league-EO terms only", requested_mode
        )
        modes.append(requested_mode)
    plans: dict[str, dict | None] = {"points": None, "chase": None, "defend": None}

    for mode in modes:
        datasource = f"rivalr_{mode}"
        _write_projection_csv(
            vendor_data, datasource, elements, projections, xmins,
            next_gw, horizon,
            weight_fn=lambda pid, m=mode: mode_weight(
                pid, m, target_squad, eo, labels, small_league
            ),
        )
        options = {
            **base_options,
            **SOLVER_OVERRIDES,
            "datasource": datasource,
            "team_id": team_id,
            "horizon": horizon,
            **(solver_options or {}),
        }
        try:
            try:
                my_data = generate_team_json(team_id, options)
                # The vendor's FT arithmetic has proven unreliable; use
                # our reconstruction from the entry's actual history.
                try:
                    from .rivals import free_transfers

                    ft = free_transfers(client, team_id, next_gw)
                    my_data["transfers"]["limit"] = ft
                    my_data["transfers"]["made"] = 0
                    log.info("free transfers going into gw%d: %d", next_gw, ft)
                except Exception:
                    log.warning("FT reconstruction failed; using vendor value",
                                exc_info=True)
            except Exception:
                # Pre-season / before GW1 picks exist: build from scratch
                # with a full budget, exactly like the vendor's preseason
                # mode.
                log.warning(
                    "no existing squad found for team %s (pre-season?) - "
                    "solving a fresh 100.0m draft instead", team_id,
                )
                options["preseason"] = True
                my_data = {
                    "picks": [], "chips": [],
                    "transfers": {"limit": None, "cost": 4, "bank": 1000, "value": 0},
                }
            try:
                data = prep_data(my_data, options)
                result = solve_multi_period_fpl(data, options)
            except Exception:
                log.warning(
                    "%s solve infeasible - retrying with pool filters open", mode
                )
                options = {**options, **RELAXED_FILTERS}
                data = prep_data(my_data, options)
                result = solve_multi_period_fpl(data, options)
            solution = result[0] if isinstance(result, list) else result
            plan = _extract_plan(solution, next_gw, horizon, projections)
            if small_league:
                plan["reasoning"] = _pairwise_reasoning(
                    plan, rivals_report, projections, elements
                )
            else:
                plan["reasoning"] = _reasoning(mode, plan, target, eo, labels, elements)
            if target:
                plan["swing_vs_target"] = _swing_vs_target(
                    plan, target, projections, elements
                )
            plans[mode] = plan
            log.info(
                "%s: %d transfer(s), xPts %.1f",
                mode, len(plan["transfers_in"]), plan["expected_points"],
            )
        except Exception:
            log.exception("solver failed for mode=%s", mode)
            plans[mode] = None
    return plans


def _reasoning(mode, plan, target, eo, labels, elements) -> list[str]:
    lines = []
    name = lambda pid: elements[pid]["web_name"] if pid in elements else f"#{pid}"
    for pid in plan["transfers_in"]:
        e = eo.get(pid, 0.0)
        label = labels.get(pid, "NEUTRAL")
        bits = [f"{name(pid)}: {label.lower()}", f"league EO {e * 100:.0f}%"]
        if target:
            owned = pid in (set(target["shared"]) | set(target["their_differentials"]))
            bits.append("target owns" if owned else "differential vs target")
        lines.append(", ".join(bits))
    if not plan["transfers_in"]:
        lines.append("best move is to bank the transfer")
    return lines


def _pairwise_reasoning(plan, rivals_report, projections, elements) -> list[str]:
    """Small-league mode: per transfer, the direct head-to-head swing
    against every named rival over the horizon, instead of EO%."""
    from .rivals import pairwise_transfer_gain, rival_squad

    name = lambda pid: elements[pid]["web_name"] if pid in elements else f"#{pid}"
    proj_sum = {pid: sum(xs) for pid, xs in projections.items()}
    rivals_list = rivals_report.get("rivals", [])

    lines: list[str] = []
    ins = plan["transfers_in"]
    outs = plan["transfers_out"]
    pairs = list(zip(outs, ins))
    pairs += [(None, p) for p in ins[len(outs):]]   # draft mode: no outs
    per_rival_total: dict[str, float] = {}
    for pid_out, pid_in in pairs:
        bits = []
        for r in rivals_list:
            first_name = r["name"].split()[0] if r["name"] else str(r["entry_id"])
            g = pairwise_transfer_gain(pid_in, pid_out, rival_squad(r), proj_sum)
            per_rival_total[first_name] = per_rival_total.get(first_name, 0.0) + g
            owns = pid_in in rival_squad(r)
            bits.append(f"{g:+.1f} vs {first_name}" + (" (owns)" if owns else ""))
        label = f"IN {name(pid_in)}" + (f" for {name(pid_out)}" if pid_out else "")
        lines.append(f"{label}: " + ", ".join(bits))
    if per_rival_total and len(pairs) > 1:
        lines.append(
            "total swing: "
            + ", ".join(f"{v:+.1f} vs {k}" for k, v in per_rival_total.items())
        )
    if not ins:
        lines.append("best move is to bank the transfer")
    return lines


def _swing_vs_target(plan, target, projections, elements) -> float:
    """Next-GW expected point swing vs the target rival: my solved XI +
    captain against their current XI + captain. A crude rank-movement
    proxy - positive means the gap is expected to close/extend in my favour."""
    def xp(pid):
        xs = projections.get(pid, [])
        return xs[0] if xs else 0.0

    mine = sum(xp(p) for p in plan["xi"])
    if plan["captain"]:
        mine += xp(plan["captain"])
    # target's starters aren't in the report explicitly; use shared+diffs
    # minus their bench? We only have their full 15 - approximate with the
    # top-11 by projection.
    their_squad = set(target["shared"]) | set(target["their_differentials"])
    top11 = sorted(their_squad, key=xp, reverse=True)[:11]
    theirs = sum(xp(p) for p in top11)
    if target.get("captain"):
        theirs += xp(target["captain"])
    return round(mine - theirs, 2)
