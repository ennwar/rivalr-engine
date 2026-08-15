"""Gameweek brief generator.

    python -m rivalr.report --team 123456 --league 98765 --mode chase --target 555555

Plain text, short lines, no markdown tables — formatted so the same string
can later be dropped straight into a Telegram message.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import defcon, ledger, minutes, model, optimise, rivals, uncertainty
from .fetch import FPLClient

log = logging.getLogger("rivalr.report")

WIDTH = 46  # keep lines phone-narrow


def _hr(char: str = "-") -> str:
    return char * WIDTH


def _name_lookup(bootstrap: dict) -> dict[int, dict]:
    return {el["id"]: el for el in bootstrap["elements"]}


def _pname(elements: dict[int, dict], pid: int) -> str:
    el = elements.get(pid)
    return el["web_name"] if el else f"#{pid}"


def build_brief(
    client: FPLClient,
    team_id: int,
    league_id: int,
    mode: str,
    target_id: int | None,
    horizon: int = 5,
) -> str:
    bootstrap = client.bootstrap()
    elements = _name_lookup(bootstrap)
    gw = client.next_gw()

    # 1. Projections for every plausible player, minutes-adjusted.
    log.info("projecting %d-GW horizon...", horizon)
    raw_projections = model.project_all(client, horizon=horizon)
    est = {
        pid: minutes.estimate_minutes(client, pid)
        for pid in raw_projections
    }
    base = minutes.apply_minutes(raw_projections, est)

    # DefCon correction: additive, separate layer; base never overwritten.
    try:
        dc = defcon.DefConModel(client)
        dc_corr = dc.corrections(list(base), est, horizon=horizon)
    except Exception:
        log.exception("DEFCON LAYER FAILED - proceeding with base projections")
        dc_corr = {}
    projections = {
        pid: [
            round(x + (dc_corr.get(pid) or [0.0] * len(xs))[i], 3)
            for i, x in enumerate(xs)
        ]
        for pid, xs in base.items()
    }
    next_gw_proj = {pid: xs[0] for pid, xs in projections.items() if xs}
    dc_next = {pid: (dc_corr.get(pid) or [0.0])[0] for pid in projections}

    # Diagnostic: BASE projections within 0.5 of the minutes-adjusted
    # model floor (~1.7 for anyone who plays) carry no distinguishable
    # signal - the floor is an OpenFPL artifact, so the margin is
    # computed on the base, not the DefCon-corrected final.
    margins = {
        pid: round(
            model.confidence_margin(xs[0], est[pid].factor if pid in est else 1.0), 2
        )
        for pid, xs in base.items() if xs
    }
    low_conf = {
        pid for pid in margins
        if margins[pid] <= model.LOW_CONFIDENCE_MARGIN
    }

    # Manager-change uncertainty: projections for these clubs rest on the
    # previous regime's tactical patterns until 5 matches of 2026-27.
    try:
        mgr_flags = uncertainty.player_flags(client)
    except Exception:
        log.exception("manager-change flags unavailable")
        mgr_flags = {}

    # 2. Mini-league intelligence.
    log.info("building rivals report...")
    rep = rivals.build_rivals_report(
        client, team_id, league_id, projections=next_gw_proj, gw=client.current_gw()
    )
    rivals.write_rivals_report(rep)

    # 3. Optimise under all three objectives.
    log.info("solving transfer plans (points / chase / defend)...")
    plans = optimise.solve_all_modes(
        client=client,
        team_id=team_id,
        projections=projections,
        rivals_report=rep,
        target_id=target_id,
        horizon=horizon,
        xmins={pid: e.expected_minutes for pid, e in est.items()},
        requested_mode=mode,
    )

    # 4. Ledger snapshot (append-only, FULL bootstrap coverage - pool
    # filtering is a solver concern, never a scoring concern). Layers
    # logged separately: base (OpenFPL x minutes), defcon, final.
    chosen = plans.get(mode) or plans["points"]
    ledger.record_predictions(
        gw,
        ledger.full_coverage(projections, bootstrap["elements"]),
        layers={"base": base, "defcon": dc_corr},
        recommendation={
            "team_id": team_id,
            "mode": mode,
            "target_id": target_id,
            "transfers_in": chosen.get("transfers_in", []),
            "transfers_out": chosen.get("transfers_out", []),
            "captain": chosen.get("captain"),
            "low_confidence_ins": [
                p for p in chosen.get("transfers_in", []) if p in low_conf
            ],
            "manager_change_ins": [
                p for p in chosen.get("transfers_in", [])
                if "MGR_CHG" in mgr_flags.get(p, {}).get("kinds", [])
            ],
            "new_club_ins": [
                p for p in chosen.get("transfers_in", [])
                if "NEW_CLUB" in mgr_flags.get(p, {}).get("kinds", [])
            ],
        },
    )

    # 5. Render.
    return render_brief(
        gw, rep, plans, mode, target_id, elements, next_gw_proj, est,
        margins, low_conf, dc_next, mgr_flags,
    )


def render_brief(
    gw: int,
    rep: dict,
    plans: dict,
    mode: str,
    target_id: int | None,
    elements: dict[int, dict],
    proj: dict[int, float],
    est: dict[int, minutes.MinutesEstimate],
    margins: dict[int, float] | None = None,
    low_conf: set[int] | None = None,
    dc_next: dict[int, float] | None = None,
    mgr_flags: dict[int, dict] | None = None,
) -> str:
    L: list[str] = []
    P = lambda pid: _pname(elements, pid)
    margins = margins or {}
    low_conf = low_conf or set()

    dc_next = dc_next or {}
    mgr_flags = mgr_flags or {}

    def mc(pid: int) -> str:
        """Uncertainty markers: MGR_CHG (club changed manager) and/or
        NEW_CLUB (player transferred; rates from a different system)."""
        info = mgr_flags.get(pid)
        if not info:
            return ""
        return "".join(f" {k}" for k in info.get("kinds", []))

    def lc(pid: int) -> str:
        """LOW_CONFIDENCE marker: projection within 0.5 of the model's
        ~1.7 played-floor - a number the model can't distinguish from a
        blank (see docs/backtest_findings.md)."""
        if pid in low_conf:
            return f" LOW_CONF(m{margins.get(pid, 0):+.1f})"
        return ""

    def dc(pid: int) -> str:
        """DefCon share of the projection, shown when material."""
        v = dc_next.get(pid, 0.0)
        return f" (+{v:.1f}dc)" if v >= 0.3 else ""

    L.append(f"RIVALR BRIEF - GW{gw}")
    L.append(f"{rep['league_name']}")
    L.append(_hr("="))

    # Manager-change watchlist: which clubs' projections still rest on
    # last season's tactical assumptions.
    changed = {}
    n_new_club = 0
    for info in mgr_flags.values():
        if "team" in info:
            changed[info["team"]] = info
        if "NEW_CLUB" in info.get("kinds", []):
            n_new_club += 1
    if changed:
        L.append("MANAGER CHANGES (uncertain until 5 matches)")
        for team in sorted(changed):
            i = changed[team]
            L.append(f"  {team}: {i['new']} in ({i['out']} out), "
                     f"{i['matches_played']}/5 played")
        if n_new_club:
            L.append(f"  + {n_new_club} transferred players carry NEW_CLUB "
                     f"(rates from old system)")
        L.append(_hr())

    # My squad + projections + flags
    L.append("MY SQUAD (next-GW xPts)")
    for pid in sorted(rep["my_squad"], key=lambda p: -proj.get(p, 0)):
        e = est.get(pid)
        flag = f"  ! {'; '.join(e.flags)}" if e and e.flags else ""
        cap = " (C)" if pid == rep.get("my_captain") else ""
        L.append(f"  {P(pid):<18}{proj.get(pid, 0):>5.2f}{cap}{dc(pid)}{lc(pid)}{mc(pid)}{flag}")
    L.append(_hr())

    # League table with gaps
    L.append("LEAGUE TABLE")
    me_line = f"  {rep['my_rank']:>2}. (me)"
    printed_me = False
    for r in rep["rivals"]:
        if not printed_me and rep["my_rank"] < r["rank"]:
            L.append(f"{me_line}  {rep['my_total_points']} pts")
            printed_me = True
        tag = " <- target" if target_id and r["entry_id"] == target_id else ""
        L.append(
            f"  {r['rank']:>2}. {r['name'][:16]:<16} {r['total_points']}"
            f" ({r['gap']:+d}){tag}"
        )
    if not printed_me:
        L.append(f"{me_line}  {rep['my_total_points']} pts")
    L.append(_hr())

    # Shields missing / swords available
    L.append("SHIELDS I'M MISSING (high league EO)")
    eo = rep["effective_ownership"]
    for pid in rep["missing_shields"][:8] or []:
        L.append(f"  {P(pid):<18}EO {float(eo.get(str(pid), 0)) * 100:.0f}%"
                 f"  xPts {proj.get(pid, 0):.2f}")
    if not rep["missing_shields"]:
        L.append("  none - fully shielded")
    L.append("")
    L.append("SWORDS AVAILABLE (low EO, high xPts)")
    for pid in rep["available_swords"][:8]:
        L.append(f"  {P(pid):<18}EO {float(eo.get(str(pid), 0)) * 100:.0f}%"
                 f"  xPts {proj.get(pid, 0):.2f}{dc(pid)}{lc(pid)}{mc(pid)}")
    L.append(_hr())

    # Transfer plans side by side
    L.append("TRANSFER PLANS (5-GW horizon)")
    for m in ["points", "chase", "defend"]:
        plan = plans.get(m)
        marker = " *" if m == mode else ""
        L.append(f"[{m.upper()}]{marker}")
        if plan is None:
            L.append("  (not solved - no target rival given)"
                     if m != "points" else "  (solver failed)")
            continue
        ins = plan.get("transfers_in", [])
        outs = plan.get("transfers_out", [])
        if not ins:
            L.append("  roll the transfer")
        for pid_out, pid_in in zip(outs, ins):
            L.append(f"  OUT {P(pid_out):<16} IN {P(pid_in)}{lc(pid_in)}{mc(pid_in)}")
        for pid_in in ins[len(outs):]:  # draft mode: no outs
            L.append(f"  IN {P(pid_in)}{lc(pid_in)}{mc(pid_in)}")
        n_lc = sum(1 for p in ins if p in low_conf)
        if n_lc:
            L.append(f"  ! {n_lc} incoming pick(s) rest on LOW_CONF projections")
        n_mc = sum(1 for p in ins
                   if "MGR_CHG" in mgr_flags.get(p, {}).get("kinds", []))
        if n_mc:
            L.append(f"  ! {n_mc} incoming pick(s) rest on last season's "
                     f"tactics (MGR_CHG)")
        n_nc = sum(1 for p in ins
                   if "NEW_CLUB" in mgr_flags.get(p, {}).get("kinds", []))
        if n_nc:
            L.append(f"  ! {n_nc} incoming pick(s) transferred clubs this "
                     f"summer (NEW_CLUB)")
        L.append(f"  xPts horizon: {plan.get('expected_points', 0):.1f}"
                 f"  hits: {plan.get('hits', 0)}")
        if plan.get("reasoning"):
            for line in plan["reasoning"]:
                L.append(f"  - {line}")
        L.append("")

    # Captaincy vs target
    L.append("CAPTAINCY")
    chosen = plans.get(mode) or plans.get("points") or {}
    my_cap = chosen.get("captain")
    if my_cap:
        L.append(f"  recommended: {P(my_cap)} ({proj.get(my_cap, 0):.2f} xPts)")
    if target_id:
        target = next((r for r in rep["rivals"] if r["entry_id"] == target_id), None)
        if target and target.get("captain"):
            tcap = target["captain"]
            L.append(f"  target's likely captain: {P(tcap)}"
                     f" ({proj.get(tcap, 0):.2f} xPts)")
            if my_cap == tcap:
                L.append("  same pick - captaincy is neutralised")
            else:
                L.append("  different pick - captaincy swing in play")
    L.append(_hr("="))
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the gameweek brief")
    parser.add_argument("--team", type=int, required=True, help="your FPL entry id")
    parser.add_argument("--league", type=int, required=True, help="classic league id")
    parser.add_argument(
        "--mode", choices=["points", "chase", "defend"], default="points"
    )
    parser.add_argument(
        "--target", type=int, default=None,
        help="rival entry id for chase/defend modes",
    )
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    if args.mode in ("chase", "defend") and args.target is None:
        log.warning(
            "--mode %s without --target: solving with league-wide terms only "
            "(pick a target once rival picks are visible)", args.mode,
        )

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    client = FPLClient()
    brief = build_brief(
        client, args.team, args.league, args.mode, args.target, args.horizon
    )
    print(brief)


if __name__ == "__main__":
    main()
