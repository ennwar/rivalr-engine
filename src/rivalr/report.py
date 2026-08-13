"""Gameweek brief generator.

    python -m rivalr.report --team 123456 --league 98765 --mode chase --target 555555

Plain text, short lines, no markdown tables — formatted so the same string
can later be dropped straight into a Telegram message.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import ledger, minutes, model, optimise, rivals
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
    projections = minutes.apply_minutes(raw_projections, est)
    next_gw_proj = {pid: xs[0] for pid, xs in projections.items() if xs}

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
    # filtering is a solver concern, never a scoring concern).
    chosen = plans.get(mode) or plans["points"]
    ledger.record_predictions(
        gw,
        ledger.full_coverage(projections, bootstrap["elements"]),
        recommendation={
            "team_id": team_id,
            "mode": mode,
            "target_id": target_id,
            "transfers_in": chosen.get("transfers_in", []),
            "transfers_out": chosen.get("transfers_out", []),
            "captain": chosen.get("captain"),
        },
    )

    # 5. Render.
    return render_brief(
        gw, rep, plans, mode, target_id, elements, next_gw_proj, est
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
) -> str:
    L: list[str] = []
    P = lambda pid: _pname(elements, pid)

    L.append(f"RIVALR BRIEF - GW{gw}")
    L.append(f"{rep['league_name']}")
    L.append(_hr("="))

    # My squad + projections + flags
    L.append("MY SQUAD (next-GW xPts)")
    for pid in sorted(rep["my_squad"], key=lambda p: -proj.get(p, 0)):
        e = est.get(pid)
        flag = f"  ! {'; '.join(e.flags)}" if e and e.flags else ""
        cap = " (C)" if pid == rep.get("my_captain") else ""
        L.append(f"  {P(pid):<18}{proj.get(pid, 0):>5.2f}{cap}{flag}")
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
                 f"  xPts {proj.get(pid, 0):.2f}")
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
            L.append(f"  OUT {P(pid_out):<16} IN {P(pid_in)}")
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
