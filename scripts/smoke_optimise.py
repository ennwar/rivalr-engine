"""Optimiser smoke test: real projections, points mode, HiGHS solve.

    uv run python scripts/smoke_optimise.py [team_id]
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from rivalr.fetch import FPLClient
from rivalr.model import OpenFPLModel
from rivalr.minutes import estimate_minutes
from rivalr.optimise import solve_all_modes, validate_squad, validate_xi


def main() -> int:
    team_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    client = FPLClient()
    bootstrap = client.bootstrap()
    elements = {el["id"]: el for el in bootstrap["elements"]}

    # Projections for a meaningful pool: top 450 by ownership.
    pool = sorted(
        bootstrap["elements"], key=lambda el: -float(el["selected_by_percent"])
    )[:450]
    pool_ids = [el["id"] for el in pool]

    m = OpenFPLModel(client)
    horizon = 5
    proj = m.project_all(horizon=horizon, pool=pool_ids)
    est = {pid: estimate_minutes(client, pid) for pid in pool_ids}
    proj = {
        pid: [round(x * est[pid].factor, 3) for x in xs] for pid, xs in proj.items()
    }
    xmins = {pid: est[pid].expected_minutes for pid in pool_ids}

    fake_rivals_report = {
        "effective_ownership": {},
        "classification": {},
        "rivals": [],
    }
    plans = solve_all_modes(
        client=client,
        team_id=team_id,
        projections=proj,
        rivals_report=fake_rivals_report,
        target_id=None,
        horizon=horizon,
        xmins=xmins,
    )
    plan = plans["points"]
    if plan is None:
        print("SOLVER FAILED")
        return 1

    squad_ids = set(plan["xi"]) | set(plan["bench"])
    squad = [
        {
            "id": pid,
            "element_type": elements[pid]["element_type"],
            "team": elements[pid]["team"],
            "cost": elements[pid]["now_cost"] / 10.0,
        }
        for pid in squad_ids
    ]
    xi = [p for p in squad if p["id"] in set(plan["xi"])]
    print("\nsquad violations:", validate_squad(squad, budget=100.0) or "none")
    print("XI violations:", validate_xi(xi, captain_id=plan["captain"]) or "none")

    name = lambda pid: elements[pid]["web_name"]
    print(f"\nPOINTS plan: xPts {plan['expected_points']}, hits {plan['hits']}")
    print("XI:", ", ".join(name(p) for p in plan["xi"]))
    print("bench:", ", ".join(name(p) for p in plan["bench"]))
    print("captain:", name(plan["captain"]) if plan["captain"] else "-")
    return 0


if __name__ == "__main__":
    sys.exit(main())
