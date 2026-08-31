"""Monte Carlo mini-league simulation over projection distributions.

All probability maths happens HERE, in the engine - the assistant's LLM
only receives the computed numbers.

Model, stated openly (also returned in the payload's assumptions):
  - each player's gameweek score ~ Normal(mu, SIGMA) truncated at a
    floor of -2, with mu = our per-GW projection; SIGMA is calibrated
    to our live per-player scoring error (~2.5 points)
  - players are sampled ONCE per simulation, so squads that share a
    player share his outcome - overlap correlation is handled naturally
  - squads, captains and XIs are frozen as of today (no future
    transfers or chips simulated)
  - beyond our projection horizon the last projected gameweek's mu is
    carried flat
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("rivalr.simulate")

SIGMA = 2.5
FLOOR = -2.0
N_SIMS = 4000


def _team_gw_mu(squad: list[int], proj_at, captain: int | None) -> tuple[list[int], int]:
    """(fixed XI by mu, captain pid) for one team at one gameweek."""
    ranked = sorted(squad, key=lambda p: -proj_at(p))
    xi = ranked[:11]
    cap = captain if captain in xi else (xi[0] if xi else None)
    return xi, cap


def finish_above(
    my_name: str,
    totals: dict[str, int],
    squads: dict[str, list[int]],
    captains: dict[str, int | None],
    projections: dict[int, list[float]],
    gws_to_sim: int,
    n_sims: int = N_SIMS,
    seed: int = 7,
) -> dict:
    """P(my final total > each rival's) after gws_to_sim more gameweeks.

    totals: current actual points per manager (model team included fine).
    projections: pid -> per-gw list starting at the next gameweek.
    """
    rng = np.random.default_rng(seed)
    horizon = max(len(v) for v in projections.values()) if projections else 0

    def proj_at_gw(g: int):
        def f(pid: int) -> float:
            xs = projections.get(pid) or [0.0]
            return xs[min(g, len(xs) - 1)]  # flat carry past horizon
        return f

    players = sorted({p for s in squads.values() for p in s})
    p_index = {p: i for i, p in enumerate(players)}

    sim_totals = {name: np.full(n_sims, float(totals.get(name, 0)))
                  for name in squads}
    for g in range(gws_to_sim):
        pa = proj_at_gw(g)
        mus = np.array([pa(p) for p in players])
        draws = rng.normal(mus, SIGMA, size=(n_sims, len(players)))
        np.clip(draws, FLOOR, None, out=draws)
        for name, squad in squads.items():
            xi, cap = _team_gw_mu(squad, pa, captains.get(name))
            idx = [p_index[p] for p in xi]
            team = draws[:, idx].sum(axis=1)
            if cap is not None:
                team += draws[:, p_index[cap]]
            sim_totals[name] += team

    mine = sim_totals[my_name]
    out = {}
    for name, arr in sim_totals.items():
        if name == my_name:
            continue
        p = float((mine > arr).mean() + 0.5 * (mine == arr).mean())
        out[name] = {
            "p_finish_above": round(p, 3),
            "their_current": totals.get(name, 0),
            "their_expected_final": round(float(arr.mean()), 1),
        }
    return {
        "my_current": totals.get(my_name, 0),
        "my_expected_final": round(float(mine.mean()), 1),
        "vs": out,
        "assumptions": (
            f"{n_sims} simulations, player scores ~ Normal(projection, "
            f"{SIGMA}) floored at {FLOOR:g}; shared players correlated; "
            "squads/captains frozen as of now (no future transfers or "
            "chips); projections carried flat beyond our horizon"
        ),
        "n_sims": n_sims,
        "gws_simulated": gws_to_sim,
    }
