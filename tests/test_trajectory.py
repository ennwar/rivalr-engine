"""Trajectory building blocks: best-XI points + cumulative anchoring."""

from rivalr import trajectory


def test_gw_points_picks_best_xi_and_doubles_captain():
    # 1 GK, 4 DEF, 4 MID, 3 FWD (a legal 15); captain = top projection
    etype = {}
    proj = {}
    squad = []
    pid = 1
    for et, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
        for _ in range(n):
            etype[pid] = et
            proj[pid] = float(pid)  # higher id = higher projection
            squad.append(pid)
            pid += 1
    pts = trajectory._gw_points(squad, proj, etype)
    # best XI is the top legal formation; captain (highest proj in XI)
    # is counted twice. Must exceed a plain sum of any 11.
    assert pts > sum(sorted(proj.values())[-11:])  # captain double lifts it


def test_standstill_cum_anchors_at_current_total():
    etype = {1: 1, 2: 2, 3: 3, 4: 4}
    squad = [1, 2, 3, 4]
    final = {p: [1.0] * 5 for p in squad}
    cum = trajectory._standstill_cum(squad, start=100.0, final=final, etype=etype)
    assert len(cum) == 5
    assert cum[0] > 100.0            # starts from the current total, adds GW
    assert cum == sorted(cum)         # cumulative, monotone non-decreasing


def test_empty_squad_scores_zero():
    assert trajectory._gw_points([], {}, {}) == 0.0
