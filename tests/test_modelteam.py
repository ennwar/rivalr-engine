"""Autonomous model team: XI legality, auto-subs, FT/hit sequence."""

from rivalr.modelteam import _auto_subs, _best_xi


def test_best_xi_is_legal_and_greedy():
    # 2 GK, 5 DEF, 5 MID, 3 FWD; projections favour mids
    etype = {**{i: 1 for i in (1, 2)}, **{i: 2 for i in range(3, 8)},
             **{i: 3 for i in range(8, 13)}, **{i: 4 for i in range(13, 16)}}
    proj = {i: float(i) for i in range(1, 16)}  # higher id = better
    xi = _best_xi(list(range(1, 16)), proj, etype)
    assert len(xi) == 11
    counts = {p: 0 for p in (1, 2, 3, 4)}
    for pid in xi:
        counts[etype[pid]] += 1
    assert counts[1] == 1
    assert 3 <= counts[2] <= 5 and 2 <= counts[3] <= 5 and 1 <= counts[4] <= 3
    assert 2 in xi  # the better GK (proj 2 > 1)
    assert 15 in xi  # best forward always in


def test_auto_subs_swap_zero_minute_players():
    etype = {**{i: 1 for i in (1, 2)}, **{i: 2 for i in range(3, 8)},
             **{i: 3 for i in range(8, 13)}, **{i: 4 for i in range(13, 16)}}
    proj = {i: float(i) for i in range(1, 16)}
    squad = list(range(1, 16))
    xi = _best_xi(squad, proj, etype)
    # best forward (15) didn't play; bench forward 13 did
    minutes = {i: 90 for i in range(1, 16)}
    minutes[15] = 0
    final = _auto_subs(xi, squad, minutes, proj, etype)
    assert 15 not in final
    assert len(final) == 11
    counts = {}
    for pid in final:
        counts[etype[pid]] = counts.get(etype[pid], 0) + 1
    assert counts[1] == 1 and counts.get(4, 0) >= 1


def test_auto_subs_keep_player_when_no_valid_bench():
    etype = {1: 1, **{i: 2 for i in (2, 3, 4)}, **{i: 3 for i in range(5, 11)},
             11: 4, 12: 1, 13: 2, 14: 3, 15: 4}
    proj = {i: 10.0 - i * 0.1 for i in range(1, 16)}
    squad = list(range(1, 16))
    xi = _best_xi(squad, proj, etype)
    minutes = {i: 0 for i in range(1, 16)}  # nobody played at all
    final = _auto_subs(xi, squad, minutes, proj, etype)
    assert len(final) == 11  # no swaps possible, XI unchanged
    assert final == xi
