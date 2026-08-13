"""Squad-constraint validation and objective-mode weighting."""

import pytest

from rivalr.optimise import (
    CHASE_EO_BONUS,
    CHASE_TARGET_BOOST,
    DEFEND_DIFF_PENALTY,
    DEFEND_OWNED_BOOST,
    DEFEND_SHIELD_BOOST,
    mode_weight,
    validate_squad,
    validate_xi,
)


def make_squad():
    """A legal 15: 2 GK, 5 DEF, 5 MID, 3 FWD, <=3/club, 99.5 total."""
    squad = []
    pid = 1
    spec = [(1, 2), (2, 5), (3, 5), (4, 3)]  # (element_type, count)
    costs = iter([4.5, 4.0, 5.5, 5.0, 4.5, 4.5, 4.0, 8.0, 7.5, 6.5, 5.0, 5.0, 9.0, 7.0, 4.5])
    team = 1
    for etype, count in spec:
        for _ in range(count):
            squad.append({"id": pid, "element_type": etype, "team": team, "cost": next(costs)})
            pid += 1
            team = team % 15 + 1  # spread over clubs
    return squad


def test_legal_squad_passes():
    assert validate_squad(make_squad(), budget=100.0) == []


def test_squad_size_enforced():
    squad = make_squad()[:14]
    assert any("squad size" in v for v in validate_squad(squad))


def test_position_quota_enforced():
    squad = make_squad()
    squad[0]["element_type"] = 2  # 1 GK, 6 DEF
    violations = validate_squad(squad)
    assert any("position 1" in v for v in violations)
    assert any("position 2" in v for v in violations)


def test_max_three_per_club():
    squad = make_squad()
    for p in squad[:4]:
        p["team"] = 99
    assert any("club 99" in v for v in validate_squad(squad))


def test_budget_enforced():
    squad = make_squad()
    squad[0]["cost"] += 50.0
    assert any("budget" in v for v in validate_squad(squad, budget=100.0))


def test_valid_xi():
    squad = make_squad()
    # 1 GK, 4 DEF, 4 MID, 2 FWD
    xi = [squad[0]] + squad[2:6] + squad[7:11] + squad[12:14]
    assert len(xi) == 11
    assert validate_xi(xi, captain_id=xi[5]["id"]) == []


def test_xi_needs_one_gk():
    squad = make_squad()
    xi = squad[2:13]  # no GK
    assert any("goalkeepers" in v for v in validate_xi(xi))


def test_captain_must_start():
    squad = make_squad()
    xi = [squad[0]] + squad[2:6] + squad[7:11] + squad[12:14]
    bench_player = squad[1]
    assert any("captain" in v for v in validate_xi(xi, captain_id=bench_player["id"]))


# -- objective mode weights ------------------------------------------------


def test_points_mode_is_identity():
    assert mode_weight(1, "points", {2, 3}, {1: 0.9}, {1: "SHIELD"}) == 1.0


def test_chase_boosts_differentials_from_target():
    target_squad = {10}
    eo = {10: 1.0, 20: 0.0}
    # target owns 10 with max EO: no boost at all
    assert mode_weight(10, "chase", target_squad, eo, {}) == pytest.approx(1.0)
    # 20 is unowned everywhere: full target boost + full EO bonus
    assert mode_weight(20, "chase", target_squad, eo, {}) == pytest.approx(
        1.0 + CHASE_TARGET_BOOST + CHASE_EO_BONUS
    )


def test_chase_eo_bonus_scales_with_eo():
    lo = mode_weight(1, "chase", set(), {1: 0.1}, {})
    hi = mode_weight(2, "chase", set(), {2: 0.9}, {})
    assert lo > hi  # lower EO -> bigger variance bonus


def test_defend_prefers_target_owned_and_shields():
    target_squad = {10}
    labels = {10: "SHIELD", 20: "SWORD"}
    eo = {10: 0.8, 20: 0.05}
    w_shield = mode_weight(10, "defend", target_squad, eo, labels)
    w_diff = mode_weight(20, "defend", target_squad, eo, labels)
    assert w_shield == pytest.approx(1.0 + DEFEND_OWNED_BOOST + DEFEND_SHIELD_BOOST)
    assert w_diff == pytest.approx(1.0 - DEFEND_DIFF_PENALTY)
    assert w_shield > 1.0 > w_diff
