"""Mini-league EO math and classification on a synthetic 6-team league."""

from rivalr.rivals import (
    ManagerState,
    classify_pool,
    compare_squads,
    mini_league_eo,
)


def make_manager(entry_id: int, squad: list[int], captain: int, chip: str | None = None):
    return ManagerState(
        entry_id=entry_id,
        name=f"manager{entry_id}",
        team_name=f"team{entry_id}",
        rank=entry_id,
        total_points=100 - entry_id,
        squad=squad,
        starters=squad[:11],
        bench=squad[11:],
        captain=captain,
        active_chip=chip,
    )


def synthetic_league() -> list[ManagerState]:
    """6 managers. Player 1 owned by all six and captained by three (one on
    TC). Player 99 owned by exactly one. Player 50 owned by three."""
    base = list(range(2, 15))  # 13 shared filler players
    return [
        make_manager(1, [1, 50, 99] + base[:12], captain=1, chip="3xc"),
        make_manager(2, [1, 50] + base[:13], captain=1),
        make_manager(3, [1, 50] + base[:13], captain=1),
        make_manager(4, [1] + base[:14], captain=2),
        make_manager(5, [1] + base[:14], captain=2),
        make_manager(6, [1] + base[:14], captain=3),
    ]


def test_eo_counts_ownership_captaincy_and_tc():
    eo = mini_league_eo(synthetic_league())
    # Player 1: owned by 6, captained by 3, TC by 1 -> (6 + 3 + 1) / 6
    assert eo[1] == (6 + 3 + 1) / 6


def test_eo_single_owner():
    eo = mini_league_eo(synthetic_league())
    assert eo[99] == 1 / 6


def test_eo_partial_ownership():
    eo = mini_league_eo(synthetic_league())
    assert eo[50] == 3 / 6


def test_eo_non_captain_owned_player_counts_once_per_owner():
    eo = mini_league_eo(synthetic_league())
    # Player 2: owned by all 6, captained by managers 4 and 5 -> (6 + 2) / 6
    assert eo[2] == (6 + 2) / 6


def test_eo_empty_league():
    assert mini_league_eo([]) == {}


def test_classification_shield_sword_neutral():
    eo = mini_league_eo(synthetic_league())
    projections = {1: 6.0, 50: 3.0, 99: 2.0, 777: 5.5, 888: 2.0}
    labels = classify_pool(eo, projections)
    assert labels[1] == "SHIELD"      # EO 1.67
    assert labels[50] == "SHIELD"     # EO 0.5, at threshold
    assert labels[777] == "SWORD"     # EO 0, projection 5.5
    assert labels[888] == "NEUTRAL"   # EO 0 but low projection
    assert labels[99] == "NEUTRAL"    # EO 0.167 > sword max? no: 0.167 > 0.15
    # player 99 has EO just above the sword cutoff and a low projection


def test_sword_requires_projection():
    labels = classify_pool({}, {10: 4.5, 11: 4.4})
    assert labels[10] == "SWORD"
    assert labels[11] == "NEUTRAL"


def test_compare_squads():
    league = synthetic_league()
    me, rival = league[0], league[3]
    cmp = compare_squads(me, rival)
    assert cmp["overlap_pct"] == round(100 * 13 / 15, 1)
    assert set(cmp["my_differentials"]) == {50, 99}
    assert cmp["their_differentials"] == [14]


def test_chips_left():
    m = make_manager(1, list(range(1, 16)), captain=1)
    m.chips_used = [{"name": "wildcard", "event": 4}, {"name": "bboost", "event": 6}]
    left = m.chips_left
    assert left["wildcard"] == 1  # two wildcards per season
    assert left["bboost"] == 0
    assert left["freehit"] == 1
    assert left["3xc"] == 1
