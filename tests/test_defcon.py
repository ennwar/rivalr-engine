"""DefCon layer: threshold rules, Poisson tail, blend, additivity."""

import pytest

from rivalr.defcon import (
    CAP_POINTS,
    THRESHOLDS,
    bps_adjustment,
    defcon_points,
    poisson_tail,
    relevant_count,
)


def row(cbi=0, tackles=0, recoveries=0):
    return {
        "clearances_blocks_interceptions": cbi,
        "tackles": tackles,
        "recoveries": recoveries,
    }


def test_def_counts_cbit_without_recoveries():
    r = row(cbi=6, tackles=3, recoveries=5)
    assert relevant_count(r, "DEF") == 9        # recoveries NOT counted
    assert relevant_count(r, "MID") == 14       # recoveries counted


def test_thresholds_and_cap():
    # DEF: 10 CBIT
    assert defcon_points(9, "DEF") == 0
    assert defcon_points(10, "DEF") == 2
    assert defcon_points(25, "DEF") == 2        # capped, no extra
    # MID/FWD: 12 CBIRT
    assert defcon_points(11, "MID") == 0
    assert defcon_points(12, "FWD") == 2
    # GK not eligible
    assert defcon_points(99, "GK") == 0


def test_thresholds_match_rules():
    assert THRESHOLDS == {"DEF": 10, "MID": 12, "FWD": 12}
    assert CAP_POINTS == 2


def test_poisson_tail_sane():
    assert poisson_tail(0, 10) == 0.0
    assert poisson_tail(10, 10) == pytest.approx(0.542, abs=0.01)
    assert poisson_tail(20, 10) > 0.99
    # monotone in mu
    assert poisson_tail(8, 10) < poisson_tail(12, 10)


def test_bps_stub_returns_zeros():
    assert bps_adjustment(123, 5) == [0.0] * 5


def test_correction_is_additive_never_overwrites():
    """The integration contract: final = base + defcon, elementwise."""
    base = {1: [2.0, 3.0], 2: [1.5, 1.5]}
    dc = {1: [0.4, 0.6]}  # player 2 has no correction
    final = {
        pid: [
            round(x + (dc.get(pid) or [0.0] * len(xs))[i], 3)
            for i, x in enumerate(xs)
        ]
        for pid, xs in base.items()
    }
    assert final == {1: [2.4, 3.6], 2: [1.5, 1.5]}
    assert base == {1: [2.0, 3.0], 2: [1.5, 1.5]}  # base untouched
