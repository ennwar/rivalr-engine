"""Venue term + ep_next retirement (both shipped 2026-09-02)."""

from rivalr.model import OpenFPLModel


def _bare_model(elements: dict) -> OpenFPLModel:
    m = object.__new__(OpenFPLModel)
    m._elements = elements
    return m


def test_venue_adjustment_signs_and_magnitudes():
    m = _bare_model({1: {"element_type": 1}, 2: {"element_type": 2},
                     3: {"element_type": 3}, 4: {"element_type": 4}})
    # GK/DEF: +/-0.220, MID/FWD: +/-0.1875 (half the fitted home coef)
    assert m._venue_adjustment(1, True) == 0.220
    assert m._venue_adjustment(2, False) == -0.220
    assert m._venue_adjustment(3, True) == 0.1875
    assert m._venue_adjustment(4, False) == -0.1875


def test_ep_next_retired_at_two_matches():
    """ep_next must have ZERO weight from 2 season matches: it is a
    trailing points average (corr 0.993 with form), not a projection."""
    m = _bare_model({1: {"ep_next": "10.0"}, 2: {"ep_next": "10.0"}})
    res = m._cold_start_blend({1: [5.0], 2: [5.0]}, {1: 2, 2: 1})
    assert res[1][0] == 5.0  # untouched: pure model at n=2
    assert m.last_blend[1]["model_weight"] == 1.0
    # n=1 still blends (model is nearly blind in GW1-2)
    assert res[2][0] == 7.5  # 0.5*5 + 0.5*10
    assert m.last_blend[2]["model_weight"] == 0.5
