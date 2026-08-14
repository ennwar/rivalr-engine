"""LOW_CONFIDENCE floor-margin diagnostics (docs/backtest_findings.md)."""

from rivalr.model import (
    BLANKS_FLOOR,
    LOW_CONFIDENCE_MARGIN,
    confidence_margin,
    is_low_confidence,
)


def test_margin_is_projection_minus_adjusted_floor():
    # full minutes: floor = 1.7
    assert confidence_margin(2.2, 1.0) == 2.2 - BLANKS_FLOOR
    # half minutes: floor scales with the minutes factor
    assert confidence_margin(1.0, 0.5) == 1.0 - BLANKS_FLOOR * 0.5


def test_flag_within_half_point_of_floor():
    assert is_low_confidence(2.19, 1.0)   # just inside the margin
    assert is_low_confidence(1.9, 1.0)
    assert not is_low_confidence(2.21, 1.0)


def test_minutes_adjustment_moves_the_floor():
    # 1.5 xPts is low-confidence for a nailed starter (floor 1.7)...
    assert is_low_confidence(1.5, 1.0)
    # ...but a genuine signal for a half-game player (floor 0.85)
    assert not is_low_confidence(1.5, 0.5)
