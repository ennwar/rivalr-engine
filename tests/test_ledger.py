"""Ledger coverage, append-only versioning, and test-file exclusion."""

import json

import pytest

from rivalr.ledger import (
    _latest_ledger_for,
    full_coverage,
    record_predictions,
)
from rivalr.rivals import pairwise_transfer_gain


def test_full_coverage_includes_unprojected_as_null():
    elements = [{"id": 1}, {"id": 2}, {"id": 3}]
    cov = full_coverage({1: [2.0, 3.0]}, elements)
    assert set(cov) == {1, 2, 3}
    assert cov[1] == [2.0, 3.0]
    assert cov[2] is None
    assert cov[3] is None


def test_record_never_overwrites(tmp_path):
    p1 = record_predictions(7, {1: [1.0]}, {}, ledger_dir=tmp_path)
    p2 = record_predictions(7, {1: [2.0]}, {}, ledger_dir=tmp_path)
    p3 = record_predictions(7, {1: [3.0]}, {}, ledger_dir=tmp_path)
    assert p1.name == "gw7.json"
    assert p2.name == "gw7_v2.json"
    assert p3.name == "gw7_v3.json"
    # original untouched
    assert json.loads(p1.read_text())["projections"]["1"] == [1.0]


def test_latest_picks_highest_version(tmp_path):
    record_predictions(7, {1: [1.0]}, {}, ledger_dir=tmp_path)
    record_predictions(7, {1: [2.0]}, {}, ledger_dir=tmp_path)
    assert _latest_ledger_for(7, tmp_path).name == "gw7_v2.json"


def test_scoring_ignores_test_and_score_files(tmp_path):
    (tmp_path / "gw7_preseason_test.json").write_text("{}")
    (tmp_path / "gw7_score.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        _latest_ledger_for(7, tmp_path)
    real = record_predictions(7, {1: [1.0]}, {}, ledger_dir=tmp_path)
    assert _latest_ledger_for(7, tmp_path) == real


def test_partial_flag_and_failures_recorded(tmp_path):
    p = record_predictions(
        3, {1: None}, {}, ledger_dir=tmp_path,
        partial=True, failures=["solver: boom"],
    )
    d = json.loads(p.read_text())
    assert d["partial"] is True
    assert d["failures"] == ["solver: boom"]
    assert d["coverage"] == {
        "bootstrap_elements_at_snapshot": 1,
        "projected": 0,
        "unprojected": 1,
    }


class StubClient:
    """Offline stand-in for FPLClient in scoring tests."""

    def __init__(self, live: dict[int, tuple[int, int]], elements: list[dict]):
        self._live = live  # pid -> (points, minutes)
        self._elements = elements

    def event_live(self, gw):
        return {
            "elements": [
                {"id": pid, "stats": {"total_points": pts, "minutes": mins}}
                for pid, (pts, mins) in self._live.items()
            ]
        }

    def bootstrap(self):
        return {"elements": self._elements}

    def entry_transfers(self, team_id):
        return []


def test_scoring_reports_unrostered_at_snapshot(tmp_path):
    from rivalr.ledger import score_gw

    # Ledger knows players 1 (projected) and 2 (null). Players 3 and 4
    # joined after the snapshot; 3 scored, 4 blanked.
    record_predictions(5, {1: [2.0], 2: None}, {}, ledger_dir=tmp_path)
    client = StubClient(
        live={1: (2, 90), 2: (0, 0), 3: (6, 90), 4: (0, 0)},
        elements=[{"id": i, "web_name": f"P{i}"} for i in (1, 2, 3, 4)],
    )
    result = score_gw(client, 5, ledger_dir=tmp_path)

    unr = result["unrostered_at_snapshot"]
    assert unr["n"] == 1                      # only the scorer counts
    assert unr["players"][0] == {"id": 3, "name": "P3", "points": 6}
    assert unr["total_points_missed"] == 6
    assert result["unprojected_players"] == 1  # null is counted separately
    assert result["accuracy"]["All"]["n"] == 1  # only player 1 is scoreable


def test_scoring_dual_bucket_views(tmp_path):
    from rivalr.ledger import score_gw

    # p1: played, 2 pts -> paper Blanks, legacy Blanks (1-3)
    # p2: did not play, 0 pts -> paper Zeros, legacy Zeros
    # p3: played, 4 pts -> paper Tickers, legacy Tickers (4-9)
    # p4: played, 0 pts -> paper Blanks (played, <=2) but legacy Zeros!
    record_predictions(
        6, {1: [2.0], 2: [1.0], 3: [3.0], 4: [1.5]}, {}, ledger_dir=tmp_path
    )
    client = StubClient(
        live={1: (2, 90), 2: (0, 0), 3: (4, 60), 4: (0, 45)},
        elements=[{"id": i, "web_name": f"P{i}"} for i in (1, 2, 3, 4)],
    )
    result = score_gw(client, 6, ledger_dir=tmp_path)

    paper, legacy = result["accuracy"], result["accuracy_legacy"]
    assert paper["Blanks"]["n"] == 2      # p1 and p4 (played, <=2 pts)
    assert paper["Zeros"]["n"] == 1       # p2 only (did not play)
    assert legacy["Zeros"]["n"] == 2      # p2 and p4 (0 points)
    assert legacy["Blanks"]["n"] == 1     # p1
    assert paper["Tickers"]["n"] == legacy["Tickers"]["n"] == 1
    assert paper["All"] == legacy["All"]


# -- small-league pairwise swing -------------------------------------------


def test_pairwise_gain_differential_in():
    # rival doesn't own the incoming player: full gain, minus the outgoing
    # player's points which the rival also doesn't own (no relative loss).
    proj = {10: 6.0, 20: 2.0}
    assert pairwise_transfer_gain(10, 20, squad=set(), proj=proj) == 4.0


def test_pairwise_gain_neutral_when_rival_owns_incoming():
    proj = {10: 6.0, 20: 2.0}
    assert pairwise_transfer_gain(10, 20, squad={10}, proj=proj) == -2.0


def test_pairwise_gain_selling_rival_owned_concedes_nothing():
    # outgoing player is owned by the rival: selling him concedes no
    # relative ground (loss term suppressed)
    proj = {10: 6.0, 20: 2.0}
    assert pairwise_transfer_gain(10, 20, squad={20}, proj=proj) == 6.0


def test_pairwise_gain_draft_mode_no_out():
    proj = {10: 6.0}
    assert pairwise_transfer_gain(10, None, squad=set(), proj=proj) == 6.0
    assert pairwise_transfer_gain(10, None, squad={10}, proj=proj) == 0.0
