"""League membership guard: a team must never be solved against a
league it is not in (live bug 2026-09-02: picker auto-selected a
43,886-member YouTuber league and rivals raised a raw ValueError)."""

import pytest

from rivalr.briefdata import LeagueMismatch, validate_pair


class Stub:
    def __init__(self, leagues):
        self._leagues = leagues

    def entry(self, team_id):
        return {"leagues": {"classic": self._leagues}}


class BrokenStub:
    def entry(self, team_id):
        raise RuntimeError("network down")


def test_member_of_small_league_passes():
    c = Stub([{"id": 517089, "name": "BHSS", "entry_rank": 3}])
    validate_pair(c, 1, 517089)  # no raise


def test_not_a_member_raises_plain_message():
    c = Stub([{"id": 517089, "name": "BHSS", "entry_rank": 3}])
    with pytest.raises(LeagueMismatch, match="not in league 3876"):
        validate_pair(c, 1, 3876)


def test_giant_public_league_raises_plain_message():
    c = Stub([{"id": 3876, "name": "YouTube.com/FPLFocal",
               "entry_rank": 43886}])
    with pytest.raises(LeagueMismatch, match="big public league"):
        validate_pair(c, 1, 3876)


def test_lookup_failure_does_not_block():
    validate_pair(BrokenStub(), 1, 517089)  # degrades open, logged
