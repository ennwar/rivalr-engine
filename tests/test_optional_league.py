"""League is optional: core analysis for anyone; rivals are an add-on."""

from rivalr.briefdata import league_usable


class LStub:
    def __init__(self, leagues):
        self._l = leagues

    def entry(self, tid):
        return {"leagues": {"classic": self._l}}


def test_no_league_is_core_only_no_note():
    ok, note = league_usable(LStub([]), 1, None)
    assert ok is False and note is None


def test_small_league_enables_rivals():
    ok, note = league_usable(
        LStub([{"id": 517089, "name": "BHSS", "entry_rank": 3}]), 1, 517089)
    assert ok is True and note is None


def test_large_league_disables_rivals_with_note():
    ok, note = league_usable(
        LStub([{"id": 3876, "name": "FPLFocal", "entry_rank": 43886}]), 1, 3876)
    assert ok is False and "big public league" in note


def test_not_a_member_disables_rivals_with_note():
    ok, note = league_usable(
        LStub([{"id": 517089, "name": "BHSS", "entry_rank": 3}]), 1, 999)
    assert ok is False and "isn't in league 999" in note


def test_lookup_failure_is_core_only_no_note():
    class Broken:
        def entry(self, tid):
            raise RuntimeError("down")
    ok, note = league_usable(Broken(), 1, 517089)
    assert ok is False and note is None
