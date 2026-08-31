"""Gameweek boundary semantics + free-transfer reconstruction."""

from datetime import datetime, timedelta, timezone

import pytest

from rivalr import gameweek
from rivalr.rivals import free_transfers


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class StubClient:
    def __init__(self, events, refreshed_events=None):
        self._events = events
        self._refreshed = refreshed_events
        self.forced = 0

    def bootstrap(self):
        return {"events": self._events}

    def get(self, path, force=False):
        self.forced += 1
        if self._refreshed is not None:
            self._events = self._refreshed
        return {"events": self._events}


def ev(id, deadline, is_next=False, is_current=False,
       finished=False, checked=False):
    return {"id": id, "deadline_time": iso(deadline), "is_next": is_next,
            "is_current": is_current, "finished": finished,
            "data_checked": checked}


def test_stale_is_next_triggers_refresh():
    """Cached bootstrap straddling a deadline must NOT let the planner
    solve for a gameweek already in progress."""
    now = datetime.now(timezone.utc)
    stale = [ev(2, now - timedelta(hours=3), is_next=True),
             ev(3, now + timedelta(days=4))]
    fresh = [ev(2, now - timedelta(hours=3), is_current=True),
             ev(3, now + timedelta(days=4), is_next=True)]
    c = StubClient(stale, refreshed_events=fresh)
    assert gameweek.next_gw(c) == 3
    assert c.forced == 1  # refresh happened


def test_stale_api_advances_manually():
    """Even if the API itself is slow to flip is_next, never return a
    gameweek whose deadline has passed."""
    now = datetime.now(timezone.utc)
    stale = [ev(2, now - timedelta(hours=1), is_next=True),
             ev(3, now + timedelta(days=6))]
    c = StubClient(stale, refreshed_events=stale)  # refresh doesn't help
    assert gameweek.next_gw(c) == 3


def test_unplayed_fixtures_never_complete():
    """A GW with fixtures still to play is not complete, whatever the
    per-fixture flags claim (event finished+data_checked required)."""
    now = datetime.now(timezone.utc)
    events = [ev(2, now - timedelta(days=3), is_current=True,
                 finished=False, checked=False),
              ev(3, now + timedelta(days=4), is_next=True)]
    c = StubClient(events)
    assert not gameweek.is_complete(c, 2)
    # and even finished without data_checked is not complete
    events[0]["finished"] = True
    assert not gameweek.is_complete(c, 2)
    events[0]["data_checked"] = True
    assert gameweek.is_complete(c, 2)


def test_fixture_played_uses_started_not_finished():
    assert gameweek.fixture_played({"started": True, "finished": False})
    assert not gameweek.fixture_played({"started": False, "finished": False})


def test_score_gw_refuses_incomplete(tmp_path):
    from rivalr.ledger import record_predictions, score_gw

    now = datetime.now(timezone.utc)
    c = StubClient([ev(2, now - timedelta(days=2), is_current=True)])
    record_predictions(2, {1: [2.0]}, {}, ledger_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not complete"):
        score_gw(c, 2, ledger_dir=tmp_path)


# -- free transfers --------------------------------------------------------


class FTStub:
    def __init__(self, made_by_gw, chips=None):
        self._made = made_by_gw
        self._chips = chips or []

    def entry_history(self, team_id):
        return {
            "current": [{"event": g, "event_transfers": n}
                        for g, n in self._made.items()],
            "chips": self._chips,
        }


def test_banked_gw2_gives_two_fts_into_gw3():
    # the reported bug: no transfers made in GW2 -> 2 FTs entering GW3
    assert free_transfers(FTStub({1: 0, 2: 0}), 1, upto_gw=3) == 2


def test_spending_keeps_one():
    assert free_transfers(FTStub({2: 1}), 1, upto_gw=3) == 1


def test_hits_cannot_go_negative():
    assert free_transfers(FTStub({2: 4}), 1, upto_gw=3) == 1


def test_cap_at_five():
    assert free_transfers(FTStub({g: 0 for g in range(1, 12)}), 1, upto_gw=12) == 5


def test_wildcard_week_consumes_nothing():
    chips = [{"event": 2, "name": "wildcard"}]
    assert free_transfers(FTStub({2: 8}, chips), 1, upto_gw=3) == 2


def test_preseason_zero():
    assert free_transfers(FTStub({}), 1, upto_gw=1) == 0
