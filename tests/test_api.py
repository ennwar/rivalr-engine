"""API mechanics: job flow, cache bypass window, rate limit, rivals-null.
Uses a stub brief builder - no live solves."""

import time

import pytest
from fastapi.testclient import TestClient

from rivalr import api


class StubClient:
    def __init__(self, deadline="2099-01-01T17:30:00Z"):
        self._deadline = deadline

    def bootstrap(self):
        return {"events": [{"id": 1, "is_next": True,
                            "deadline_time": self._deadline}]}

    def league_standings(self, league_id, page=1):
        return {
            "league": {"name": "Stub League"},
            "standings": {"results": []},
            "new_entries": {"results": [
                {"entry": 111, "player_first_name": "A",
                 "player_last_name": "B", "entry_name": "AB FC"},
            ]},
        }


@pytest.fixture(autouse=True)
def wire_stubs(monkeypatch):
    monkeypatch.setattr(api, "client_factory", lambda: StubClient())
    monkeypatch.setattr(api, "cache", api.MemoryCache())
    api._jobs.clear()
    api._jobs_by_key.clear()
    api._hits.clear()
    yield


client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["gameweek"] == 1


def test_league_preseason_uses_new_entries():
    r = client.get("/league?league_id=517089")
    assert r.status_code == 200
    assert r.json()["entries"][0] == {
        "entry_id": 111, "name": "A B", "team_name": "AB FC",
        "rank": None, "points": 0,
    }


def test_fast_brief_returns_inline_and_caches(monkeypatch):
    monkeypatch.setattr(api, "brief_builder",
                        lambda c, **kw: {"gameweek": 1, "squad": [],
                                         "rivals": None, "warnings": []})
    r = client.get("/brief?team_id=1&league_id=2")
    assert r.status_code == 200 and r.json()["gameweek"] == 1
    # second call served from cache
    monkeypatch.setattr(api, "brief_builder",
                        lambda c, **kw: (_ for _ in ()).throw(AssertionError))
    r2 = client.get("/brief?team_id=1&league_id=2")
    assert r2.status_code == 200 and r2.json()["cached"] is True


def test_slow_brief_returns_job_and_polls(monkeypatch):
    monkeypatch.setattr(api, "SYNC_WAIT_S", 0.5)

    def slow(c, **kw):
        time.sleep(1.2)
        return {"gameweek": 1, "slow": True}

    monkeypatch.setattr(api, "brief_builder", slow)
    r = client.get("/brief?team_id=1&league_id=2")
    assert r.status_code == 202
    jid = r.json()["job_id"]
    for _ in range(40):
        s = client.get(f"/brief/status?job_id={jid}").json()
        if s["status"] == "done":
            assert s["result"]["slow"] is True
            break
        time.sleep(0.1)
    else:
        raise AssertionError("job never finished")


def test_pre_deadline_window_bypasses_cache(monkeypatch):
    from datetime import datetime, timedelta, timezone

    soon = (datetime.now(timezone.utc) + timedelta(hours=2))
    monkeypatch.setattr(api, "client_factory",
                        lambda: StubClient(soon.strftime("%Y-%m-%dT%H:%M:%SZ")))
    calls = {"n": 0}

    def builder(c, **kw):
        calls["n"] += 1
        return {"gameweek": 1, "n": calls["n"]}

    monkeypatch.setattr(api, "brief_builder", builder)
    client.get("/brief?team_id=1&league_id=2")
    r2 = client.get("/brief?team_id=1&league_id=2")
    assert calls["n"] == 2                # cache bypassed inside window
    assert "cached" not in r2.json()


def test_failed_build_returns_500_not_hang(monkeypatch):
    monkeypatch.setattr(
        api, "brief_builder",
        lambda c, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    r = client.get("/brief?team_id=1&league_id=2")
    assert r.status_code == 500


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT", 5)
    codes = [client.get("/health").status_code for _ in range(7)]
    assert codes.count(429) >= 2


def test_invalid_mode_rejected():
    r = client.get("/brief?team_id=1&league_id=2&mode=yolo")
    assert r.status_code == 422
