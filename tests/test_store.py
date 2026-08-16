"""Pair tracking + stale-key logic (memory store; PgStore mirrors it)."""

import time

from rivalr.store import MemoryStore, SERVE_TTL_S


def test_record_and_list_pairs():
    s = MemoryStore()
    s.record_pair(1, 2, "points", None)
    s.record_pair(1, 2, "points", None)
    s.record_pair(1, 2, "chase", 99)
    pairs = s.pairs()
    assert len(pairs) == 2
    assert {p["hits"] for p in pairs} == {2, 1}


def test_ever_cached_is_pair_level():
    s = MemoryStore()
    assert not s.ever_cached(1, 2)
    s.put((1, 2, "points", 0, 5), {"x": 1})
    assert s.ever_cached(1, 2)
    assert not s.ever_cached(1, 3)


def test_stale_keys_missing_and_old():
    s = MemoryStore()
    s.record_pair(1, 2, "points", None)   # never cached -> stale
    s.record_pair(3, 4, "points", None)   # cached fresh -> not stale
    s.put((3, 4, "points", 0, 5), {"x": 1})
    stale = s.stale_keys(gw=5, older_than_s=3600)
    assert (1, 2, "points", 0, 5) in stale
    assert (3, 4, "points", 0, 5) not in stale
    # force refresh: threshold 0 makes everything stale
    assert len(s.stale_keys(gw=5, older_than_s=0)) == 2


def test_get_respects_max_age():
    s = MemoryStore()
    s.put((1, 2, "points", 0, 5), {"x": 1})
    assert s.get((1, 2, "points", 0, 5), max_age_s=SERVE_TTL_S) == {"x": 1}
    # entry written "now" fails an impossible freshness bar
    time.sleep(0.01)
    assert s.get((1, 2, "points", 0, 5), max_age_s=0) is None
