"""Shared Postgres store: brief cache + requested-pair tracking.

Used by the API (serve + record) and the worker (pre-warm). Without
DATABASE_URL everything degrades to in-memory (dev/tests).

Key = (team_id, league_id, mode, target-or-0, gw).
"""

from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger("rivalr.store")

# Payload schema version: part of every cache key, so a deploy that
# changes payload content/shape invalidates stale entries instead of
# serving pre-fix briefs for up to 6 hours (this happened; bump on any
# payload-affecting change).
CACHE_SCHEMA_V = 15  # v15: transfer driver blocks + form-warn guardrail + causation rules


def cache_key(team_id: int, league_id: int, mode: str, target: int | None,
              gw: int) -> tuple:
    return (team_id, league_id, f"{mode}@v{CACHE_SCHEMA_V}", target or 0, gw)


# Serving freshness matches the pre-warm refresh threshold (6h): between
# refreshes a pre-warmed brief must still serve, or "first visitor gets a
# cached response" fails for most of the day. Near-deadline freshness is
# protected separately by WINDOW_TTL_S.
SERVE_TTL_S = 6 * 3600
WINDOW_TTL_S = 1800         # freshness required inside the pre-deadline
                            # window (pre-warmed entries qualify; stale
                            # ones force a live solve)
STALE_REFRESH_S = 6 * 3600  # pre-warm refreshes entries older than this


class MemoryStore:
    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._pairs: dict[tuple, dict] = {}
        self._snapshots: list[dict] = []
        self._scores: dict[int, dict] = {}

    def record_snapshot(self, gw: int, filename: str, partial: bool) -> None:
        self._snapshots.append({
            "gw": gw, "filename": filename, "partial": partial,
            "recorded_at": time.time(),
        })

    def last_snapshot(self) -> dict | None:
        return self._snapshots[-1] if self._snapshots else None

    def put_score(self, gw: int, payload: dict) -> None:
        self._scores[gw] = payload

    def scores(self) -> list[dict]:
        return [self._scores[g] for g in sorted(self._scores)]

    def put_model_gw(self, gw: int, payload: dict) -> None:
        self._model = getattr(self, "_model", {})
        self._model[gw] = payload

    def model_rows(self) -> list[dict]:
        m = getattr(self, "_model", {})
        return [m[g] for g in sorted(m)]

    def put_model_decision(self, gw: int, payload: dict) -> None:
        self._decisions = getattr(self, "_decisions", {})
        self._decisions[gw] = payload

    def model_decisions(self) -> list[dict]:
        d = getattr(self, "_decisions", {})
        return [d[g] for g in sorted(d)]

    def put_gw_projections(self, gw: int, payload: dict) -> None:
        self._gwproj = getattr(self, "_gwproj", {})
        self._gwproj[gw] = payload

    def gw_projections(self, gw: int) -> dict:
        return getattr(self, "_gwproj", {}).get(gw, {})

    def add_llm_usage(self, day: str, input_tokens: int, output_tokens: int) -> None:
        u = self._llm = getattr(self, "_llm", {})
        row = u.setdefault(day, {"day": day, "calls": 0,
                                 "input_tokens": 0, "output_tokens": 0})
        row["calls"] += 1
        row["input_tokens"] += input_tokens
        row["output_tokens"] += output_tokens

    def llm_usage(self) -> list[dict]:
        return [getattr(self, "_llm", {})[d] for d in sorted(getattr(self, "_llm", {}))]

    def get(self, key: tuple, max_age_s: int = SERVE_TTL_S) -> dict | None:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < max_age_s:
            return hit[1]
        return None

    def put(self, key: tuple, payload: dict) -> None:
        self._cache[key] = (time.time(), payload)

    def ever_cached(self, team_id: int, league_id: int) -> bool:
        return any(k[0] == team_id and k[1] == league_id for k in self._cache)

    def record_pair(self, team_id: int, league_id: int, mode: str,
                    target: int | None) -> None:
        k = (team_id, league_id, mode, target or 0)
        p = self._pairs.setdefault(k, {"hits": 0})
        p["hits"] += 1
        p["last_seen"] = time.time()

    def pairs(self) -> list[dict]:
        return [
            {"team_id": k[0], "league_id": k[1], "mode": k[2],
             "target": k[3] or None, "hits": v["hits"]}
            for k, v in self._pairs.items()
        ]

    def stale_keys(self, gw: int, older_than_s: int = STALE_REFRESH_S) -> list[tuple]:
        """Work list of RAW pair tuples (unversioned mode); staleness is
        judged against the current-version cache key."""
        keys = [
            (p["team_id"], p["league_id"], p["mode"], p["target"] or 0, gw)
            for p in self.pairs()
        ]
        if older_than_s <= 0:  # force refresh: everything is stale
            return keys
        cutoff = time.time() - older_than_s
        out = []
        for k in keys:
            vk = cache_key(k[0], k[1], k[2], k[3], k[4])
            hit = self._cache.get(vk)
            if hit is None or hit[0] < cutoff:
                out.append(k)
        return out


class PgStore:
    DDL = [
        """CREATE TABLE IF NOT EXISTS brief_cache (
            team_id BIGINT NOT NULL,
            league_id BIGINT NOT NULL,
            mode TEXT NOT NULL,
            target BIGINT NOT NULL,
            gw INT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (team_id, league_id, mode, target, gw)
        )""",
        """CREATE TABLE IF NOT EXISTS requested_pairs (
            team_id BIGINT NOT NULL,
            league_id BIGINT NOT NULL,
            mode TEXT NOT NULL,
            target BIGINT NOT NULL,
            hits INT NOT NULL DEFAULT 1,
            first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (team_id, league_id, mode, target)
        )""",
        """CREATE TABLE IF NOT EXISTS snapshot_meta (
            id SERIAL PRIMARY KEY,
            gw INT NOT NULL,
            filename TEXT NOT NULL,
            partial BOOLEAN NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS gw_scores (
            gw INT PRIMARY KEY,
            payload JSONB NOT NULL,
            scored_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS model_team (
            gw INT PRIMARY KEY,
            payload JSONB NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS model_decisions (
            gw INT PRIMARY KEY,
            payload JSONB NOT NULL,
            decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS gw_projections (
            gw INT PRIMARY KEY,
            payload JSONB NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS llm_usage (
            day TEXT PRIMARY KEY,
            calls INT NOT NULL DEFAULT 0,
            input_tokens BIGINT NOT NULL DEFAULT 0,
            output_tokens BIGINT NOT NULL DEFAULT 0
        )""",
    ]

    def put_model_decision(self, gw: int, payload: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO model_decisions (gw, payload, decided_at) "
                "VALUES (%s,%s,now()) ON CONFLICT (gw) DO UPDATE "
                "SET payload=EXCLUDED.payload, decided_at=now()",
                (gw, json.dumps(payload)),
            )
            conn.commit()

    def model_decisions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM model_decisions ORDER BY gw",
            ).fetchall()
        return [r[0] for r in rows]

    def put_gw_projections(self, gw: int, payload: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO gw_projections (gw, payload, recorded_at) "
                "VALUES (%s,%s,now()) ON CONFLICT (gw) DO UPDATE "
                "SET payload=EXCLUDED.payload, recorded_at=now()",
                (gw, json.dumps(payload)),
            )
            conn.commit()

    def gw_projections(self, gw: int) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM gw_projections WHERE gw=%s", (gw,),
            ).fetchone()
        return row[0] if row else {}

    def add_llm_usage(self, day: str, input_tokens: int, output_tokens: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO llm_usage (day, calls, input_tokens, output_tokens) "
                "VALUES (%s, 1, %s, %s) "
                "ON CONFLICT (day) DO UPDATE SET "
                "calls = llm_usage.calls + 1, "
                "input_tokens = llm_usage.input_tokens + EXCLUDED.input_tokens, "
                "output_tokens = llm_usage.output_tokens + EXCLUDED.output_tokens",
                (day, input_tokens, output_tokens),
            )

    def llm_usage(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT day, calls, input_tokens, output_tokens "
                "FROM llm_usage ORDER BY day",
            ).fetchall()
        return [{"day": r[0], "calls": r[1], "input_tokens": r[2],
                 "output_tokens": r[3]} for r in rows]

    def put_model_gw(self, gw: int, payload: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO model_team (gw, payload, computed_at) "
                "VALUES (%s,%s,now()) ON CONFLICT (gw) DO UPDATE "
                "SET payload=EXCLUDED.payload, computed_at=now()",
                (gw, json.dumps(payload)),
            )
            conn.commit()

    def model_rows(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM model_team ORDER BY gw",
            ).fetchall()
        return [r[0] for r in rows]

    def record_snapshot(self, gw: int, filename: str, partial: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO snapshot_meta (gw, filename, partial) "
                "VALUES (%s,%s,%s)", (gw, filename, partial),
            )
            conn.commit()

    def last_snapshot(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT gw, filename, partial, recorded_at FROM snapshot_meta "
                "ORDER BY recorded_at DESC LIMIT 1",
            ).fetchone()
        if not row:
            return None
        return {"gw": row[0], "filename": row[1], "partial": row[2],
                "recorded_at": row[3].isoformat()}

    def put_score(self, gw: int, payload: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO gw_scores (gw, payload, scored_at) "
                "VALUES (%s,%s,now()) ON CONFLICT (gw) DO UPDATE "
                "SET payload=EXCLUDED.payload, scored_at=now()",
                (gw, json.dumps(payload)),
            )
            conn.commit()

    def scores(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM gw_scores ORDER BY gw",
            ).fetchall()
        return [r[0] for r in rows]

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn
        with psycopg.connect(dsn) as conn:
            for ddl in self.DDL:
                conn.execute(ddl)
            conn.commit()

    def _conn(self):
        return self._psycopg.connect(self._dsn)

    def get(self, key: tuple, max_age_s: int = SERVE_TTL_S) -> dict | None:
        team, league, mode, target, gw = key
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM brief_cache WHERE team_id=%s AND "
                "league_id=%s AND mode=%s AND target=%s AND gw=%s AND "
                "created_at > now() - %s * interval '1 second'",
                (team, league, mode, target or 0, gw, max_age_s),
            ).fetchone()
        return row[0] if row else None

    def put(self, key: tuple, payload: dict) -> None:
        team, league, mode, target, gw = key
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO brief_cache "
                "(team_id, league_id, mode, target, gw, payload, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,now()) "
                "ON CONFLICT (team_id, league_id, mode, target, gw) "
                "DO UPDATE SET payload=EXCLUDED.payload, created_at=now()",
                (team, league, mode, target or 0, gw, json.dumps(payload)),
            )
            conn.commit()

    def ever_cached(self, team_id: int, league_id: int) -> bool:
        """Any cache row ever, any gw/mode - the 'is this pair new' test."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM brief_cache WHERE team_id=%s AND league_id=%s "
                "LIMIT 1", (team_id, league_id),
            ).fetchone()
        return row is not None

    def record_pair(self, team_id: int, league_id: int, mode: str,
                    target: int | None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO requested_pairs (team_id, league_id, mode, target) "
                "VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (team_id, league_id, mode, target) DO UPDATE "
                "SET hits = requested_pairs.hits + 1, last_seen = now()",
                (team_id, league_id, mode, target or 0),
            )
            conn.commit()

    def pairs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT team_id, league_id, mode, target, hits "
                "FROM requested_pairs ORDER BY hits DESC",
            ).fetchall()
        return [
            {"team_id": r[0], "league_id": r[1], "mode": r[2],
             "target": r[3] or None, "hits": r[4]}
            for r in rows
        ]

    def stale_keys(self, gw: int, older_than_s: int = STALE_REFRESH_S) -> list[tuple]:
        """Requested pairs whose CURRENT-VERSION cache entry for this gw
        is missing or older than the threshold - the pre-warm work list
        (raw, unversioned modes). Threshold <= 0 force-refreshes all."""
        if older_than_s <= 0:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT team_id, league_id, mode, target "
                    "FROM requested_pairs ORDER BY hits DESC",
                ).fetchall()
            return [(r[0], r[1], r[2], r[3], gw) for r in rows]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT p.team_id, p.league_id, p.mode, p.target "
                "FROM requested_pairs p LEFT JOIN brief_cache c "
                "ON c.team_id=p.team_id AND c.league_id=p.league_id "
                "AND c.mode = p.mode || %s AND c.target=p.target AND c.gw=%s "
                "WHERE c.created_at IS NULL "
                "OR c.created_at < now() - %s * interval '1 second' "
                "ORDER BY p.hits DESC",
                (f"@v{CACHE_SCHEMA_V}", gw, older_than_s),
            ).fetchall()
        return [(r[0], r[1], r[2], r[3], gw) for r in rows]


def make_store():
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        try:
            s = PgStore(dsn)
            log.info("store: postgres")
            return s
        except Exception:
            log.exception("postgres store unavailable, using in-memory")
    else:
        log.warning("DATABASE_URL not set - in-memory store")
    return MemoryStore()
