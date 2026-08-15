"""FastAPI web API over the engine. The CLI is untouched.

    uvicorn rivalr.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET /health                  status, gw, deadline, last snapshot
    GET /brief?team_id&league_id&mode&target   JSON brief (see briefdata)
    GET /brief/status?job_id     poll a slow solve
    GET /league?league_id        standings + entries for a target picker

Design notes:
  - Solves are slow (minutes). Requests wait up to SYNC_WAIT_S; past
    that they get {job_id} (HTTP 202) and poll /brief/status. One
    in-process worker thread serialises solves (run uvicorn with a
    single worker process or jobs/cache go per-process).
  - Cache: Postgres via DATABASE_URL, keyed (team, league, mode,
    target, gw), TTL 1h - ALWAYS bypassed in the 4h pre-deadline window
    (injury flags move). No DATABASE_URL -> in-memory cache (dev).
  - Rate limit: 30 req/min/IP, in-process sliding window.
  - CORS: localhost dev ports + RIVALR_CORS_ORIGINS (comma-separated).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import briefdata, ledger
from .fetch import FPLClient

log = logging.getLogger("rivalr.api")

SYNC_WAIT_S = 8
CACHE_TTL_S = 3600
PRE_DEADLINE_BYPASS_H = 4
RATE_LIMIT = 30          # requests
RATE_WINDOW_S = 60       # per this many seconds per IP
JOB_RETENTION_S = 3600

app = FastAPI(title="rivalr", version="0.1.0")

_origins = ["http://localhost:3000", "http://localhost:5173",
            "http://127.0.0.1:3000", "http://127.0.0.1:5173"]
_origins += [o.strip() for o in os.environ.get("RIVALR_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=_origins,
    allow_methods=["GET"], allow_headers=["*"],
)

# injectable for tests
brief_builder = briefdata.build_brief_json
client_factory = FPLClient


# -- rate limiting ---------------------------------------------------------

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "?"
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > RATE_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse(
                {"detail": "rate limit exceeded"}, status_code=429
            )
        q.append(now)
    return await call_next(request)


# -- cache -----------------------------------------------------------------


class MemoryCache:
    def __init__(self) -> None:
        self._d: dict[tuple, tuple[float, dict]] = {}

    def get(self, key: tuple) -> dict | None:
        hit = self._d.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL_S:
            return hit[1]
        return None

    def put(self, key: tuple, payload: dict) -> None:
        self._d[key] = (time.time(), payload)


class PgCache:
    DDL = """
    CREATE TABLE IF NOT EXISTS brief_cache (
        team_id BIGINT NOT NULL,
        league_id BIGINT NOT NULL,
        mode TEXT NOT NULL,
        target BIGINT NOT NULL,
        gw INT NOT NULL,
        payload JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (team_id, league_id, mode, target, gw)
    )"""

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn
        with psycopg.connect(dsn) as conn:
            conn.execute(self.DDL)
            conn.commit()

    def get(self, key: tuple) -> dict | None:
        team, league, mode, target, gw = key
        with self._psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT payload FROM brief_cache WHERE team_id=%s AND "
                "league_id=%s AND mode=%s AND target=%s AND gw=%s AND "
                "created_at > now() - interval '1 hour'",
                (team, league, mode, target or 0, gw),
            ).fetchone()
        return row[0] if row else None

    def put(self, key: tuple, payload: dict) -> None:
        team, league, mode, target, gw = key
        with self._psycopg.connect(self._dsn) as conn:
            conn.execute(
                "INSERT INTO brief_cache "
                "(team_id, league_id, mode, target, gw, payload, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,now()) "
                "ON CONFLICT (team_id, league_id, mode, target, gw) "
                "DO UPDATE SET payload=EXCLUDED.payload, created_at=now()",
                (team, league, mode, target or 0, gw, json.dumps(payload)),
            )
            conn.commit()


def _make_cache():
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        try:
            c = PgCache(dsn)
            log.info("brief cache: postgres")
            return c
        except Exception:
            log.exception("postgres cache unavailable, using in-memory")
    else:
        log.warning("DATABASE_URL not set - in-memory brief cache")
    return MemoryCache()


cache = _make_cache()


# -- job queue -------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=1)  # solves are heavy: serialise
_jobs: dict[str, dict] = {}
_jobs_by_key: dict[tuple, str] = {}
_jobs_lock = threading.Lock()


def _gc_jobs() -> None:
    cutoff = time.time() - JOB_RETENTION_S
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if v["created"] < cutoff]:
            key = _jobs[jid].get("key")
            _jobs.pop(jid, None)
            if key and _jobs_by_key.get(key) == jid:
                _jobs_by_key.pop(key, None)


def _run_job(jid: str, key: tuple, kwargs: dict) -> None:
    with _jobs_lock:
        _jobs[jid]["status"] = "running"
    try:
        payload = brief_builder(client_factory(), **kwargs)
        cache.put(key, payload)
        with _jobs_lock:
            _jobs[jid].update(status="done", result=payload)
    except Exception as exc:
        log.exception("brief job failed")
        with _jobs_lock:
            _jobs[jid].update(status="failed", error=repr(exc))


# -- helpers ---------------------------------------------------------------


def _gw_and_deadline(client) -> tuple[int, datetime]:
    bootstrap = client.bootstrap()
    for ev in bootstrap["events"]:
        if ev["is_next"]:
            return ev["id"], datetime.fromisoformat(
                ev["deadline_time"].replace("Z", "+00:00")
            )
    raise HTTPException(503, "no upcoming gameweek")


def _in_pre_deadline_window(deadline: datetime) -> bool:
    now = datetime.now(timezone.utc)
    return deadline - timedelta(hours=PRE_DEADLINE_BYPASS_H) <= now < deadline


# -- endpoints -------------------------------------------------------------


@app.get("/health")
def health():
    client = client_factory()
    gw, deadline = _gw_and_deadline(client)
    last_snapshot = None
    try:
        snaps = [
            p for p in ledger.LEDGER_DIR.iterdir()
            if ledger._SNAPSHOT_RE.match(p.name)
        ]
        if snaps:
            latest = max(snaps, key=lambda p: p.stat().st_mtime)
            last_snapshot = datetime.fromtimestamp(
                latest.stat().st_mtime, tz=timezone.utc
            ).isoformat()
    except FileNotFoundError:
        pass
    return {
        "status": "ok",
        "gameweek": gw,
        "deadline": deadline.isoformat(),
        "last_snapshot": last_snapshot,
    }


@app.get("/brief")
def brief(
    team_id: int = Query(...),
    league_id: int = Query(...),
    mode: str = Query("points", pattern="^(points|chase|defend)$"),
    target: int | None = Query(None),
):
    _gc_jobs()
    client = client_factory()
    gw, deadline = _gw_and_deadline(client)
    key = (team_id, league_id, mode, target or 0, gw)

    if not _in_pre_deadline_window(deadline):
        cached = cache.get(key)
        if cached is not None:
            return {"cached": True, **cached}

    kwargs = dict(team_id=team_id, league_id=league_id, mode=mode,
                  target_id=target)
    with _jobs_lock:
        jid = _jobs_by_key.get(key)
        job = _jobs.get(jid) if jid else None
        # Reuse only in-flight jobs. A done job must NOT serve as a shadow
        # cache - the real cache handles freshness (and the pre-deadline
        # window deliberately bypasses it).
        if job is None or job["status"] not in ("queued", "running"):
            jid = uuid.uuid4().hex
            _jobs[jid] = {"status": "queued", "created": time.time(),
                          "key": key}
            _jobs_by_key[key] = jid
            _executor.submit(_run_job, jid, key, kwargs)
            job = _jobs[jid]

    deadline_t = time.time() + SYNC_WAIT_S
    while time.time() < deadline_t:
        with _jobs_lock:
            status = _jobs[jid]["status"]
            if status == "done":
                return _jobs[jid]["result"]
            if status == "failed":
                raise HTTPException(500, _jobs[jid].get("error", "solve failed"))
        time.sleep(0.25)
    return JSONResponse({"job_id": jid, "status": "pending"}, status_code=202)


@app.get("/brief/status")
def brief_status(job_id: str = Query(...)):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job id")
        if job["status"] == "done":
            return {"status": "done", "result": job["result"]}
        if job["status"] == "failed":
            return {"status": "failed", "error": job.get("error")}
        return {"status": job["status"]}


@app.get("/league")
def league(league_id: int = Query(...)):
    client = client_factory()
    data = client.league_standings(league_id)
    results = data["standings"]["results"]
    entries = [
        {"entry_id": r["entry"], "name": r.get("player_name", ""),
         "team_name": r.get("entry_name", ""), "rank": r.get("rank"),
         "points": r.get("total")}
        for r in results
    ]
    if not entries:  # pre-season
        entries = [
            {"entry_id": r["entry"],
             "name": f"{r.get('player_first_name', '')} "
                     f"{r.get('player_last_name', '')}".strip(),
             "team_name": r.get("entry_name", ""), "rank": None,
             "points": 0}
            for r in data.get("new_entries", {}).get("results", [])
        ]
    return {
        "league_id": league_id,
        "name": data["league"]["name"],
        "entries": entries,
    }
