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


# -- store (cache + pair tracking) ----------------------------------------

from .store import SERVE_TTL_S, WINDOW_TTL_S, make_store  # noqa: E402

cache = make_store()


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


def _run_job(jid: str, key: tuple, kwargs: dict, builder=None) -> None:
    with _jobs_lock:
        _jobs[jid]["status"] = "running"
    try:
        payload = (builder or brief_builder)(client_factory(), **kwargs)
        cache.put(key, payload)
        with _jobs_lock:
            _jobs[jid].update(status="done", result=payload)
            notify = _jobs[jid].get("notify")
        if notify:
            from .notify import telegram_send_to

            telegram_send_to(
                notify,
                f"rivalr: your GW{payload.get('gameweek', '?')} brief for "
                f"team {key[0]} is ready - reload the page and it will be "
                f"instant.",
            )
    except Exception as exc:
        log.exception("brief job failed")
        with _jobs_lock:
            _jobs[jid].update(status="failed", error=repr(exc))


# -- helpers ---------------------------------------------------------------


def _gw_and_deadline(client) -> tuple[int, datetime]:
    from . import gameweek

    try:
        return gameweek.next_deadline(client)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


def _in_pre_deadline_window(deadline: datetime) -> bool:
    now = datetime.now(timezone.utc)
    return deadline - timedelta(hours=PRE_DEADLINE_BYPASS_H) <= now < deadline


# -- endpoints -------------------------------------------------------------


@app.get("/health")
def health():
    client = client_factory()
    gw, deadline = _gw_and_deadline(client)
    # Snapshot metadata comes from Postgres: snapshots are written on the
    # worker's volume, which this service cannot see directly.
    last_snapshot = None
    try:
        meta = cache.last_snapshot()
        if meta:
            last_snapshot = {
                "gw": meta["gw"],
                "at": meta["recorded_at"],
                "partial": meta["partial"],
            }
    except Exception:
        log.warning("snapshot meta unavailable", exc_info=True)
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
    notify_chat_id: str | None = Query(None, max_length=32),
):
    _gc_jobs()
    client = client_factory()
    gw, deadline = _gw_and_deadline(client)
    key = (team_id, league_id, mode, target or 0, gw)

    # Track the pair so the worker pre-warms it from now on (best-effort).
    try:
        cache.record_pair(team_id, league_id, mode, target)
    except Exception:
        log.warning("pair tracking failed", exc_info=True)

    # Inside the pre-deadline window only a FRESH entry qualifies (the
    # worker pre-warms during the window, so pre-warmed briefs still
    # serve instantly); outside it, the normal TTL applies.
    max_age = (
        WINDOW_TTL_S if _in_pre_deadline_window(deadline) else SERVE_TTL_S
    )
    cached = cache.get(key, max_age_s=max_age)
    if cached is not None:
        return {"cached": True, **cached}

    # A pair with no cache entry EVER is a genuinely-new visitor: the UI
    # gets told honestly that the first solve takes minutes.
    try:
        first_time = not cache.ever_cached(team_id, league_id)
    except Exception:
        first_time = False

    kwargs = dict(team_id=team_id, league_id=league_id, mode=mode,
                  target_id=target)
    with _jobs_lock:
        jid = _jobs_by_key.get(key)
        job = _jobs.get(jid) if jid else None
        # Reuse only in-flight jobs. A done job must NOT serve as a shadow
        # cache - the real cache handles freshness (and the pre-deadline
        # window deliberately tightens it).
        if job is None or job["status"] not in ("queued", "running"):
            jid = uuid.uuid4().hex
            _jobs[jid] = {"status": "queued", "created": time.time(),
                          "key": key}
            _jobs_by_key[key] = jid
            _executor.submit(_run_job, jid, key, kwargs)
            job = _jobs[jid]
        if notify_chat_id:
            _jobs[jid]["notify"] = notify_chat_id

    deadline_t = time.time() + SYNC_WAIT_S
    while time.time() < deadline_t:
        with _jobs_lock:
            status = _jobs[jid]["status"]
            if status == "done":
                return _jobs[jid]["result"]
            if status == "failed":
                raise HTTPException(500, _jobs[jid].get("error", "solve failed"))
        time.sleep(0.25)
    return JSONResponse(
        {"job_id": jid, "status": "pending", "first_time": first_time},
        status_code=202,
    )


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


@app.get("/plan")
def plan(
    team_id: int = Query(...),
    league_id: int = Query(...),
    horizon: int = Query(5, ge=1, le=8),
    locked: str = Query("", max_length=200),
    banned: str = Query("", max_length=200),
    hits: bool = Query(False),
):
    """Week-by-week transfer plan. Same job/poll pattern as /brief."""
    _gc_jobs()
    client = client_factory()
    gw, deadline = _gw_and_deadline(client)

    def ids(csv: str) -> list[int]:
        try:
            return [int(x) for x in csv.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(422, "locked/banned must be comma-separated ids")

    locked_ids, banned_ids = ids(locked), ids(banned)
    mode_key = (
        f"plan:h{horizon}"
        + (":hits" if hits else "")
        + (f":l{','.join(map(str, sorted(locked_ids)))}" if locked_ids else "")
        + (f":b{','.join(map(str, sorted(banned_ids)))}" if banned_ids else "")
    )
    key = (team_id, league_id, mode_key, 0, gw)

    # Only the unconstrained base plan joins the pre-warm list; ad-hoc
    # lock combinations are solved on demand.
    if not locked_ids and not banned_ids:
        try:
            cache.record_pair(team_id, league_id, mode_key, None)
        except Exception:
            pass

    max_age = WINDOW_TTL_S if _in_pre_deadline_window(deadline) else SERVE_TTL_S
    cached = cache.get(key, max_age_s=max_age)
    if cached is not None:
        return {"cached": True, **cached}

    kwargs = dict(team_id=team_id, league_id=league_id, horizon=horizon,
                  locked=locked_ids, banned=banned_ids, allow_hits=hits)
    with _jobs_lock:
        jid = _jobs_by_key.get(key)
        job = _jobs.get(jid) if jid else None
        if job is None or job["status"] not in ("queued", "running"):
            jid = uuid.uuid4().hex
            _jobs[jid] = {"status": "queued", "created": time.time(), "key": key}
            _jobs_by_key[key] = jid
            _executor.submit(_run_job, jid, key, kwargs,
                             briefdata.build_plan_json)

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


@app.get("/season")
def season(team_id: int = Query(...), league_id: int = Query(...)):
    """Me vs the model vs the rivals, cumulatively.

    'Model' = my actual points plus each scored gameweek's
    recommendation edge (recommended transfers minus my actual ones).
    That is an approximation - it assumes the model started from my real
    squad each week rather than compounding its own decisions - and the
    payload says so."""
    client = client_factory()

    data = client.league_standings(league_id)
    rows = data["standings"]["results"] or [
        {"entry": r["entry"],
         "player_name": f"{r.get('player_first_name', '')} "
                        f"{r.get('player_last_name', '')}".strip()}
        for r in data.get("new_entries", {}).get("results", [])
    ]

    edges: dict[int, int] = {}
    try:
        for s in cache.scores():
            cf = s.get("counterfactual") or {}
            edges[s["gw"]] = cf.get("recommendation_edge", 0)
    except Exception:
        log.warning("score store unavailable for /season", exc_info=True)

    people = []
    me = None
    for r in rows:
        try:
            hist = client.entry_history(r["entry"]).get("current", [])
        except Exception:
            continue
        per_gw = {h["event"]: h["points"] - h.get("event_transfers_cost", 0)
                  for h in hist}
        gws = sorted(per_gw)
        cum, total = [], 0
        for g in gws:
            total += per_gw[g]
            cum.append(total)
        person = {
            "entry_id": r["entry"],
            "name": r.get("player_name", str(r["entry"])),
            "gameweeks": gws,
            "points": [per_gw[g] for g in gws],
            "cum": cum,
        }
        if r["entry"] == team_id:
            me = person
        else:
            people.append(person)

    model_path = None
    if me:
        cum, total = [], 0
        for i, g in enumerate(me["gameweeks"]):
            total += me["points"][i] + edges.get(g, 0)
            cum.append(total)
        model_path = {
            "cum": cum,
            "edges": {str(g): edges.get(g, 0) for g in me["gameweeks"]},
            "caveat": (
                "model line = your actual points plus each scored "
                "gameweek's recommendation edge; unscored gameweeks "
                "contribute zero edge, and compounding (the model building "
                "on its own squad) is not simulated"
            ),
        }

    return {"me": me, "model": model_path, "rivals": people}


@app.get("/fixtures")
def fixtures(horizon: int = Query(8, ge=1, le=12)):
    from . import fixtures as fx

    return fx.fixture_grid(client_factory(), horizon=horizon)


# Backtest results are verified facts from docs/backtest_findings.md -
# reproduce with `uv run python tests/backtest_openfpl.py`.
BACKTEST = {
    "reference": "OpenFPL, arXiv 2508.09992, Table 4 (2024-25 GW32-38)",
    "note": (
        "Blanks is compared against the honest outfield-only benchmark of "
        "1.136, not the paper's published 1.291 - the published aggregate "
        "is inflated by ~15 assistant-manager rows at RMSE 6.192 which our "
        "evaluation excludes. Zeros carries a known availability handicap "
        "(see limitations)."
    ),
    "buckets": [
        {"bucket": "Zeros", "n": 3209, "ours": 0.958, "paper": 0.818,
         "benchmark": 0.818, "validated": False},
        {"bucket": "Blanks", "n": 1464, "ours": 1.610, "paper": 1.291,
         "benchmark": 1.136, "validated": False},
        {"bucket": "Tickers", "n": 185, "ours": 1.393, "paper": 1.517,
         "benchmark": 1.517, "validated": True},
        {"bucket": "Haulers", "n": 455, "ours": 5.208, "paper": 5.142,
         "benchmark": 5.142, "validated": True},
    ],
}


@app.get("/accuracy")
def accuracy():
    """Public accuracy data: the backtest vs the paper, plus every scored
    live gameweek (paper buckets, base vs DefCon-corrected)."""
    def row(s: dict) -> dict:
        return {
            "gw": s["gw"],
            "partial_snapshot": s.get("ledger_partial", False),
            "accuracy": s["accuracy"],
            "accuracy_base": s.get("accuracy_base"),
            "counterfactual": s.get("counterfactual"),
            "unrostered": s.get("unrostered_at_snapshot", {}).get("n", 0),
        }

    by_gw: dict[int, dict] = {}
    # Postgres first (scores are produced on the worker's volume and
    # synced there); local score files fill any gaps (dev).
    try:
        for s in cache.scores():
            by_gw[s["gw"]] = row(s)
    except Exception:
        log.warning("score store unavailable", exc_info=True)
    try:
        for p in sorted(ledger.LEDGER_DIR.glob("gw*_score.json")):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                by_gw.setdefault(s["gw"], row(s))
            except Exception:
                log.warning("unreadable score file %s", p.name)
    except FileNotFoundError:
        pass
    live = [by_gw[g] for g in sorted(by_gw)]
    return {"backtest": BACKTEST, "live": live}


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
