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

from .store import SERVE_TTL_S, WINDOW_TTL_S, cache_key, make_store  # noqa: E402

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
    key = cache_key(team_id, league_id, mode, target, gw)

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
    key = cache_key(team_id, league_id, mode_key, None, gw)

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
                "secondary view: your actual points plus each scored "
                "gameweek's recommendation edge (you + the advice). The "
                "autonomous model team below is the primary comparison."
            ),
        }

    # The autonomous model team: its own draft, its own transfers, its
    # own captain - never sees anyone's squad. Settled GWs only.
    model_team = None
    try:
        rows = cache.model_rows()
        if rows:
            model_team = {
                "name": "The Model",
                "gameweeks": [r["gw"] for r in rows],
                "points": [r["points"] for r in rows],
                "cum": [r["total"] for r in rows],
                "hits": sum(r.get("hits", 0) for r in rows),
            }
    except Exception:
        log.warning("model team unavailable", exc_info=True)

    return {"me": me, "model": model_path, "model_team": model_team,
            "rivals": people}


@app.get("/ask/questions")
def ask_questions(team_id: int = Query(...), league_id: int = Query(...)):
    """Contextual suggested-question chips (the primary interface)."""
    from . import assistant

    return {"chips": assistant.chips_for(team_id, league_id)}


@app.get("/ask/usage")
def ask_usage():
    """Daily LLM spend for the assistant, so the cost is visible before
    it ever isn't. Estimated cost uses claude-haiku-4-5 list pricing
    ($1/M input, $5/M output)."""
    rows = cache.llm_usage()
    for r in rows:
        r["est_cost_usd"] = round(
            r["input_tokens"] * 1.0 / 1e6 + r["output_tokens"] * 5.0 / 1e6, 4,
        )
    return {
        "days": rows,
        "total_est_cost_usd": round(sum(r["est_cost_usd"] for r in rows), 4),
        "caps": {"max_output_tokens": 400, "max_input_chars": 8000,
                 "cache_ttl_h": 6},
    }


@app.get("/ask")
def ask(
    team_id: int = Query(...),
    league_id: int = Query(...),
    qid: str | None = Query(None, max_length=24),
    text: str | None = Query(None, max_length=300),
):
    """Grounded assistant answer. Same job/poll pattern as /brief for
    slow questions (simulation); cached like everything else. Pass a
    registry qid OR free `text` - free text goes through the same
    grounding contract (LLM sees engine JSON only, never invents)."""
    import hashlib

    from . import assistant

    if not qid and not (text and text.strip()):
        raise HTTPException(422, "pass qid or text")
    _gc_jobs()
    client = client_factory()
    gw, deadline = _gw_and_deadline(client)
    if qid:
        if qid not in assistant.QUESTIONS:
            raise HTTPException(422, "unknown question id")
        key = cache_key(team_id, league_id, f"ask:{qid}", None, gw)
        kwargs = dict(team_id=team_id, league_id=league_id, qid=qid)

        def builder(c, **kw):
            return assistant.answer(c, usage_store=cache, **kw)
    else:
        norm = " ".join(text.split()).lower()
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]
        key = cache_key(team_id, league_id, f"ask:free:{h}", None, gw)
        kwargs = dict(team_id=team_id, league_id=league_id, text=text)

        def builder(c, **kw):
            return assistant.answer_free(c, usage_store=cache, **kw)

    cached = cache.get(key, max_age_s=SERVE_TTL_S)
    if cached is not None:
        return {"cached": True, **cached}

    with _jobs_lock:
        jid = _jobs_by_key.get(key)
        job = _jobs.get(jid) if jid else None
        if job is None or job["status"] not in ("queued", "running"):
            jid = uuid.uuid4().hex
            _jobs[jid] = {"status": "queued", "created": time.time(), "key": key}
            _jobs_by_key[key] = jid
            _executor.submit(_run_job, jid, key, kwargs, builder)

    deadline_t = time.time() + SYNC_WAIT_S
    while time.time() < deadline_t:
        with _jobs_lock:
            status = _jobs[jid]["status"]
            if status == "done":
                return _jobs[jid]["result"]
            if status == "failed":
                raise HTTPException(500, _jobs[jid].get("error", "ask failed"))
        time.sleep(0.25)
    return JSONResponse({"job_id": jid, "status": "pending"}, status_code=202)


@app.get("/season/squads")
def season_squads(
    team_id: int = Query(...),
    league_id: int = Query(...),
    gw: int = Query(..., ge=1, le=38),
):
    """The evidence behind the chart: everyone's actual squad for one
    gameweek, plus the model's squad (mine with the recommended
    transfers applied). Budget figures are real (entry history); player
    prices are CURRENT now_cost, labelled approximate."""
    client = client_factory()
    bootstrap = client.bootstrap()
    els = {el["id"]: el for el in bootstrap["elements"]}
    tnames = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    try:
        live = {el["id"]: el["stats"]["total_points"]
                for el in client.event_live(gw)["elements"]}
    except Exception:
        live = {}

    data = client.league_standings(league_id)
    rows = data["standings"]["results"] or [
        {"entry": r["entry"],
         "player_name": f"{r.get('player_first_name', '')} "
                        f"{r.get('player_last_name', '')}".strip()}
        for r in data.get("new_entries", {}).get("results", [])
    ]

    def squad_of(entry_id: int) -> dict | None:
        try:
            picks = client.entry_picks(entry_id, gw)
        except Exception:
            return None
        hist = {h["event"]: h for h in
                client.entry_history(entry_id).get("current", [])}
        h = hist.get(gw, {})
        players = []
        for p in picks["picks"]:
            el = els.get(p["element"], {})
            players.append({
                "id": p["element"],
                "name": el.get("web_name", f"#{p['element']}"),
                "club": tnames.get(el.get("team"), "?"),
                "position": p["position"],
                "price_now": el.get("now_cost", 0) / 10.0,
                "points": live.get(p["element"], 0),
                "captain": p.get("is_captain", False),
                "in_xi": p["position"] <= 11,
            })
        return {
            "players": players,
            "chip": picks.get("active_chip"),
            "squad_value": h.get("value", 0) / 10.0 if h else None,
            "bank": h.get("bank", 0) / 10.0 if h else None,
            "gw_points": h.get("points"),
        }

    people = []
    mine = None
    for r in rows:
        s = squad_of(r["entry"])
        if s is None:
            continue
        entry = {"entry_id": r["entry"],
                 "name": r.get("player_name", str(r["entry"])), **s}
        if r["entry"] == team_id:
            mine = entry
        else:
            people.append(entry)

    # Model squad: the AUTONOMOUS model team only - its settled row, or
    # its pre-deadline DECISION for unsettled gameweeks. The old
    # "mine + recommended transfers" overlay is gone: it rendered the
    # human's squad under the model's name.
    try:
        row = next((r for r in cache.model_rows() if r["gw"] == gw), None)
        if row is None:
            dec = next(
                (d for d in cache.model_decisions() if d["gw"] == gw), None,
            )
            if dec:
                # In-progress GW: the humans' totals above come from entry
                # history (live XI points, captain doubled, minus hits, no
                # auto-subs yet). Compute the model's running total on the
                # SAME basis or the table compares numbers that mean
                # different things.
                from . import modelteam
                proj = {int(k): v for k, v in
                        (cache.gw_projections(gw) or {}).items()}
                etype = {p: els[p]["element_type"]
                         for p in dec["squad"] if p in els}
                try:
                    xi = modelteam._best_xi(dec["squad"], proj, etype)
                except Exception:
                    xi = list(dec["squad"])[:11]
                cap = dec.get("captain")
                live_total = (
                    sum(live.get(p, 0) for p in xi)
                    + (live.get(cap, 0) if cap in xi else 0)
                    - 4 * dec.get("hits", 0)
                )
                row = {
                    "gw": gw,
                    "players": [
                        {"id": p, "name": els.get(p, {}).get("web_name", f"#{p}"),
                         "points": live.get(p, 0), "in_xi": p in xi,
                         "captain": p == cap}
                        for p in dec["squad"]
                    ],
                    "points": live_total,
                    "live": True,
                    "chip": None,
                    "hits": dec.get("hits", 0),
                    "transfers": dec.get("transfers", {}),
                    "pending": True,
                }
    except Exception:
        row = None
    if row:
        model_sq = {
            "players": [
                {"id": p["id"], "name": p["name"],
                 "club": tnames.get(els.get(p["id"], {}).get("team"), ""),
                 "position": 0,
                 "price_now": els.get(p["id"], {}).get("now_cost", 0) / 10.0,
                 "points": p["points"],
                 "captain": p["captain"], "in_xi": p["in_xi"]}
                for p in row["players"]
            ],
            "gw_points": row["points"],
            "live": row.get("live", False),
            "league_independent": True,
            "chip": row.get("chip"),
            "note": (
                "the autonomous model team's actual squad this gameweek "
                "(its own draft + its own transfers; solver-validated "
                "within budget at each deadline"
                + (f"; {row['hits']} hit(s) taken" if row.get("hits") else "")
                + ")"
            ),
            "transfers": row.get("transfers", {}),
        }
        return {"gw": gw, "me": mine, "model": model_sq, "rivals": people}

    return {"gw": gw, "me": mine, "model": None, "rivals": people}

    model_sq = None
    if mine:
        rec_in: list[int] = []
        rec_out: list[int] = []
        try:
            for s in cache.scores():
                if s["gw"] == gw:
                    cf = s.get("counterfactual") or {}
                    rec_in = cf.get("recommended", {}).get("in", [])
                    rec_out = cf.get("recommended", {}).get("out", [])
        except Exception:
            pass
        def mini(pid: int, rec: bool) -> dict:
            el = els.get(pid, {})
            return {
                "id": pid, "name": el.get("web_name", f"#{pid}"),
                "club": tnames.get(el.get("team"), "?"),
                "position": 0, "price_now": el.get("now_cost", 0) / 10.0,
                "points": live.get(pid, 0), "captain": False,
                "in_xi": True, "recommended_in": rec,
            }

        if len(rec_in) >= 11 and not rec_out:
            # A full recommended draft (GW1 case): the model's squad IS
            # the draft, not my squad plus fifteen extras.
            players = [mini(pid, True) for pid in rec_in]
        else:
            players = [p for p in mine["players"] if p["id"] not in rec_out]
            for pid in rec_in:
                if not any(p["id"] == pid for p in players):
                    players.append(mini(pid, True))
        model_sq = {
            "players": players,
            "note": (
                "your squad with the pre-deadline recommended transfers "
                "applied - full 15 within your real budget; prices shown "
                "are current, not purchase"
            ),
            "transfers": {"in": rec_in, "out": rec_out},
        }

    return {"gw": gw, "me": mine, "model": model_sq, "rivals": people}


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

    # Human-readable headline: the autonomous model team vs the league.
    headline = None
    try:
        from .notify import _load_env

        env = _load_env()
        team_id = int(env.get("RIVALR_TEAM_ID") or 0)
        league_id = int(env.get("RIVALR_LEAGUE_ID") or 0)
        rows = cache.model_rows()
        if rows and team_id and league_id:
            client = client_factory()
            through = rows[-1]["gw"]
            model_total = rows[-1]["total"]

            standings = client.league_standings(league_id)
            entries = standings["standings"]["results"] or []
            vs = []
            for r in entries:
                hist = client.entry_history(r["entry"]).get("current", [])
                # fair comparison: totals through the model's last settled GW
                total = sum(
                    h["points"] - h.get("event_transfers_cost", 0)
                    for h in hist if h["event"] <= through
                )
                vs.append({
                    "name": r.get("player_name", str(r["entry"])),
                    "is_me": r["entry"] == team_id,
                    "points": total,
                    "diff": model_total - total,
                })
            vs.sort(key=lambda v: -v["points"])
            rank = 1 + sum(1 for v in vs if v["points"] > model_total)

            captain = {"model": 0, "human": 0, "tie": 0}
            for row in rows:
                gw = row["gw"]
                m_cap = next(
                    (p["points"] for p in row["players"] if p.get("captain")), None
                )
                try:
                    picks = client.entry_picks(team_id, gw)["picks"]
                    my_cap_id = next(
                        p["element"] for p in picks if p.get("is_captain")
                    )
                    live_pts = {
                        el["id"]: el["stats"]["total_points"]
                        for el in client.event_live(gw)["elements"]
                    }
                    h_cap = live_pts.get(my_cap_id, 0)
                except Exception:
                    continue
                if m_cap is None:
                    continue
                if m_cap > h_cap:
                    captain["model"] += 1
                elif h_cap > m_cap:
                    captain["human"] += 1
                else:
                    captain["tie"] += 1

            edges = [
                (r["gw"], (r.get("counterfactual") or {}).get(
                    "recommendation_edge", 0))
                for r in cache.scores() if r.get("counterfactual")
            ]
            transfer_rec = {
                "gained": sum(1 for _, e in edges if e > 0),
                "lost": sum(1 for _, e in edges if e < 0),
                "net_points": sum(e for _, e in edges),
            }
            headline = {
                "through_gw": through,
                "model_points": model_total,
                "model_rank": rank,
                "league_size": len(vs) + 1,
                "vs": vs,
                "captain_record": captain,
                "transfer_record": transfer_rec,
                "hits_taken": sum(r.get("hits", 0) for r in rows),
            }
    except Exception:
        log.warning("accuracy headline unavailable", exc_info=True)

    return {"backtest": BACKTEST, "live": live, "headline": headline}


@app.get("/myleagues")
def myleagues(team_id: int = Query(...)):
    """The classic mini-leagues this entry is in, for the league picker.

    entry/{id}/ lists every classic league including the giant system
    ones (Overall, country, region, GW leagues) - those carry
    league_type 's'; private/invitational mini-leagues carry 'x'. Only
    the mini-leagues are useful for rival analysis, so system leagues
    are filtered out. Manual league-ID entry stays as the fallback."""
    client = client_factory()
    try:
        entry = client.entry(team_id)
    except Exception:
        raise HTTPException(404, "team not found")
    leagues = []
    for lg in (entry.get("leagues") or {}).get("classic", []):
        if lg.get("league_type") == "s":
            continue  # Overall / country / region / broadcast leagues
        leagues.append({
            "league_id": lg["id"],
            "name": lg.get("name", str(lg["id"])),
            "entry_rank": lg.get("entry_rank"),
            "entry_last_rank": lg.get("entry_last_rank"),
        })
    return {
        "team_id": team_id,
        "team_name": entry.get("name", ""),
        "player_name": f"{entry.get('player_first_name', '')} "
                       f"{entry.get('player_last_name', '')}".strip(),
        "leagues": leagues,
    }


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
