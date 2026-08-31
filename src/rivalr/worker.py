"""Railway worker: hourly scheduler + brief pre-warmer.

    python -m rivalr.worker

Each tick: run snapshot --auto (same code path as the local Task
Scheduler entry), then pre-warm the brief cache so visitors never sit
through a cold solve:

  - post-GW settle: once a gameweek's data is checked, re-solve every
    requested pair (marker file prevents repeats)
  - pre-deadline: within 6h of the deadline, keep entries fresher than
    the API's 30-min window freshness (tick interval drops to 15 min)
  - quiet periods: refresh entries older than 6h

Only genuinely new team/league pairs ever hit a cold solve.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone

log = logging.getLogger("rivalr.worker")

INTERVAL_S = 3600
INTERVAL_NEAR_DEADLINE_S = 900
PREWARM_WINDOW_H = 6
MAX_SOLVES_PER_TICK = 8


def score_settled_gws() -> None:
    """Once a GW's data is checked, score its ledger snapshot (paper
    buckets + base-vs-final) so /accuracy has live numbers without any
    manual step. Skips GWs with no snapshot or an existing score file."""
    from . import ledger
    from .fetch import FPLClient

    import json as _json

    from .store import make_store

    client = FPLClient()
    st = make_store()
    settled = [
        ev["id"] for ev in client.bootstrap()["events"]
        if ev.get("finished") and ev.get("data_checked")
    ]
    for gw in settled:
        score_file = ledger.LEDGER_DIR / f"gw{gw}_score.json"
        if not score_file.exists():
            try:
                ledger._latest_ledger_for(gw, ledger.LEDGER_DIR)
            except FileNotFoundError:
                continue  # no snapshot to score (e.g. pre-launch gameweeks)
            log.info("auto-scoring settled gw%d", gw)
            result = ledger.score_gw(client, gw)
            log.info("gw%d scored: All rmse=%s", gw,
                     result["accuracy"]["All"]["rmse"])
        # sync to Postgres (idempotent upsert) so /accuracy on the web
        # service sees scores that live on this worker's volume
        try:
            st.put_score(
                gw, _json.loads(score_file.read_text(encoding="utf-8"))
            )
        except Exception:
            log.warning("score sync to store failed for gw%d", gw,
                        exc_info=True)


def prewarm_tick() -> None:
    from . import briefdata, ledger, snapshot
    from .fetch import FPLClient
    from .store import STALE_REFRESH_S, WINDOW_TTL_S, make_store

    st = make_store()
    client = FPLClient()
    gw, deadline = snapshot._next_deadline(client)
    hours_to_dl = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600

    # post-GW settle: force-refresh everything once per settled gameweek
    force = False
    settled = [
        ev["id"] for ev in client.bootstrap()["events"]
        if ev.get("finished") and ev.get("data_checked")
    ]
    marker = None
    if settled:
        marker = ledger.LEDGER_DIR / f"WARMED_after_gw{max(settled)}.txt"
        force = not marker.exists()

    if force:
        threshold = 0
    elif hours_to_dl <= PREWARM_WINDOW_H:
        threshold = WINDOW_TTL_S  # keep fresher than the API's window bar
    else:
        threshold = STALE_REFRESH_S

    keys = st.stale_keys(gw, threshold)[:MAX_SOLVES_PER_TICK]
    if not keys:
        log.info("prewarm: nothing stale (%d pairs tracked)", len(st.pairs()))
        return
    log.info("prewarm: %d pair(s) to refresh (threshold %ds, force=%s)",
             len(keys), threshold, force)
    from .store import cache_key

    for team, league, mode, target, key_gw in keys:
        try:
            payload = briefdata.build_for_mode(
                client, team, league, mode, target or None
            )
            st.put(cache_key(team, league, mode, target, key_gw), payload)
            log.info("prewarm: cached %s/%s %s target=%s", team, league,
                     mode, target or "-")
        except Exception:
            log.exception("prewarm failed for %s/%s (continuing)", team, league)
    if force and marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    return


def _serve_health() -> None:
    """Minimal /health endpoint so Railway's healthcheck (shared
    railway.toml with the web service) passes for the worker too."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(os.environ.get("PORT", "8000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "role": "worker"}).encode()
            self.send_response(200 if self.path == "/health" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/health":
                self.wfile.write(body)

        def log_message(self, *a):  # keep worker logs clean
            pass

    srv = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("worker health endpoint on :%d/health", port)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    team = os.environ.get("RIVALR_TEAM_ID")
    league = os.environ.get("RIVALR_LEAGUE_ID")
    if not team or not league:
        log.error("RIVALR_TEAM_ID / RIVALR_LEAGUE_ID must be set")
        sys.exit(2)
    _serve_health()

    from . import snapshot

    log.info("worker up: hourly snapshot --auto for team %s league %s", team, league)
    while True:
        try:
            sys.argv = ["snapshot", "--team", team, "--league", league, "--auto"]
            rc = snapshot.main()
            log.info("snapshot --auto exited %s", rc)
        except SystemExit as exc:
            log.info("snapshot --auto exited %s", exc.code)
        except Exception:
            log.exception("worker iteration failed (continuing)")

        try:
            score_settled_gws()
        except Exception:
            log.exception("auto-score tick failed (continuing)")

        try:
            prewarm_tick()
        except Exception:
            log.exception("prewarm tick failed (continuing)")

        # Tick faster near the deadline so window-fresh entries exist.
        interval = INTERVAL_S
        try:
            from .fetch import FPLClient

            _, deadline = snapshot._next_deadline(FPLClient())
            hrs = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
            if 0 <= hrs <= PREWARM_WINDOW_H + 1:
                interval = INTERVAL_NEAR_DEADLINE_S
        except Exception:
            pass
        time.sleep(interval)


if __name__ == "__main__":
    main()
