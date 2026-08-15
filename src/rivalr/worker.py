"""Railway worker: the hourly scheduler, same code path as local.

    python -m rivalr.worker

Runs snapshot --auto once an hour, forever. Identical logic to the
Windows Task Scheduler entry - do not run both against the same league
long-term or every window produces a local AND a cloud snapshot
(append-only versioning keeps both, but pick one as canonical).
"""

from __future__ import annotations

import logging
import os
import sys
import time

log = logging.getLogger("rivalr.worker")

INTERVAL_S = 3600


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
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
