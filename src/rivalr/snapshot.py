"""Pre-deadline ledger snapshot runner. The ledger write must never fail.

    python -m rivalr.snapshot --team 2616874 --league 517089            # run now
    python -m rivalr.snapshot --team ... --league ... --auto            # scheduled

--auto is meant for a scheduler (e.g. Windows Task Scheduler, hourly):
it reads the next deadline from bootstrap-static -> events ->
deadline_time and only fires inside the window WINDOW_HOURS before the
deadline, and only if no snapshot exists yet for that GW. Outside the
window it logs a skip heartbeat and exits 0.

Degradation ladder (each failure recorded, never fatal to the snapshot):
  1. OpenFPL projections  -> fall back to FPL ep_next (model.py handles)
  2. minutes estimates    -> fall back to raw projections
  3. rivals report        -> ledger flagged partial, no EO context
  4. solver               -> ledger flagged partial, no recommendation
  5. everything           -> ledger with null projections, partial=True

Every run appends one JSON line to logs/predictions/run_log.jsonl:
  {ts, gw, action: written|skip-outside-window|skip-already-exists,
   partial, failures, ledger_file, deadline}

On a partial or failed run an alert file logs/predictions/ALERT_gw{N}.txt
is written and a Windows toast is attempted (best-effort). Exit code:
0 clean, 1 partial, 2 ledger write itself failed (should never happen).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import ledger, minutes, model, optimise, rivals
from .fetch import FPLClient

log = logging.getLogger("rivalr.snapshot")

WINDOW_HOURS = 4
RUN_LOG = ledger.LEDGER_DIR / "run_log.jsonl"


def _log_run(entry: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _alert(gw: int, message: str) -> None:
    """Loud failure surface: alert file + best-effort Windows toast."""
    path = ledger.LEDGER_DIR / f"ALERT_gw{gw}.txt"
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp}  {message}\n")
    log.error("ALERT: %s (written to %s)", message, path)
    try:  # toast is nice-to-have; never let it break the run
        ps = (
            "[Windows.UI.Notifications.ToastNotificationManager, "
            "Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
            "$t = [Windows.UI.Notifications.ToastTemplateType]::ToastText02; "
            "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($t); "
            "$x.GetElementsByTagName('text')[0].AppendChild($x.CreateTextNode('rivalr snapshot')) | Out-Null; "
            f"$x.GetElementsByTagName('text')[1].AppendChild($x.CreateTextNode('{message}')) | Out-Null; "
            "$n = [Windows.UI.Notifications.ToastNotification]::new($x); "
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('rivalr')"
            ".Show($n)"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, timeout=20,
        )
    except Exception:
        log.warning("toast notification failed (alert file still written)")


def _next_deadline(client: FPLClient) -> tuple[int, datetime]:
    for ev in client.bootstrap()["events"]:
        if ev["is_next"]:
            dt = datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))
            return ev["id"], dt
    raise RuntimeError("no upcoming gameweek in bootstrap-static")


def _has_snapshot(gw: int) -> bool:
    try:
        ledger._latest_ledger_for(gw, ledger.LEDGER_DIR)
        return True
    except FileNotFoundError:
        return False


def take_snapshot(
    client: FPLClient,
    team_id: int,
    league_id: int,
    gw: int,
    mode: str = "chase",
    target_id: int | None = None,
    horizon: int = 5,
) -> tuple[Path, bool, list[str]]:
    """Build everything that can be built, then ALWAYS write the ledger.
    Returns (path, partial, failures)."""
    failures: list[str] = []
    elements: list[dict] = []
    projections: dict[int, list[float]] = {}
    recommendation: dict = {"team_id": team_id, "mode": mode, "target_id": target_id}

    try:
        elements = client.bootstrap()["elements"]
    except Exception as exc:
        failures.append(f"bootstrap: {exc!r}")

    try:
        raw = model.project_all(client, horizon=horizon)
        try:
            est = {pid: minutes.estimate_minutes(client, pid) for pid in raw}
            projections = minutes.apply_minutes(raw, est)
        except Exception as exc:
            failures.append(f"minutes: {exc!r}")
            projections = raw
            est = {}
    except Exception as exc:
        failures.append(f"projections: {exc!r}")
        est = {}

    rep = None
    try:
        next_gw_proj = {pid: xs[0] for pid, xs in projections.items() if xs}
        rep = rivals.build_rivals_report(
            client, team_id, league_id, projections=next_gw_proj
        )
        rivals.write_rivals_report(rep)
    except Exception as exc:
        failures.append(f"rivals: {exc!r}")

    if rep is not None and projections:
        try:
            plans = optimise.solve_all_modes(
                client=client,
                team_id=team_id,
                projections=projections,
                rivals_report=rep,
                target_id=target_id,
                horizon=horizon,
                xmins={pid: e.expected_minutes for pid, e in est.items()},
                requested_mode=mode,
            )
            chosen = plans.get(mode) or plans.get("points")
            if chosen:
                recommendation.update(
                    transfers_in=chosen.get("transfers_in", []),
                    transfers_out=chosen.get("transfers_out", []),
                    captain=chosen.get("captain"),
                    expected_points=chosen.get("expected_points"),
                )
            else:
                failures.append("solver: all modes returned None")
        except Exception as exc:
            failures.append(f"solver: {exc!r}")

    partial = bool(failures)
    coverage = ledger.full_coverage(projections, elements) if elements else {
        pid: xs for pid, xs in projections.items()
    }
    path = ledger.record_predictions(
        gw, coverage, recommendation, partial=partial, failures=failures
    )
    return path, partial, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-deadline ledger snapshot")
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--league", type=int, required=True)
    parser.add_argument("--mode", default="chase",
                        choices=["points", "chase", "defend"])
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--auto", action="store_true",
        help=f"only fire within {WINDOW_HOURS}h of the next deadline, once",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    client = FPLClient()

    try:
        gw, deadline = _next_deadline(client)
    except Exception as exc:
        _alert(0, f"could not read next deadline: {exc!r}")
        _log_run({"gw": None, "action": "failed-no-deadline", "error": repr(exc)})
        return 2

    now = datetime.now(timezone.utc)
    if args.auto:
        if now < deadline - timedelta(hours=WINDOW_HOURS) or now >= deadline:
            _log_run({
                "gw": gw, "action": "skip-outside-window",
                "deadline": deadline.isoformat(),
                "window_opens": (deadline - timedelta(hours=WINDOW_HOURS)).isoformat(),
            })
            log.info("outside snapshot window for gw%d (deadline %s); nothing to do",
                     gw, deadline.isoformat())
            return 0
        if _has_snapshot(gw):
            _log_run({"gw": gw, "action": "skip-already-exists",
                      "deadline": deadline.isoformat()})
            log.info("snapshot for gw%d already exists; nothing to do", gw)
            return 0

    try:
        path, partial, failures = take_snapshot(
            client, args.team, args.league, gw,
            mode=args.mode, target_id=args.target, horizon=args.horizon,
        )
    except Exception as exc:
        # The one thing that must never fail, failed. Scream.
        _alert(gw, f"LEDGER SNAPSHOT FAILED for gw{gw}: {exc!r}")
        _log_run({"gw": gw, "action": "failed", "error": repr(exc),
                  "deadline": deadline.isoformat()})
        return 2

    _log_run({
        "gw": gw, "action": "written", "partial": partial,
        "failures": failures, "ledger_file": path.name,
        "deadline": deadline.isoformat(),
    })
    if partial:
        _alert(gw, f"gw{gw} snapshot written but PARTIAL: {'; '.join(failures)}")
        return 1
    log.info("clean snapshot for gw%d: %s", gw, path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
