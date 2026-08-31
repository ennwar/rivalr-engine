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

from . import defcon, ledger, minutes, model, optimise, rivals, uncertainty
from .fetch import FPLClient
from .notify import require_config, telegram_send

log = logging.getLogger("rivalr.snapshot")

WINDOW_HOURS = 4
RUN_LOG = ledger.LEDGER_DIR / "run_log.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_publish(gw: int, path: Path, partial: bool) -> bool:
    """Best-effort: commit + push the new ledger file to origin.

    Never raises and never blocks the snapshot - a failed push is logged
    (and recorded in the run log by the caller) but the local ledger file
    is already safe on disk."""
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=120,
        )

    try:
        rel = path.resolve().relative_to(REPO_ROOT)
        r = run("add", str(rel))
        if r.returncode:
            raise RuntimeError(f"git add: {r.stderr.strip()}")
        msg = f"ledger: gw{gw} snapshot {path.name}"
        if partial:
            msg += " (partial)"
        r = run("commit", "-m",
                msg + "\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
        if r.returncode:
            raise RuntimeError(f"git commit: {(r.stdout + r.stderr).strip()}")
        r = run("push")
        if r.returncode:
            raise RuntimeError(f"git push: {r.stderr.strip()}")
        log.info("ledger pushed to origin: %s", path.name)
        return True
    except Exception as exc:
        log.warning("LEDGER GIT PUSH FAILED (snapshot unaffected): %s", exc)
        return False


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
        # LoadXml pattern: PS 5.1's WinRT DOM enumeration
        # (GetElementsByTagName indexing) is broken on this machine -
        # verified 2026-08-15, see dry-run notes.
        safe = (message.replace("&", "&amp;").replace("<", "&lt;")
                       .replace(">", "&gt;").replace("'", "&apos;"))
        xml = ("<toast><visual><binding template=\"ToastText02\">"
               "<text id=\"1\">rivalr snapshot</text>"
               f"<text id=\"2\">{safe}</text>"
               "</binding></visual></toast>")
        ps = (
            "[Windows.UI.Notifications.ToastNotificationManager, "
            "Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null; "
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
            "ContentType=WindowsRuntime] | Out-Null; "
            "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument; "
            f"$xml.LoadXml('{xml}'); "
            "$n = [Windows.UI.Notifications.ToastNotification]::new($xml); "
            "[Windows.UI.Notifications.ToastNotificationManager]"
            "::CreateToastNotifier('rivalr').Show($n)"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0 or r.stderr.strip():
            log.warning("toast failed (alert file still written): %s",
                        r.stderr.strip()[:200])
    except Exception:
        log.warning("toast notification failed (alert file still written)")


def _next_deadline(client: FPLClient) -> tuple[int, datetime]:
    from . import gameweek

    return gameweek.next_deadline(client)


def _check_missed_windows(client: FPLClient, now: datetime) -> None:
    """Heartbeat: if any deadline passed in the last 24h with NO snapshot
    written, alarm once (marker file prevents repeats). Silence must
    never mean 'probably fine'."""
    try:
        events = client.bootstrap()["events"]
    except Exception:
        return
    for ev in events:
        dt = datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))
        if not (timedelta(0) <= now - dt <= timedelta(hours=24)):
            continue
        gw = ev["id"]
        if _has_snapshot(gw):
            continue
        marker = ledger.LEDGER_DIR / f"MISSED_gw{gw}.txt"
        if marker.exists():
            continue
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{now.isoformat()} window closed, no snapshot\n",
                          encoding="utf-8")
        msg = (f"rivalr ALARM: GW{gw} deadline ({ev['deadline_time']}) "
               f"passed with NO ledger snapshot written. The machine was "
               f"probably off/asleep during the window.")
        _alert(gw, msg)
        telegram_send(msg)
        _log_run({"gw": gw, "action": "missed-window",
                  "deadline": ev["deadline_time"]})


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

    # Injury flags move in the hours before a deadline: bypass the 6h TTL
    # and force a fresh bootstrap-static. On failure, fall back to the
    # cached copy (its age is logged by the caller).
    try:
        client.get("bootstrap-static/", force=True)
    except Exception as exc:
        log.warning("fresh bootstrap fetch failed, using cached copy: %r", exc)

    try:
        elements = client.bootstrap()["elements"]
    except Exception as exc:
        failures.append(f"bootstrap: {exc!r}")

    base: dict[int, list[float]] = {}
    dc_corr: dict[int, list[float]] = {}
    try:
        raw = model.project_all(client, horizon=horizon)
        try:
            est = {pid: minutes.estimate_minutes(client, pid) for pid in raw}
            base = minutes.apply_minutes(raw, est)
        except Exception as exc:
            failures.append(f"minutes: {exc!r}")
            base = raw
            est = {}
        try:
            dc_corr = defcon.DefConModel(client).corrections(
                list(base), est, horizon=horizon
            )
        except Exception as exc:
            failures.append(f"defcon: {exc!r}")
            dc_corr = {}
        projections = {
            pid: [
                round(x + (dc_corr.get(pid) or [0.0] * len(xs))[i], 3)
                for i, x in enumerate(xs)
            ]
            for pid, xs in base.items()
        }
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
                try:
                    mgr = uncertainty.player_flags(client)
                except Exception:
                    mgr = {}
                ins = chosen.get("transfers_in", [])
                recommendation.update(
                    transfers_in=ins,
                    transfers_out=chosen.get("transfers_out", []),
                    captain=chosen.get("captain"),
                    expected_points=chosen.get("expected_points"),
                    manager_change_ins=[
                        p for p in ins
                        if "MGR_CHG" in mgr.get(p, {}).get("kinds", [])
                    ],
                    new_club_ins=[
                        p for p in ins
                        if "NEW_CLUB" in mgr.get(p, {}).get("kinds", [])
                    ],
                )
            else:
                failures.append("solver: all modes returned None")
        except Exception as exc:
            failures.append(f"solver: {exc!r}")

    partial = bool(failures)
    coverage = ledger.full_coverage(projections, elements) if elements else {
        pid: xs for pid, xs in projections.items()
    }
    # What we knew at snapshot time: news + availability per player, so
    # post-GW scoring can separate "we knew" from "we couldn't have".
    availability = {
        el["id"]: {
            "status": el.get("status"),
            "news": el.get("news") or "",
            "news_added": el.get("news_added"),
            "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
        }
        for el in elements
    }
    path = ledger.record_predictions(
        gw, coverage, recommendation, partial=partial, failures=failures,
        layers={"base": base, "defcon": dc_corr},
        availability=availability,
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

    # Loud startup check: missing telegram config must never be silent -
    # but it also must never stop the ledger, which outranks everything.
    try:
        require_config()
    except RuntimeError as exc:
        log.error("%s", exc)
        _alert(0, f"TELEGRAM NOT CONFIGURED: {exc}")

    # Element count goes into every run-log entry: the player universe
    # grows through the transfer window, and we want to watch it move.
    try:
        n_elements = len(client.bootstrap()["elements"])
    except Exception:
        n_elements = None

    try:
        gw, deadline = _next_deadline(client)
    except Exception as exc:
        _alert(0, f"could not read next deadline: {exc!r}")
        _log_run({"gw": None, "action": "failed-no-deadline", "error": repr(exc),
                  "elements": n_elements})
        return 2

    now = datetime.now(timezone.utc)
    if args.auto:
        _check_missed_windows(client, now)
        if now < deadline - timedelta(hours=WINDOW_HOURS) or now >= deadline:
            _log_run({
                "gw": gw, "action": "skip-outside-window",
                "elements": n_elements,
                "deadline": deadline.isoformat(),
                "window_opens": (deadline - timedelta(hours=WINDOW_HOURS)).isoformat(),
            })
            log.info("outside snapshot window for gw%d (deadline %s); nothing to do",
                     gw, deadline.isoformat())
            return 0
        if _has_snapshot(gw):
            _log_run({"gw": gw, "action": "skip-already-exists",
                      "elements": n_elements, "deadline": deadline.isoformat()})
            log.info("snapshot for gw%d already exists; nothing to do", gw)
            return 0

    try:
        path, partial, failures = take_snapshot(
            client, args.team, args.league, gw,
            mode=args.mode, target_id=args.target, horizon=args.horizon,
        )
    except Exception as exc:
        # The one thing that must never fail, failed. Scream.
        msg = f"LEDGER SNAPSHOT FAILED for gw{gw}: {exc!r}"
        _alert(gw, msg)
        telegram_send(f"rivalr ALARM: {msg}")
        _log_run({"gw": gw, "action": "failed", "error": repr(exc),
                  "elements": n_elements, "deadline": deadline.isoformat()})
        return 2

    pushed = _git_publish(gw, path, partial)
    try:  # cross-service visibility (/health reads this from Postgres)
        from .store import make_store

        make_store().record_snapshot(gw, path.name, partial)
    except Exception:
        log.warning("snapshot meta store failed (ledger unaffected)",
                    exc_info=True)
    age = client.cache_age("bootstrap-static/")
    _log_run({
        "gw": gw, "action": "written", "partial": partial,
        "failures": failures, "ledger_file": path.name,
        "pushed": pushed,
        "elements": n_elements,
        "bootstrap_age_s": round(age, 1) if age is not None else None,
        "deadline": deadline.isoformat(),
    })

    # Confirmation to Telegram (works while asleep/away, unlike toasts).
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
        cov = snap.get("coverage", {})
        players_line = (f"{cov.get('projected', '?')}/"
                        f"{cov.get('bootstrap_elements_at_snapshot', '?')} projected")
    except Exception:
        players_line = "?"
    status = "PARTIAL" if partial else "OK"
    msg = (
        f"rivalr GW{gw} snapshot {status}\n"
        f"time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"file: {path.name}\n"
        f"players: {players_line}\n"
        f"partial: {partial}\n"
        f"bootstrap age: {age:.0f}s\n"
        f"pushed to GitHub: {pushed}"
    )
    if failures:
        msg += "\nfailures: " + "; ".join(failures)
    telegram_send(msg)

    if partial:
        _alert(gw, f"gw{gw} snapshot written but PARTIAL: {'; '.join(failures)}")
        return 1
    log.info("clean snapshot for gw%d: %s", gw, path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
