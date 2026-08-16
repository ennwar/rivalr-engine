"""Append-only prediction ledger + post-GW accuracy scoring.

Coverage rule: the ledger records EVERY element in bootstrap-static at
snapshot time. Players the model can't project carry a null projection —
pool filtering is a solver concern and must never shrink the scoring
universe, or accuracy numbers bias in our favour.

Naming / append-only rule:
    gw{N}.json            first snapshot for the GW
    gw{N}_v2.json, _v3…   later snapshots (logged loudly, nothing overwritten)
    gw{N}_score.json      scoring output (not a snapshot)
    *_test.json           plumbing artifacts — never scored

Scoring (python -m rivalr.score --gw N) uses the highest-versioned real
snapshot and reports TWO bucket views:

  primary  - the OpenFPL paper's buckets (arXiv 2508.09992 Table 4),
             directly comparable to the published benchmark:
               Zeros    did not play (0 minutes)
               Blanks   played, <= 2 points
               Tickers  3-4 points
               Haulers  5+ points
  secondary - legacy point-range buckets (more intuitive for a
             user-facing report): 0 / 1-3 / 4-9 / 10+

plus a counterfactual: points from the recommended transfers vs points
from the transfers actually made.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from .fetch import FPLClient

log = logging.getLogger("rivalr.ledger")

import os

LEDGER_DIR = Path(os.environ.get("RIVALR_LEDGER_DIR", "logs/predictions"))

LEGACY_BUCKETS = [
    ("Zeros", 0, 0),
    ("Blanks", 1, 3),
    ("Tickers", 4, 9),
    ("Haulers", 10, 10_000),
]


def paper_bucket(minutes: int, points: int) -> str:
    """OpenFPL paper (arXiv 2508.09992) Table 4 buckets."""
    if minutes == 0:
        return "Zeros"
    if points <= 2:
        return "Blanks"
    if points <= 4:
        return "Tickers"
    return "Haulers"


BUCKET_NAMES = ["Zeros", "Blanks", "Tickers", "Haulers"]


def _bucket_stats(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0, "rmse": None, "mae": None}
    return {
        "n": len(errs),
        "rmse": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 3),
        "mae": round(sum(abs(e) for e in errs) / len(errs), 3),
    }

_SNAPSHOT_RE = re.compile(r"^gw(\d+)(?:_v(\d+))?\.json$")


def full_coverage(
    projections: dict[int, list[float]], elements: list[dict]
) -> dict[int, list[float] | None]:
    """Every bootstrap element; None where we have no projection."""
    return {el["id"]: projections.get(el["id"]) for el in elements}


def _snapshot_path(gw: int, ledger_dir: Path) -> Path:
    """Next free versioned filename for this GW. Never overwrites."""
    base = ledger_dir / f"gw{gw}.json"
    if not base.exists():
        return base
    v = 2
    while (ledger_dir / f"gw{gw}_v{v}.json").exists():
        v += 1
    path = ledger_dir / f"gw{gw}_v{v}.json"
    log.warning(
        "LEDGER: snapshot for gw%d already exists - writing %s "
        "(append-only guarantee held)", gw, path.name,
    )
    return path


def record_predictions(
    gw: int,
    projections: dict[int, list[float] | None],
    recommendation: dict,
    ledger_dir: str | Path | None = None,
    partial: bool = False,
    failures: list[str] | None = None,
    layers: dict[str, dict] | None = None,
    availability: dict[int, dict] | None = None,
) -> Path:
    """Write the pre-deadline snapshot. Never overwrites an existing file.

    ledger_dir defaults to LEDGER_DIR at call time (late-bound, so tests
    that patch ledger.LEDGER_DIR actually redirect the write)."""
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else LEDGER_DIR
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(gw, ledger_dir)
    projected = sum(1 for v in projections.values() if v is not None)
    payload = {
        "gw": gw,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "partial": partial,
        "failures": failures or [],
        "coverage": {
            # bootstrap-static element count at snapshot time; scoring
            # compares this against who actually played to surface players
            # who registered after the snapshot ("unrostered").
            "bootstrap_elements_at_snapshot": len(projections),
            "projected": projected,
            "unprojected": len(projections) - projected,
        },
        # "projections" is the FINAL number (base + corrections) and is
        # what gets scored; "layers" preserves each component separately
        # so we can score which layer earns its place.
        "projections": {str(pid): xs for pid, xs in projections.items()},
        "layers": {
            name: {str(pid): xs for pid, xs in vals.items()}
            for name, vals in (layers or {}).items()
        },
        # Point-in-time knowledge: news/flags as they stood at snapshot,
        # for later "what we knew vs what happened" scoring.
        "availability": {
            str(pid): a for pid, a in (availability or {}).items()
        },
        "recommendation": recommendation,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "ledger written: %s (%d/%d projected%s)",
        path, projected, len(projections), ", PARTIAL" if partial else "",
    )
    return path


def _latest_ledger_for(gw: int, ledger_dir: Path) -> Path:
    """Highest-versioned real snapshot. *_test.json and *_score.json never
    match; timestamped legacy names never match."""
    best: tuple[int, Path] | None = None
    for p in ledger_dir.iterdir():
        m = _SNAPSHOT_RE.match(p.name)
        if not m or int(m.group(1)) != gw:
            continue
        version = int(m.group(2) or 1)
        if best is None or version > best[0]:
            best = (version, p)
    if best is None:
        raise FileNotFoundError(f"no ledger snapshot for gw{gw} in {ledger_dir}")
    return best[1]


def actual_stats(client: FPLClient, gw: int) -> dict[int, dict]:
    """{player_id: {points, minutes}} for a finished GW."""
    live = client.event_live(gw)
    return {
        el["id"]: {
            "points": el["stats"]["total_points"],
            "minutes": el["stats"].get("minutes", 0),
        }
        for el in live["elements"]
    }


def score_gw(
    client: FPLClient,
    gw: int,
    ledger_dir: str | Path | None = None,
) -> dict:
    """RMSE/MAE per outcome bucket + transfer counterfactual for one GW."""
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else LEDGER_DIR
    ledger_path = _latest_ledger_for(gw, ledger_dir)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    stats = actual_stats(client, gw)
    actuals = {pid: s["points"] for pid, s in stats.items()}

    # (pid, predicted, actual points, actual minutes)
    pairs: list[tuple[int, float, int, int]] = []
    unprojected = 0
    for pid_str, xs in ledger["projections"].items():
        pid = int(pid_str)
        if xs is None:
            unprojected += 1
            continue
        if pid in stats and xs:
            pairs.append(
                (pid, float(xs[0]), stats[pid]["points"], stats[pid]["minutes"])
            )

    paper_errs: dict[str, list[float]] = {name: [] for name in BUCKET_NAMES}
    legacy_errs: dict[str, list[float]] = {name: [] for name in BUCKET_NAMES}
    all_errs: list[float] = []
    for _, pred, pts, mins in pairs:
        err = pred - pts
        all_errs.append(err)
        paper_errs[paper_bucket(mins, pts)].append(err)
        for name, lo, hi in LEGACY_BUCKETS:
            if lo <= pts <= hi:
                legacy_errs[name].append(err)
                break

    table = {name: _bucket_stats(errs) for name, errs in paper_errs.items()}
    table["All"] = _bucket_stats(all_errs)
    table_legacy = {name: _bucket_stats(errs) for name, errs in legacy_errs.items()}
    table_legacy["All"] = table["All"]

    # Base-layer scoring (paper buckets) when the snapshot carried layers:
    # lets us show whether the DefCon correction earns its place.
    table_base = None
    base_layer = ledger.get("layers", {}).get("base") or {}
    if base_layer:
        base_errs: dict[str, list[float]] = {n: [] for n in BUCKET_NAMES}
        base_all: list[float] = []
        for pid_str, xs in base_layer.items():
            pid = int(pid_str)
            if xs and pid in stats:
                err = float(xs[0]) - stats[pid]["points"]
                base_all.append(err)
                base_errs[
                    paper_bucket(stats[pid]["minutes"], stats[pid]["points"])
                ].append(err)
        table_base = {n: _bucket_stats(e) for n, e in base_errs.items()}
        table_base["All"] = _bucket_stats(base_all)

    counterfactual = _transfer_counterfactual(client, gw, ledger, actuals)

    # Players who scored (non-zero) but have NO ledger entry at all: they
    # registered after the snapshot. Distinct from nulls (rostered but
    # unprojectable) - if this is material, the coverage claim is weaker
    # than the null count suggests.
    ledger_ids = {int(k) for k in ledger["projections"]}
    names = {}
    try:
        names = {el["id"]: el["web_name"] for el in client.bootstrap()["elements"]}
    except Exception:
        log.warning("could not resolve names for unrostered players")
    unrostered = sorted(
        (
            {"id": pid, "name": names.get(pid, f"#{pid}"), "points": pts}
            for pid, pts in actuals.items()
            if pid not in ledger_ids and pts != 0
        ),
        key=lambda r: -r["points"],
    )

    result = {
        "gw": gw,
        "ledger_file": ledger_path.name,
        "ledger_partial": ledger.get("partial", False),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "unprojected_players": unprojected,
        "unrostered_at_snapshot": {
            "n": len(unrostered),
            "total_points_missed": sum(r["points"] for r in unrostered),
            "players": unrostered,
        },
        "accuracy": table,          # paper buckets (arXiv 2508.09992 Table 4)
        "accuracy_legacy": table_legacy,  # point-range buckets 0/1-3/4-9/10+
        "accuracy_base": table_base,      # base layer (no DefCon), if logged
        "counterfactual": counterfactual,
    }
    out = ledger_dir / f"gw{gw}_score.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _transfer_counterfactual(
    client: FPLClient, gw: int, ledger: dict, actuals: dict[int, int]
) -> dict:
    """Points from recommended moves vs the moves actually made."""
    rec = ledger.get("recommendation", {})
    rec_in = rec.get("transfers_in", [])
    rec_out = rec.get("transfers_out", [])
    rec_delta = sum(actuals.get(p, 0) for p in rec_in) - sum(
        actuals.get(p, 0) for p in rec_out
    )

    team_id = rec.get("team_id")
    made_in: list[int] = []
    made_out: list[int] = []
    if team_id:
        transfers = client.entry_transfers(team_id)
        for t in transfers:
            if t["event"] == gw:
                made_in.append(t["element_in"])
                made_out.append(t["element_out"])
    made_delta = sum(actuals.get(p, 0) for p in made_in) - sum(
        actuals.get(p, 0) for p in made_out
    )

    return {
        "recommended": {"in": rec_in, "out": rec_out, "points_delta": rec_delta},
        "actual": {"in": made_in, "out": made_out, "points_delta": made_delta},
        "recommendation_edge": rec_delta - made_delta,
    }


def format_score_table(result: dict) -> str:
    lines = [f"GW{result['gw']} accuracy ({result['ledger_file']})"]
    if result.get("ledger_partial"):
        lines.append("NOTE: scored against a PARTIAL snapshot")
    if result.get("unprojected_players"):
        lines.append(f"unprojected players excluded from RMSE: "
                     f"{result['unprojected_players']}")
    unr = result.get("unrostered_at_snapshot", {})
    if unr.get("n"):
        lines.append(
            f"UNROSTERED AT SNAPSHOT: {unr['n']} scorer(s), "
            f"{unr['total_points_missed']} pts outside the ledger universe"
        )
        for r in unr["players"][:10]:
            lines.append(f"  {r['name']} ({r['points']} pts)")
    def render(table: dict, title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append(f"{'Bucket':<9}{'n':>6}{'RMSE':>8}{'MAE':>8}")
        for name in ["Zeros", "Blanks", "Tickers", "Haulers", "All"]:
            row = table[name]
            rmse = f"{row['rmse']:.3f}" if row["rmse"] is not None else "-"
            mae = f"{row['mae']:.3f}" if row["mae"] is not None else "-"
            lines.append(f"{name:<9}{row['n']:>6}{rmse:>8}{mae:>8}")

    render(result["accuracy"],
           "Paper buckets (arXiv 2508.09992: DNP / <=2 / 3-4 / 5+)")
    render(result["accuracy_legacy"],
           "Legacy buckets (points 0 / 1-3 / 4-9 / 10+)")
    cf = result["counterfactual"]
    lines += [
        "",
        "Counterfactual",
        f"  recommended moves: {cf['recommended']['points_delta']:+d} pts",
        f"  actual moves:      {cf['actual']['points_delta']:+d} pts",
        f"  edge:              {cf['recommendation_edge']:+d} pts",
    ]
    return "\n".join(lines)
