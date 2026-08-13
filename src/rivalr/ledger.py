"""Append-only prediction ledger + post-GW accuracy scoring.

Before every deadline, report.py calls record_predictions() which writes
logs/predictions/gw{N}.json — every projection, the recommendation, and a
timestamp. Files are never overwritten: if gw{N}.json exists, a timestamped
sibling is written instead.

After a GW finishes, run:

    python -m rivalr.score --gw N

which scores the ledger against actual points with RMSE/MAE split into the
OpenFPL paper's outcome buckets:

    Zeros    0 points
    Blanks   1-3 points
    Tickers  4-9 points
    Haulers  10+ points

plus a counterfactual: points from the recommended transfers vs points from
the transfers actually made.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from .fetch import FPLClient

log = logging.getLogger("rivalr.ledger")

LEDGER_DIR = Path("logs/predictions")

BUCKETS = [
    ("Zeros", 0, 0),
    ("Blanks", 1, 3),
    ("Tickers", 4, 9),
    ("Haulers", 10, 10_000),
]


def record_predictions(
    gw: int,
    projections: dict[int, list[float]],
    recommendation: dict,
    ledger_dir: str | Path = LEDGER_DIR,
) -> Path:
    """Write the pre-deadline snapshot. Never overwrites an existing file."""
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = ledger_dir / f"gw{gw}.json"
    if path.exists():
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        path = ledger_dir / f"gw{gw}_{stamp}.json"
        log.info("gw%d ledger exists; appending as %s", gw, path.name)
    payload = {
        "gw": gw,
        "recorded_at": now.isoformat(),
        "projections": {str(pid): xs for pid, xs in projections.items()},
        "recommendation": recommendation,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("ledger written: %s", path)
    return path


def _latest_ledger_for(gw: int, ledger_dir: Path) -> Path:
    candidates = sorted(ledger_dir.glob(f"gw{gw}*.json"))
    if not candidates:
        raise FileNotFoundError(f"no ledger file for gw{gw} in {ledger_dir}")
    return candidates[-1]  # timestamped siblings sort after the base file


def actual_points(client: FPLClient, gw: int) -> dict[int, int]:
    live = client.event_live(gw)
    return {el["id"]: el["stats"]["total_points"] for el in live["elements"]}


def score_gw(
    client: FPLClient,
    gw: int,
    ledger_dir: str | Path = LEDGER_DIR,
) -> dict:
    """RMSE/MAE per outcome bucket + transfer counterfactual for one GW."""
    ledger_dir = Path(ledger_dir)
    ledger_path = _latest_ledger_for(gw, ledger_dir)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    actuals = actual_points(client, gw)

    pairs: list[tuple[int, float, int]] = []  # (pid, predicted_gw1, actual)
    for pid_str, xs in ledger["projections"].items():
        pid = int(pid_str)
        if pid in actuals and xs:
            pairs.append((pid, float(xs[0]), actuals[pid]))

    table = {}
    for name, lo, hi in BUCKETS:
        bucket = [(p, a) for _, p, a in pairs if lo <= a <= hi]
        if bucket:
            errs = [p - a for p, a in bucket]
            table[name] = {
                "n": len(bucket),
                "rmse": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 3),
                "mae": round(sum(abs(e) for e in errs) / len(errs), 3),
            }
        else:
            table[name] = {"n": 0, "rmse": None, "mae": None}
    all_errs = [p - a for _, p, a in pairs]
    table["All"] = {
        "n": len(pairs),
        "rmse": round(math.sqrt(sum(e * e for e in all_errs) / len(all_errs)), 3)
        if all_errs else None,
        "mae": round(sum(abs(e) for e in all_errs) / len(all_errs), 3)
        if all_errs else None,
    }

    counterfactual = _transfer_counterfactual(client, gw, ledger, actuals)

    result = {
        "gw": gw,
        "ledger_file": ledger_path.name,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": table,
        "counterfactual": counterfactual,
    }
    out = ledger_dir / f"gw{gw}_score.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _transfer_counterfactual(
    client: FPLClient, gw: int, ledger: dict, actuals: dict[int, int]
) -> dict:
    """Points from recommended moves vs the moves actually made.

    Recommended: from the ledger's recommendation block (transfers_in/out).
    Actual: from the entry's transfer log for this GW.
    Both are scored as sum(actual points of ins) - sum(actual points of outs).
    """
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
    lines = [f"GW{result['gw']} accuracy ({result['ledger_file']})", ""]
    lines.append(f"{'Bucket':<9}{'n':>6}{'RMSE':>8}{'MAE':>8}")
    for name in ["Zeros", "Blanks", "Tickers", "Haulers", "All"]:
        row = result["accuracy"][name]
        rmse = f"{row['rmse']:.3f}" if row["rmse"] is not None else "-"
        mae = f"{row['mae']:.3f}" if row["mae"] is not None else "-"
        lines.append(f"{name:<9}{row['n']:>6}{rmse:>8}{mae:>8}")
    cf = result["counterfactual"]
    lines += [
        "",
        "Counterfactual",
        f"  recommended moves: {cf['recommended']['points_delta']:+d} pts",
        f"  actual moves:      {cf['actual']['points_delta']:+d} pts",
        f"  edge:              {cf['recommendation_edge']:+d} pts",
    ]
    return "\n".join(lines)
