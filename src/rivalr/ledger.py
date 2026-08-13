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
snapshot, with RMSE/MAE split into the OpenFPL paper's outcome buckets:

    Zeros    0 points
    Blanks   1-3 points
    Tickers  4-9 points
    Haulers  10+ points

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

LEDGER_DIR = Path("logs/predictions")

BUCKETS = [
    ("Zeros", 0, 0),
    ("Blanks", 1, 3),
    ("Tickers", 4, 9),
    ("Haulers", 10, 10_000),
]

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
    ledger_dir: str | Path = LEDGER_DIR,
    partial: bool = False,
    failures: list[str] | None = None,
) -> Path:
    """Write the pre-deadline snapshot. Never overwrites an existing file."""
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(gw, ledger_dir)
    projected = sum(1 for v in projections.values() if v is not None)
    payload = {
        "gw": gw,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "partial": partial,
        "failures": failures or [],
        "coverage": {
            "total_elements": len(projections),
            "projected": projected,
            "unprojected": len(projections) - projected,
        },
        "projections": {str(pid): xs for pid, xs in projections.items()},
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
    unprojected = 0
    for pid_str, xs in ledger["projections"].items():
        pid = int(pid_str)
        if xs is None:
            unprojected += 1
            continue
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
        "ledger_partial": ledger.get("partial", False),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "unprojected_players": unprojected,
        "accuracy": table,
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
    lines.append("")
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
