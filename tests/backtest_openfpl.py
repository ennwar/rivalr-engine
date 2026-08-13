"""Backtest: validate OUR OpenFPL feature pipeline against the paper.

Reference: OpenFPL (arXiv 2508.09992), Table 4 - one-gameweek-ahead RMSE
on 2024-25, evaluation window GW32-GW38:

    Zeros   0.818   (did not play, 0 points)
    Blanks  1.291   (played, <= 2 points)
    Tickers 1.517   (3-4 points)
    Haulers 5.142   (>= 5 points)

PASS = our RMSE within ~10% of the paper on every bucket.
This is a correctness check of the reimplementation, NOT a tuning pass.

What is reused from production (the code actually under test):
  - artifact loading, 228-feature scaler order, NaN->0-before-scaling,
    per-position slicing, 50-model ensemble median, inverse transform
    (OpenFPLModel._load_artifacts / _predict_position)
  - windowed-mean construction (model._windowed), the FPL/Understat
    metric maps, and the relevant-points approximation

Explicitly BYPASSED (post-paper additions of ours, not under test):
  - GW1 cold-start ep_next blend (we call _predict_position directly;
    project_all, which applies the blend, is never invoked)
  - expected-minutes scaling (minutes.py)
  - there is NO DefCon logic anywhere in this codebase to bypass; noted
    because 2025-26 defensive-contribution points do NOT exist in the
    2024-25 target data either

Known input-fidelity limitations (disclosed, not silently patched):
  - `status player availability` is set to 1.0 for everyone: vaastav has
    no point-in-time injury flags. Depresses Zeros accuracy vs the paper.
  - `player relevant fpl points` uses our documented approximation
    (total - appearance - bonus); the paper's exact definition is not in
    the OpenFPL repo.
  - windows use within-2024-25 rows only; at GW32+ the {1,3,5,10}
    windows are saturated, only the 38-window is shorter than a full 38.
  - AM (manager) rows are excluded; we do not run the AM models.

Run:  uv run python tests/backtest_openfpl.py
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rivalr.fetch import FPLClient
from rivalr.model import (
    FPL_PLAYER_METRICS,
    US_PLAYER_METRICS,
    US_TEAM_METRICS,
    OpenFPLModel,
    _appearance_points,
    _windowed,
)
from rivalr.understat import FPL_TO_UNDERSTAT_TEAM, Understat

log = logging.getLogger("rivalr.backtest")

SEASON = "2024-25"
US_SEASON = 2024
EVAL_GWS = range(32, 39)
VAASTAV = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    f"master/data/{SEASON}"
)
CACHE = Path("data/cache/backtest")

PAPER = {"Zeros": 0.818, "Blanks": 1.291, "Tickers": 1.517, "Haulers": 5.142}
TOLERANCE = 0.10

POS_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def bucket_of(minutes: int, points: int) -> str:
    """Paper bucket definitions (NOT our ledger's)."""
    if minutes == 0:
        return "Zeros"
    if points <= 2:
        return "Blanks"
    if points <= 4:
        return "Tickers"
    return "Haulers"


def fetch_csv(name: str) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name.replace("/", "_")
    if not path.exists():
        url = f"{VAASTAV}/{name}"
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, path)
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_features_row(
    fpl_hist: list[dict],
    us_hist: list[dict],
    team_hist: list[dict],
    opp_hist: list[dict],
) -> dict[str, float]:
    feats: dict[str, float] = {}
    for base, key in FPL_PLAYER_METRICS.items():
        series = [float(h[key]) for h in fpl_hist]
        for w, val in _windowed(series).items():
            feats[f"{base} {w}"] = val
    relevant = [
        float(h["total_points"])
        - _appearance_points(int(h["minutes"]))
        - float(h["bonus"])
        for h in fpl_hist
    ]
    for w, val in _windowed(relevant).items():
        feats[f"player relevant fpl points {w}"] = val

    for base, key in US_PLAYER_METRICS.items():
        series = [float(m.get(key) or 0.0) for m in us_hist]
        for w, val in _windowed(series).items():
            feats[f"{base} {w}"] = val

    for scope, hist in (("team", team_hist), ("opponent", opp_hist)):
        for base, extract in US_TEAM_METRICS.items():
            series = [extract(m) for m in hist]
            for w, val in _windowed(series).items():
                feats[f"{scope} {base} {w}"] = val

    # vaastav has no point-in-time injury flags: everyone available.
    feats["status player availability"] = 1.0
    return feats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    print("=" * 66)
    print("OpenFPL reimplementation backtest - 2024-25 GW32-38 vs paper")
    print("cold-start blend: BYPASSED (direct _predict_position calls)")
    print("minutes scaling:  BYPASSED (raw model output)")
    print("DefCon logic:     N/A (none exists in this codebase)")
    print("=" * 66)

    merged = fetch_csv("gws/merged_gw.csv")
    players_raw = fetch_csv("players_raw.csv")
    teams_csv = fetch_csv("teams.csv")

    team_name = {int(t["id"]): t["name"] for t in teams_csv}
    us_title = lambda fpl_name: FPL_TO_UNDERSTAT_TEAM.get(fpl_name, fpl_name)

    understat = Understat(season=US_SEASON, cache_dir="data/cache")
    league = understat.league_data(US_SEASON)
    raw_teams = league["teams"]
    team_hist_all: dict[str, list[dict]] = {
        t["title"]: sorted(t.get("history", []), key=lambda m: m["date"])
        for t in (raw_teams.values() if isinstance(raw_teams, dict) else raw_teams)
    }
    missing_teams = {
        us_title(n) for n in team_name.values() if us_title(n) not in team_hist_all
    }
    if missing_teams:
        print(f"FATAL: unmapped Understat teams: {missing_teams}")
        return 2

    # Understat player mapping via the production matcher, fed with the
    # season's players_raw as the element list.
    elements = [
        {
            "id": int(p["id"]),
            "first_name": p["first_name"],
            "second_name": p["second_name"],
            "web_name": p["web_name"],
            "team": int(p["team"]),
        }
        for p in players_raw
    ]
    teams_arg = [{"id": tid, "name": name} for tid, name in team_name.items()]
    us_map = understat.map_fpl_players(elements, teams_arg)
    print(f"understat player mapping: {len(us_map)}/{len(elements)} matched")

    # Per-player FPL fixture rows, kickoff order (includes 0-minute rows,
    # same as element-summary history in production).
    fpl_rows: dict[int, list[dict]] = defaultdict(list)
    for r in merged:
        fpl_rows[int(r["element"])].append(r)
    for rows in fpl_rows.values():
        rows.sort(key=lambda r: r["kickoff_time"])

    us_matches_cache: dict[int, list[dict]] = {}

    def us_matches(element: int) -> list[dict]:
        if element not in us_matches_cache:
            uid = us_map.get(element)
            if uid is None:
                us_matches_cache[element] = []
            else:
                ms = understat.player_matches(uid)
                us_matches_cache[element] = [
                    m for m in ms if m.get("season") == str(US_SEASON)
                ]
        return us_matches_cache[element]

    model = OpenFPLModel(FPLClient())
    model._load_artifacts()
    order = model._feature_order

    # Build one feature row per (player, fixture) in the eval window.
    rows, meta = [], []
    for g in EVAL_GWS:
        gw_rows = [r for r in merged if int(r["GW"]) == g]
        cutoff: date = min(
            date.fromisoformat(r["kickoff_time"][:10]) for r in gw_rows
        )
        for r in gw_rows:
            pos = POS_MAP.get(r["position"])
            if pos is None:  # AM managers etc.
                continue
            el = int(r["element"])
            hist = [h for h in fpl_rows[el] if int(h["GW"]) < g]
            us_h = [m for m in us_matches(el) if m["date"][:10] < cutoff.isoformat()]
            t_title = us_title(r["team"])
            o_title = us_title(team_name[int(r["opponent_team"])])
            t_h = [m for m in team_hist_all[t_title] if m["date"][:10] < cutoff.isoformat()]
            o_h = [m for m in team_hist_all[o_title] if m["date"][:10] < cutoff.isoformat()]
            feats = build_features_row(hist, us_h, t_h, o_h)
            feats["_pos"] = pos
            rows.append(feats)
            meta.append(
                {"element": el, "gw": g, "minutes": int(r["minutes"]),
                 "points": int(r["total_points"])}
            )
        log.info("gw%d: %d prediction rows built", g, sum(1 for m in meta if m["gw"] == g))

    df = pd.DataFrame(rows)
    for col in order:
        if col not in df.columns:
            df[col] = float("nan")  # rank features etc., as in production

    preds = pd.Series(0.0, index=df.index)
    for pos in ["GK", "DEF", "MID", "FWD"]:
        mask = df["_pos"] == pos
        if mask.any():
            preds[mask] = model._predict_position(pos, df[mask])

    # Aggregate per (player, GW): DGW fixtures sum, like production.
    agg: dict[tuple[int, int], dict] = {}
    for m, p in zip(meta, preds):
        key = (m["element"], m["gw"])
        a = agg.setdefault(key, {"pred": 0.0, "points": 0, "minutes": 0})
        a["pred"] += float(p)
        a["points"] += m["points"]
        a["minutes"] += m["minutes"]

    errors: dict[str, list[float]] = defaultdict(list)
    for a in agg.values():
        b = bucket_of(a["minutes"], a["points"])
        errors[b].append(a["pred"] - a["points"])

    print()
    print(f"{'Bucket':<9}{'n':>6}{'ours':>8}{'paper':>8}{'dev':>9}")
    all_ok = True
    for name in ["Zeros", "Blanks", "Tickers", "Haulers"]:
        errs = errors[name]
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan")
        dev = (rmse - PAPER[name]) / PAPER[name]
        flag = "" if abs(dev) <= TOLERANCE else "  <-- outside tolerance"
        if abs(dev) > TOLERANCE:
            all_ok = False
        print(f"{name:<9}{len(errs):>6}{rmse:>8.3f}{PAPER[name]:>8.3f}{dev:>8.1%}{flag}")

    print()
    print("RESULT:", "PASS (all buckets within ~10% of the paper)" if all_ok
          else "FAIL - reimplementation deviates; diagnose before changing code")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
