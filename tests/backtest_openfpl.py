"""Backtest: validate OUR OpenFPL feature pipeline against the paper.

Reference: OpenFPL (arXiv 2508.09992), Table 4 - one-gameweek-ahead RMSE
on 2024-25, evaluation window GW32-GW38:

    Zeros   0.818   (did not play, 0 points)
    Blanks  1.291   (played, <= 2 points)
    Tickers 1.517   (3-4 points)
    Haulers 5.142   (>= 5 points)

Feature-series rules (all verified against vendor/OpenFPL/data/samples.csv,
see docs/backtest_findings.md):
  - a player's history is a timeline of FIXTURE SLOTS: previous season's
    rows (joined across seasons by player code) followed by the current
    season's rows; windows {1,3,5,10,38} take the trailing N slots FIRST,
    then drop previous-season slots the player didn't play, and average
    the remainder (current-season 0-minute rows are kept)
  - `player relevant fpl points` = points in slots at the VENUE of the
    upcoming match (home/away split)
  - Understat player metrics are DATE-ALIGNED onto those fixture slots,
    0.0 where the player has no Understat match that day
  - team/opponent Understat histories also cross the season boundary

Explicitly BYPASSED (post-paper additions of ours, not under test):
  - GW1 cold-start ep_next blend (direct _predict_position calls)
  - expected-minutes scaling
  - no DefCon logic exists in this codebase

Known input-fidelity limitations (disclosed):
  - `status player availability` = 1.0 for everyone (vaastav has no
    point-in-time injury flags). Depresses Zeros/Blanks vs the paper.
  - AM (manager) rows are excluded; the paper's Table 4 includes them.

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
    _windowed,
    aligned_windowed,
    slot_windowed,
    venue_windowed,
)
from rivalr.understat import FPL_TO_UNDERSTAT_TEAM, Understat

log = logging.getLogger("rivalr.backtest")

SEASON = "2024-25"
PREV_SEASON = "2023-24"
US_SEASON = 2024
EVAL_GWS = range(32, 39)
VAASTAV = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}"
)
CACHE = Path("data/cache/backtest")

PAPER = {"Zeros": 0.818, "Blanks": 1.291, "Tickers": 1.517, "Haulers": 5.142}
# Agreed pass criteria (fixed BEFORE the post-fix rerun):
TOLERANCES = {"Zeros": 0.25, "Blanks": 0.25, "Tickers": 0.15, "Haulers": 0.05}
# First-run deviations (pre-fix), for the delta column and the
# "Zeros/Blanks must improve" directional check:
BASELINE_DEV = {"Zeros": 0.157, "Blanks": 0.198, "Tickers": -0.127, "Haulers": 0.030}

POS_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def bucket_of(minutes: int, points: int) -> str:
    """Paper bucket definitions."""
    if minutes == 0:
        return "Zeros"
    if points <= 2:
        return "Blanks"
    if points <= 4:
        return "Tickers"
    return "Haulers"


def fetch_csv(name: str, season: str = SEASON) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{season}_{name.replace('/', '_')}"
    if not path.exists():
        url = f"{VAASTAV.format(season=season)}/{name}"
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, path)
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_features_row(
    slots: list[dict],
    upcoming_home: bool,
    us_by_date: dict[str, dict],
    team_hist: list[dict],
    opp_hist: list[dict],
) -> dict[str, float]:
    """slots: full fixture-slot timeline (prev-season ALL rows tagged
    _prev=True + current-season rows), oldest first. Window semantics live
    in rivalr.model.slot_windowed - the code under test."""
    feats: dict[str, float] = {}
    for base, key in FPL_PLAYER_METRICS.items():
        for w, val in slot_windowed(slots, lambda s, k=key: float(s[k])).items():
            feats[f"{base} {w}"] = val

    for w, val in venue_windowed(slots, upcoming_home).items():
        feats[f"player relevant fpl points {w}"] = val

    for base, key in US_PLAYER_METRICS.items():
        for w, val in aligned_windowed(slots, us_by_date, key).items():
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
    # DIAGNOSTIC ONLY: --oracle-availability replaces the hardcoded 1.0
    # availability with hindsight (0 if the player ended up not playing).
    # This LEAKS the outcome and is NOT a valid accuracy claim - it exists
    # solely to bound how much of the Zeros/Blanks gap is explained by
    # vaastav's missing point-in-time injury flags.
    oracle = "--oracle-availability" in sys.argv
    print("=" * 70)
    print("OpenFPL reimplementation backtest - 2024-25 GW32-38 vs paper")
    print("cold-start blend: BYPASSED (direct _predict_position calls)")
    print("minutes scaling:  BYPASSED (raw model output)")
    print("DefCon logic:     N/A (none exists in this codebase)")
    if oracle:
        print("MODE: ORACLE AVAILABILITY (outcome-leaking diagnostic, "
              "NOT a valid accuracy claim)")
    print("=" * 70)

    merged = fetch_csv("gws/merged_gw.csv")
    players_raw = fetch_csv("players_raw.csv")
    teams_csv = fetch_csv("teams.csv")
    prev_merged = fetch_csv("gws/merged_gw.csv", season=PREV_SEASON)
    prev_players = fetch_csv("players_raw.csv", season=PREV_SEASON)

    team_name = {int(t["id"]): t["name"] for t in teams_csv}
    us_title = lambda fpl_name: FPL_TO_UNDERSTAT_TEAM.get(fpl_name, fpl_name)

    # Cross-season Understat team histories (2023 + 2024), per title.
    understat = Understat(season=US_SEASON, cache_dir="data/cache")
    team_hist_all: dict[str, list[dict]] = defaultdict(list)
    for season in (US_SEASON - 1, US_SEASON):
        raw = understat.league_data(season)["teams"]
        for t in (raw.values() if isinstance(raw, dict) else raw):
            team_hist_all[t["title"]].extend(
                sorted(t.get("history", []), key=lambda m: m["date"])
            )
    missing = {us_title(n) for n in team_name.values()} - set(team_hist_all)
    if missing:
        print(f"FATAL: unmapped Understat teams: {missing}")
        return 2

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
    us_map = understat.map_fpl_players(
        elements, [{"id": k, "name": v} for k, v in team_name.items()]
    )
    print(f"understat player mapping: {len(us_map)}/{len(elements)} matched")

    # Fixture slots: previous-season ALL rows (joined by player code,
    # tagged _prev so slot_windowed can drop unplayed ones positionally)
    # + current-season all rows, kickoff order.
    code_to_prev_el = {p["code"]: int(p["id"]) for p in prev_players}
    el_to_code = {int(p["id"]): p["code"] for p in players_raw}
    prev_rows: dict[int, list[dict]] = defaultdict(list)
    for r in prev_merged:
        r["_prev"] = True
        prev_rows[int(r["element"])].append(r)
    cur_rows: dict[int, list[dict]] = defaultdict(list)
    for r in merged:
        cur_rows[int(r["element"])].append(r)

    def fixture_slots(el: int) -> list[dict]:
        prev_el = code_to_prev_el.get(el_to_code.get(el, ""))
        slots = list(prev_rows.get(prev_el, [])) if prev_el else []
        slots += cur_rows[el]
        slots.sort(key=lambda r: r["kickoff_time"])
        return slots

    us_by_date_cache: dict[int, dict[str, dict]] = {}

    def us_by_date(el: int) -> dict[str, dict]:
        if el not in us_by_date_cache:
            uid = us_map.get(el)
            ms = understat.player_matches(uid) if uid else []
            us_by_date_cache[el] = {m["date"][:10]: m for m in ms}
        return us_by_date_cache[el]

    model = OpenFPLModel(FPLClient())
    model._load_artifacts()
    order = model._feature_order

    rows, meta = [], []
    for g in EVAL_GWS:
        gw_rows = [r for r in merged if int(r["GW"]) == g]
        cutoff: str = min(r["kickoff_time"][:10] for r in gw_rows)
        for r in gw_rows:
            pos = POS_MAP.get(r["position"])
            if pos is None:  # AM managers etc.
                continue
            el = int(r["element"])
            slots = [h for h in fixture_slots(el) if h["kickoff_time"][:10] < cutoff]
            t_h = [m for m in team_hist_all[us_title(r["team"])] if m["date"][:10] < cutoff]
            o_h = [m for m in team_hist_all[us_title(team_name[int(r["opponent_team"])])]
                   if m["date"][:10] < cutoff]
            feats = build_features_row(
                slots, str(r["was_home"]) == "True", us_by_date(el), t_h, o_h
            )
            if oracle:
                feats["status player availability"] = (
                    1.0 if int(r["minutes"]) > 0 else 0.0
                )
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

    agg: dict[tuple[int, int], dict] = {}
    for m, p in zip(meta, preds):
        key = (m["element"], m["gw"])
        a = agg.setdefault(key, {"pred": 0.0, "points": 0, "minutes": 0})
        a["pred"] += float(p)
        a["points"] += m["points"]
        a["minutes"] += m["minutes"]

    errors: dict[str, list[float]] = defaultdict(list)
    for a in agg.values():
        errors[bucket_of(a["minutes"], a["points"])].append(a["pred"] - a["points"])

    print()
    print(f"{'Bucket':<9}{'n':>6}{'ours':>8}{'paper':>8}{'dev':>9}{'was':>9}{'delta':>9}")
    all_ok = True
    for name in ["Zeros", "Blanks", "Tickers", "Haulers"]:
        errs = errors[name]
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan")
        dev = (rmse - PAPER[name]) / PAPER[name]
        was = BASELINE_DEV[name]
        ok = abs(dev) <= TOLERANCES[name]
        if name in ("Zeros", "Blanks"):
            ok = ok and dev < was  # directional: must improve on the baseline
        if not ok:
            all_ok = False
        print(f"{name:<9}{len(errs):>6}{rmse:>8.3f}{PAPER[name]:>8.3f}{dev:>8.1%}"
              f"{was:>8.1%}{dev - was:>+8.1%}{'' if ok else '  <-- outside criteria'}")

    print()
    print("Pass criteria (fixed pre-run): Haulers 5%, Tickers 15%, "
          "Zeros/Blanks 25% AND improved vs baseline")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
