"""Fixture layer: pre-build check + calibration on 2025-26.

Phase 1 (slow, cached): rebuild as-of OpenFPL predictions for every
2025-26 player-GW (same harness rules as tests/backtest_openfpl.py) and
store, per row, the opponent's last-6 match xGA/xG/goals-conceded lists
as of that GW's cutoff. Written to data/cache/fixture_calib/rows.csv.

Phase 2 (fast): from the cached rows -
  A. PRE-BUILD CHECK: decile the rows by opponent recent form and print
     mean actual vs mean predicted vs mean residual per decile. If the
     residual slopes with opponent form, the model UNDERWEIGHTS a signal
     it already has (an additive residual layer is then the calibrated
     amplification of that signal); if the residual is flat, the signal
     is already used and the layer must not ship.
  B. CALIBRATION: grid over window W in {4,5,6} and shrinkage m in
     {2,4,6}; per position group (GKDEF / MIDFWD) OLS of residual on
     [opp_xGA_form, opp_xG_form, home], 5-fold CV by contiguous GW
     blocks. Gates (fixed in docs/fixture_layer_design.md):
       1. OOF decile calibration monotone-ish, slope in [0.6, 1.4]
       2. OOF RMSE cut >= 0.5%, no paper bucket worse by > 2%
       3. mean per-GW Spearman top-30 not falling

Run:  uv run python scripts/fixture_layer_calib.py [--phase1-only]
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from backtest_openfpl import POS_MAP, build_features_row, bucket_of  # noqa: E402
from rivalr.fetch import FPLClient  # noqa: E402
from rivalr.model import OpenFPLModel  # noqa: E402
from rivalr.understat import FPL_TO_UNDERSTAT_TEAM, Understat  # noqa: E402

log = logging.getLogger("fixture_calib")

SEASON = "2025-26"
PREV_SEASON = "2024-25"
US_SEASON = 2025
EVAL_GWS = range(1, 39)
VAASTAV = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
           "master/data/{season}")
CACHE = Path("data/cache/fixture_calib")
ROWS_CSV = CACHE / "rows.csv"
CAP = 1.5
FOLDS = 5


def fetch_csv(name: str, season: str) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{season}_{name.replace('/', '_')}"
    if not path.exists():
        url = f"{VAASTAV.format(season=season)}/{name}"
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, path)
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def phase1() -> None:
    merged = fetch_csv("gws/merged_gw.csv", SEASON)
    players_raw = fetch_csv("players_raw.csv", SEASON)
    teams_csv = fetch_csv("teams.csv", SEASON)
    prev_merged = fetch_csv("gws/merged_gw.csv", PREV_SEASON)
    prev_players = fetch_csv("players_raw.csv", PREV_SEASON)

    team_name = {int(t["id"]): t["name"] for t in teams_csv}
    us_title = lambda n: FPL_TO_UNDERSTAT_TEAM.get(n, n)

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
        raise SystemExit(f"FATAL: unmapped understat teams: {missing}")

    elements = [{"id": int(p["id"]), "first_name": p["first_name"],
                 "second_name": p["second_name"], "web_name": p["web_name"],
                 "team": int(p["team"])} for p in players_raw]
    us_map = understat.map_fpl_players(
        elements, [{"id": k, "name": v} for k, v in team_name.items()])
    log.info("understat player mapping: %d/%d", len(us_map), len(elements))

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

    out_rows = []
    rows, meta = [], []
    for g in EVAL_GWS:
        gw_rows = [r for r in merged if int(r["GW"]) == g]
        if not gw_rows:
            continue
        cutoff = min(r["kickoff_time"][:10] for r in gw_rows)
        for r in gw_rows:
            pos = POS_MAP.get(r["position"])
            if pos is None:
                continue
            el = int(r["element"])
            slots = [h for h in fixture_slots(el)
                     if h["kickoff_time"][:10] < cutoff]
            opp_title = us_title(team_name[int(r["opponent_team"])])
            t_h = [m for m in team_hist_all[us_title(r["team"])]
                   if m["date"][:10] < cutoff]
            o_h = [m for m in team_hist_all[opp_title]
                   if m["date"][:10] < cutoff]
            feats = build_features_row(
                slots, str(r["was_home"]) == "True", us_by_date(el), t_h, o_h)
            feats["_pos"] = pos
            rows.append(feats)
            # opponent's last-6 matches as-of cutoff, newest last
            tail = o_h[-6:]
            meta.append({
                "element": el, "gw": g, "pos": pos,
                "minutes": int(r["minutes"]), "points": int(r["total_points"]),
                "was_home": str(r["was_home"]) == "True",
                "opp": opp_title,
                "opp_xga6": ";".join(f"{float(m['xGA']):.3f}" for m in tail),
                "opp_xg6": ";".join(f"{float(m['xG']):.3f}" for m in tail),
                "opp_conc6": ";".join(str(int(m["missed"])) for m in tail),
            })
        log.info("gw%d: %d rows", g, len(gw_rows))

    df = pd.DataFrame(rows)
    for col in order:
        if col not in df.columns:
            df[col] = float("nan")
    preds = pd.Series(0.0, index=df.index)
    for pos in ["GK", "DEF", "MID", "FWD"]:
        mask = df["_pos"] == pos
        if mask.any():
            preds[mask] = model._predict_position(pos, df[mask])

    # DGW rows stay separate: the layer is per-fixture.
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()) + ["pred"])
        w.writeheader()
        for mrow, p in zip(meta, preds):
            w.writerow({**mrow, "pred": round(float(p), 4)})
    log.info("phase 1 done: %s (%d rows)", ROWS_CSV, len(meta))


# ---------------------------------------------------------------------------


def load_rows() -> list[dict]:
    with ROWS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gw"] = int(r["gw"])
        r["minutes"] = int(r["minutes"])
        r["points"] = int(r["points"])
        r["pred"] = float(r["pred"])
        r["was_home"] = r["was_home"] == "True"
        for k in ("opp_xga6", "opp_xg6"):
            r[k] = [float(x) for x in r[k].split(";")] if r[k] else []
        r["opp_conc6"] = ([int(x) for x in r["opp_conc6"].split(";")]
                          if r["opp_conc6"] else [])
    return rows


def form_feats(r: dict, W: int, m: int, mu_xga: float, mu_xg: float):
    xga, xg = r["opp_xga6"][-W:], r["opp_xg6"][-W:]
    n = len(xga)
    a = ((sum(xga) + m * mu_xga) / (n + m)) - mu_xga if n or m else 0.0
    d = ((sum(xg) + m * mu_xg) / (n + m)) - mu_xg if n or m else 0.0
    return a, d, (1.0 if r["was_home"] else 0.0)


def ols3(X: list[list[float]], y: list[float]) -> list[float]:
    """OLS with intercept via normal equations (4x4)."""
    import numpy as np
    A = np.array([[1.0] + x for x in X])
    b = np.array(y)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef.tolist()  # [intercept, a, d, home]


GROUP = {"GK": "GKDEF", "DEF": "GKDEF", "MID": "MIDFWD", "FWD": "MIDFWD"}


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s %(levelname)s %(message)s")
    if not ROWS_CSV.exists():
        phase1()
    if "--phase1-only" in sys.argv:
        return 0
    rows = load_rows()
    # league means per match (xGA == xG league-wide; keep both anyway)
    all_xga = [v for r in rows for v in r["opp_xga6"]]
    all_xg = [v for r in rows for v in r["opp_xg6"]]
    mu_xga = sum(all_xga) / len(all_xga)
    mu_xg = sum(all_xg) / len(all_xg)
    print(f"rows: {len(rows)}  league mean xGA/match {mu_xga:.3f}  "
          f"xG/match {mu_xg:.3f}")

    # ---- A. PRE-BUILD CHECK -------------------------------------------
    print("\n== PRE-BUILD CHECK: does the model already use opponent form? ==")
    for grp, key_mu, label in (("MIDFWD", (mu_xga, "opp_xga6"),
                                "opponent last-5 xGA (attacking channel)"),
                               ("GKDEF", (mu_xg, "opp_xg6"),
                                "opponent last-5 xG (clean-sheet channel)")):
        mu, key = key_mu
        sub = [r for r in rows if GROUP[r["pos"]] == grp and r["minutes"] >= 60
               and len(r[key]) >= 3]
        sub.sort(key=lambda r: sum(r[key][-5:]) / len(r[key][-5:]))
        k = len(sub) // 5
        print(f"\n{grp} vs {label} ({len(sub)} rows, quintiles):")
        print(f"  {'quintile':<10}{'form':>7}{'actual':>8}{'pred':>8}{'resid':>8}")
        for q in range(5):
            band = sub[q * k:(q + 1) * k] if q < 4 else sub[4 * k:]
            fmean = sum(sum(r[key][-5:]) / len(r[key][-5:]) for r in band) / len(band)
            am = sum(r["points"] for r in band) / len(band)
            pm = sum(r["pred"] for r in band) / len(band)
            print(f"  Q{q + 1:<9}{fmean:>7.2f}{am:>8.2f}{pm:>8.2f}{am - pm:>+8.2f}")

    # venue check
    for grp in ("MIDFWD", "GKDEF"):
        sub = [r for r in rows if GROUP[r["pos"]] == grp and r["minutes"] >= 60]
        for hv, lab in ((True, "home"), (False, "away")):
            band = [r for r in sub if r["was_home"] == hv]
            am = sum(r["points"] for r in band) / len(band)
            pm = sum(r["pred"] for r in band) / len(band)
            print(f"{grp} {lab}: actual {am:.2f} pred {pm:.2f} resid {am - pm:+.2f}")

    # ---- B. GRID + GATES ----------------------------------------------
    gws = sorted({r["gw"] for r in rows})
    fold_of = {g: min(i * FOLDS // len(gws), FOLDS - 1)
               for i, g in enumerate(gws)}

    def run_config(W: int, m: int):
        adj = {}  # id -> oof adjustment
        for fold in range(FOLDS):
            train = [r for r in rows if fold_of[r["gw"]] != fold
                     and r["minutes"] >= 60]
            test = [r for r in rows if fold_of[r["gw"]] == fold]
            coefs = {}
            for grp in ("GKDEF", "MIDFWD"):
                tg = [r for r in train if GROUP[r["pos"]] == grp]
                X = [list(form_feats(r, W, m, mu_xga, mu_xg)) for r in tg]
                y = [r["points"] - r["pred"] for r in tg]
                coefs[grp] = ols3(X, y)
            for r in test:
                c = coefs[GROUP[r["pos"]]]
                a, d, h = form_feats(r, W, m, mu_xga, mu_xg)
                v = c[1] * a + c[2] * d + c[3] * (h - 0.5)  # centre venue
                # intercept deliberately NOT applied (global bias is the
                # blend's business, not the fixture layer's)
                adj[id(r)] = max(-CAP, min(CAP, v))
        return adj

    def evaluate(adj: dict) -> dict:
        base_se = adj_se = 0.0
        buckets_base = defaultdict(list)
        buckets_adj = defaultdict(list)
        sp_base, sp_adj = [], []
        for g in gws:
            sub = [r for r in rows if r["gw"] == g]
            for r in sub:
                e0 = r["pred"] - r["points"]
                e1 = r["pred"] + adj.get(id(r), 0.0) - r["points"]
                base_se += e0 * e0
                adj_se += e1 * e1
                b = bucket_of(r["minutes"], r["points"])
                buckets_base[b].append(e0)
                buckets_adj[b].append(e1)
            top0 = sorted(sub, key=lambda r: -r["pred"])[:30]
            top1 = sorted(sub, key=lambda r: -(r["pred"] + adj.get(id(r), 0.0)))[:30]

            def sp(top):
                pr = [t["pred"] + adj.get(id(t), 0.0) for t in top]
                ac = [t["points"] for t in top]
                rk = lambda v: {i: rank for rank, i in enumerate(
                    sorted(range(len(v)), key=lambda j: -v[j]))}
                ra, rb = rk(pr), rk(ac)
                n = len(top)
                d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
                return 1 - 6 * d2 / (n * (n * n - 1))
            sp_base.append(sp(top0))
            sp_adj.append(sp(top1))
        n = len(rows)
        out = {
            "rmse_base": math.sqrt(base_se / n),
            "rmse_adj": math.sqrt(adj_se / n),
            "spearman_base": sum(sp_base) / len(sp_base),
            "spearman_adj": sum(sp_adj) / len(sp_adj),
            "buckets": {},
        }
        for b in ("Zeros", "Blanks", "Tickers", "Haulers"):
            eb, ea = buckets_base[b], buckets_adj[b]
            out["buckets"][b] = (
                math.sqrt(sum(e * e for e in eb) / len(eb)),
                math.sqrt(sum(e * e for e in ea) / len(ea)),
            )
        return out

    print("\n== GRID (OOF, 5 folds by GW blocks) ==")
    best = None
    for W in (4, 5, 6):
        for m in (2, 4, 6):
            adj = run_config(W, m)
            ev = evaluate(adj)
            cut = 1 - ev["rmse_adj"] / ev["rmse_base"]
            worst_bucket = max(
                (a - b) / b for b, a in ev["buckets"].values())
            print(f"W={W} m={m}: rmse {ev['rmse_base']:.4f} -> "
                  f"{ev['rmse_adj']:.4f} (cut {cut:+.2%})  "
                  f"spearman {ev['spearman_base']:.3f} -> "
                  f"{ev['spearman_adj']:.3f}  worst-bucket {worst_bucket:+.2%}")
            score = cut
            if best is None or score > best[0]:
                best = (score, W, m, ev, adj)

    score, W, m, ev, adj = best
    print(f"\nBEST: W={W} m={m}")

    # decile calibration on OOF adjustments
    scored = [(adj.get(id(r), 0.0), r["points"] - r["pred"]) for r in rows
              if r["minutes"] >= 60]
    scored.sort(key=lambda t: t[0])
    k = len(scored) // 10
    print("decile calibration (predicted adj vs actual residual, mins>=60):")
    xs, ys = [], []
    for q in range(10):
        band = scored[q * k:(q + 1) * k] if q < 9 else scored[9 * k:]
        pa = sum(t[0] for t in band) / len(band)
        ar = sum(t[1] for t in band) / len(band)
        xs.append(pa)
        ys.append(ar)
        print(f"  D{q + 1:<3} pred_adj {pa:+.3f}  actual_resid {ar:+.3f}")
    mx = sum(xs) / 10
    my = sum(ys) / 10
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / max(sum((x - mx) ** 2 for x in xs), 1e-9))
    print(f"calibration slope: {slope:.2f} (gate: 0.6-1.4)")

    # final full-data coefficients for the production layer
    print("\nfull-data coefficients [intercept, xGA_form, xG_form, home]:")
    for grp in ("GKDEF", "MIDFWD"):
        tg = [r for r in rows if GROUP[r["pos"]] == grp and r["minutes"] >= 60]
        X = [list(form_feats(r, W, m, mu_xga, mu_xg)) for r in tg]
        y = [r["points"] - r["pred"] for r in tg]
        c = ols3(X, y)
        print(f"  {grp}: {[round(v, 4) for v in c]}  (n={len(tg)})")
    print(f"league means: mu_xga={mu_xga:.4f} mu_xg={mu_xg:.4f}")

    cut = 1 - ev["rmse_adj"] / ev["rmse_base"]
    worst = max((a - b) / b for b, a in ev["buckets"].values())
    gates = {
        "rmse_cut>=0.5%": cut >= 0.005,
        "no_bucket_worse_2%": worst <= 0.02,
        "spearman_not_falling": ev["spearman_adj"] >= ev["spearman_base"] - 1e-9,
        "slope_0.6-1.4": 0.6 <= slope <= 1.4,
    }
    print("\nGATES:", gates)
    print("RESULT:", "PASS" if all(gates.values()) else "FAIL")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
