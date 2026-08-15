"""DefCon correction layer: expected defensive-contribution points.

The OpenFPL models were trained on 2020-21..2023-24 and have never seen
defensive contribution points (introduced 2025-26, unchanged 2026-27),
so they systematically underrate defensive players. This layer models
that missing scoring channel and adds it as a SEPARATE, ADDITIVE column
- the OpenFPL base projection is never overwritten, and the ledger logs
base / defcon / final separately so each layer can be scored on its own.

Rules (2025-26 and 2026-27):
    DEF        10+ CBIT  (clearances + blocks + interceptions + tackles)
    MID/FWD    12+ CBIRT (CBIT + ball recoveries)
    GK         not eligible
    capped at 2 points per match regardless of margin

Model: per-group (DEF vs MID/FWD) logistic regression fit on 2025-26
player-matches with 60+ minutes. Features per player-fixture:
    rate90      per-90 relevant-action rate (expanding prior)
    mins_share  average minutes when playing / 90
    opp_xg5     opponent xG, last-5 rolling mean (defensive workload)
    opp_deep5   opponent deep completions, last-5 rolling mean
    home        venue flag
Coefficients are cached in data/cache/defcon_model.json and refit from
the 2025-26 data when absent.

Expected points per fixture = P(threshold | 60+ mins) * 2 * P(plays 60+).

Cold start: 2025-26 full-season per-90 rates are the prior, blended
toward 2026-27 observed rates with weight n_matches/5 (same pattern as
the projection cold-start blend).

Calibration backtest (fit GW2-19, validate GW20-38 of 2025-26):
    python -m rivalr.defcon

BPS adjustment: NOT built. 2026-27 BPS was retuned and there is no data
on it yet; see bps_adjustment() stub. Revisit at GW8.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

from .fetch import FPLClient
from .minutes import MinutesEstimate
from .understat import FPL_TO_UNDERSTAT_TEAM, Understat

log = logging.getLogger("rivalr.defcon")

THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}  # GK not eligible
CAP_POINTS = 2
GROUP = {"DEF": "DEF", "MID": "MIDFWD", "FWD": "MIDFWD"}
# ptail = P(Poisson(rate90 * mins_share) >= threshold): the threshold
# crossing is steeply nonlinear in the rate; giving the logistic this
# well-scaled tail probability fixes the calibration compression seen
# with raw rates alone.
FEATURES = ["ptail", "rate90", "mins_share", "opp_xg5", "opp_deep5", "home"]
GROUP_THRESHOLD = {"DEF": 10, "MIDFWD": 12}


def poisson_tail(mu: float, threshold: int) -> float:
    """P(X >= threshold) for X ~ Poisson(mu)."""
    if mu <= 0:
        return 0.0
    term = math.exp(-mu)
    cdf = term
    for k in range(1, threshold):
        term *= mu / k
        cdf += term
    return max(0.0, 1.0 - cdf)
COLD_START_MATCHES = 5   # same pattern as model.py's projection blend
MIN_PRIOR_MATCHES = 3    # played matches needed before trusting a rate
ROLL = 5                 # opponent rolling window

PRIOR_SEASON = "2025-26"
PRIOR_US_SEASON = 2025

MODEL_CACHE = "defcon_model.json"


def relevant_count(row: dict, pos: str) -> float:
    """CBIT for defenders; CBIRT (adds recoveries) for mids/forwards."""
    cbit = float(row["clearances_blocks_interceptions"]) + float(row["tackles"])
    if pos == "DEF":
        return cbit
    return cbit + float(row["recoveries"])


def defcon_points(count: float, pos: str) -> int:
    """Awarded points for a single match (capped, threshold rule)."""
    if pos not in THRESHOLDS:
        return 0
    return CAP_POINTS if count >= THRESHOLDS[pos] else 0


def p60_from_minutes(est: MinutesEstimate) -> float:
    """v1 approximation of P(plays 60+) from the minutes estimate.
    TODO: replace with an explicit 60+ rate from match history."""
    return min(1.0, max(0.0, (est.expected_minutes - 25.0) / 50.0))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class DefConModel:
    """Logistic P(hits threshold | plays 60+) per group, plus the
    production correction pipeline."""

    def __init__(self, client: FPLClient) -> None:
        self.client = client
        self.cache_dir = Path(client.cache_dir)
        self._params: dict | None = None
        self._prior_rates: dict[str, dict] | None = None  # player code -> stats

    # -- 2025-26 prior data ------------------------------------------------

    def _prior_csv(self, name: str) -> list[dict]:
        path = self.cache_dir / f"vaastav_{PRIOR_SEASON}_{name.replace('/', '_')}"
        if not path.exists():
            import urllib.request
            url = ("https://raw.githubusercontent.com/vaastav/"
                   f"Fantasy-Premier-League/master/data/{PRIOR_SEASON}/{name}")
            log.info("downloading %s", url)
            urllib.request.urlretrieve(url, path)
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def prior_player_rows(self) -> dict[int, list[dict]]:
        rows = defaultdict(list)
        for r in self._prior_csv("gws/merged_gw.csv"):
            rows[int(r["element"])].append(r)
        for v in rows.values():
            v.sort(key=lambda r: r["kickoff_time"])
        return rows

    def prior_rates_by_code(self) -> dict[str, dict]:
        """{player_code: {pos, rate90, mins_avg, n}} from full 2025-26."""
        if self._prior_rates is not None:
            return self._prior_rates
        id_to_code = {
            int(p["id"]): p["code"] for p in self._prior_csv("players_raw.csv")
        }
        out: dict[str, dict] = {}
        for el, rows in self.prior_player_rows().items():
            played = [r for r in rows if int(r["minutes"]) > 0]
            if not played:
                continue
            pos = played[-1]["position"]
            if pos not in THRESHOLDS:
                continue
            mins = sum(int(r["minutes"]) for r in played)
            count = sum(relevant_count(r, pos) for r in played)
            code = id_to_code.get(el)
            if code and mins > 0:
                out[code] = {
                    "pos": pos,
                    "rate90": 90.0 * count / mins,
                    "mins_avg": mins / len(played),
                    "n": len(played),
                }
        self._prior_rates = out
        return out

    def group_fallback_rates(self) -> dict[str, float]:
        """Median per-90 rate per group, for players with no history."""
        by_group = defaultdict(list)
        for stats in self.prior_rates_by_code().values():
            by_group[GROUP[stats["pos"]]].append(stats["rate90"])
        return {
            g: sorted(v)[len(v) // 2] if v else 5.0 for g, v in by_group.items()
        }

    # -- opponent features (understat) -------------------------------------

    @staticmethod
    def _opp_rolling(history: list[dict], before_date: str) -> tuple[float, float]:
        past = [m for m in history if m["date"][:10] < before_date]
        tail = past[-ROLL:]
        if not tail:
            return 1.4, 8.0  # league-ish averages
        xg = sum(float(m["xG"]) for m in tail) / len(tail)
        deep = sum(float(m["deep"]) for m in tail) / len(tail)
        return xg, deep

    # -- fitting -----------------------------------------------------------

    def _build_samples(
        self, gw_lo: int, gw_hi: int, team_hist: dict[str, list[dict]],
        team_name_by_id: dict[int, str],
    ) -> dict[str, tuple[list[list[float]], list[int]]]:
        """Feature/outcome samples from 2025-26, GWs [gw_lo, gw_hi].
        Features use only data available BEFORE each match (expanding)."""
        fallback = self.group_fallback_rates()
        samples: dict[str, tuple[list, list]] = {
            "DEF": ([], []), "MIDFWD": ([], []),
        }
        for el, rows in self.prior_player_rows().items():
            pos = rows[-1]["position"]
            if pos not in THRESHOLDS:
                continue
            group = GROUP[pos]
            hist_counts: list[float] = []   # per-90 rates of prior played
            hist_mins: list[int] = []
            for r in rows:
                mins = int(r["minutes"])
                gw = int(r["GW"])
                if mins >= 60 and gw_lo <= gw <= gw_hi:
                    if len(hist_counts) >= MIN_PRIOR_MATCHES:
                        rate90 = sum(hist_counts) / len(hist_counts)
                    else:
                        rate90 = fallback[group]
                    mins_share = (
                        sum(hist_mins) / len(hist_mins) / 90.0 if hist_mins else 0.8
                    )
                    opp = team_name_by_id.get(int(r["opponent_team"]), "")
                    title = FPL_TO_UNDERSTAT_TEAM.get(opp, opp)
                    xg5, deep5 = self._opp_rolling(
                        team_hist.get(title, []), r["kickoff_time"][:10]
                    )
                    ptail = poisson_tail(
                        rate90 * mins_share, GROUP_THRESHOLD[group]
                    )
                    x = [ptail, rate90, mins_share, xg5, deep5,
                         1.0 if str(r["was_home"]) == "True" else 0.0]
                    y = 1 if relevant_count(r, pos) >= THRESHOLDS[pos] else 0
                    samples[group][0].append(x)
                    samples[group][1].append(y)
                if mins > 0:
                    hist_counts.append(90.0 * relevant_count(r, pos) / mins)
                    hist_mins.append(mins)
        return samples

    def _understat_2025(self) -> tuple[dict[str, list[dict]], dict[int, str]]:
        us = Understat(season=PRIOR_US_SEASON, cache_dir=self.cache_dir)
        raw = us.league_data(PRIOR_US_SEASON)["teams"]
        team_hist = {
            t["title"]: sorted(t.get("history", []), key=lambda m: m["date"])
            for t in (raw.values() if isinstance(raw, dict) else raw)
        }
        teams_csv = self._prior_csv("teams.csv")
        team_name_by_id = {int(t["id"]): t["name"] for t in teams_csv}
        return team_hist, team_name_by_id

    def fit(self, gw_lo: int = 2, gw_hi: int = 38) -> dict:
        """Fit both group models on 2025-26 and cache the coefficients."""
        from sklearn.linear_model import LogisticRegression

        team_hist, team_name_by_id = self._understat_2025()
        samples = self._build_samples(gw_lo, gw_hi, team_hist, team_name_by_id)
        params: dict = {"fitted_on": f"{PRIOR_SEASON} GW{gw_lo}-{gw_hi}",
                        "features": FEATURES, "groups": {}}
        for group, (X, y) in samples.items():
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            params["groups"][group] = {
                "coef": [round(c, 6) for c in clf.coef_[0]],
                "intercept": round(float(clf.intercept_[0]), 6),
                "n": len(y),
                "base_rate": round(sum(y) / len(y), 4),
            }
            log.info("defcon fit %s: n=%d base_rate=%.3f", group, len(y),
                     sum(y) / len(y))
        (self.cache_dir / MODEL_CACHE).write_text(
            json.dumps(params, indent=2), encoding="utf-8"
        )
        self._params = params
        return params

    def params(self) -> dict:
        if self._params is None:
            cache = self.cache_dir / MODEL_CACHE
            if cache.exists():
                self._params = json.loads(cache.read_text(encoding="utf-8"))
            else:
                log.info("no cached defcon model - fitting on %s", PRIOR_SEASON)
                self.fit()
        return self._params

    def p_threshold(self, group: str, x: list[float]) -> float:
        g = self.params()["groups"][group]
        z = g["intercept"] + sum(c * v for c, v in zip(g["coef"], x))
        return _sigmoid(z)

    # -- production corrections --------------------------------------------

    def blended_rate(self, el: dict, pos: str) -> tuple[float, float]:
        """(rate90, mins_avg) blending the 2025-26 prior toward observed
        2026-27 data with weight n/COLD_START_MATCHES."""
        prior = self.prior_rates_by_code().get(str(el.get("code")))
        fallback = self.group_fallback_rates()[GROUP[pos]]
        history = self.client.element_summary(el["id"]).get("history", [])
        played = [h for h in history if h["minutes"] > 0]
        cur_rate = cur_mins = None
        if played:
            mins = sum(h["minutes"] for h in played)
            count = sum(relevant_count(h, pos) for h in played)
            cur_rate = 90.0 * count / mins
            cur_mins = mins / len(played)
        w = min(len(played) / COLD_START_MATCHES, 1.0)

        prior_rate = prior["rate90"] if prior else fallback
        prior_mins = prior["mins_avg"] if prior else 70.0
        rate = w * (cur_rate or 0.0) + (1 - w) * prior_rate
        mins_avg = w * (cur_mins or 0.0) + (1 - w) * prior_mins
        return rate, mins_avg

    def corrections(
        self,
        pool: list[int],
        est: dict[int, MinutesEstimate],
        horizon: int = 5,
        team_hist: dict[str, list[dict]] | None = None,
    ) -> dict[int, list[float]]:
        """Expected DefCon points per player per GW over the horizon.
        Purely additive - callers add this to the base projection."""
        bootstrap = self.client.bootstrap()
        elements = {e["id"]: e for e in bootstrap["elements"]}
        team_name = {t["id"]: t["name"] for t in bootstrap["teams"]}
        pos_of = {1: None, 2: "DEF", 3: "MID", 4: "FWD"}

        if team_hist is None:
            year = int(bootstrap["events"][0]["deadline_time"][:4])
            us = Understat(season=year, cache_dir=self.cache_dir)
            try:
                team_hist = us.teams_data()
            except Exception:
                log.error("defcon: understat unavailable, opponent features "
                          "fall back to league averages")
                team_hist = {}

        next_gw = self.client.next_gw()
        gws = list(range(next_gw, min(39, next_gw + horizon)))
        fixtures: dict[int, dict[int, list[dict]]] = {}
        for f in self.client.fixtures():
            if f.get("event") in gws and not f.get("finished"):
                fixtures.setdefault(f["team_h"], {}).setdefault(
                    f["event"], []).append({"opp": f["team_a"], "home": True})
                fixtures.setdefault(f["team_a"], {}).setdefault(
                    f["event"], []).append({"opp": f["team_h"], "home": False})

        today = "9999-12-31"  # rolling means over everything played so far
        out: dict[int, list[float]] = {}
        for pid in pool:
            el = elements.get(pid)
            pos = pos_of.get(el["element_type"]) if el else None
            if pos is None:
                out[pid] = [0.0] * len(gws)
                continue
            e = est.get(pid)
            p60 = p60_from_minutes(e) if e else 0.5
            rate, mins_avg = self.blended_rate(el, pos)
            xs = []
            for gw in gws:
                total = 0.0
                for fx in fixtures.get(el["team"], {}).get(gw, []):
                    opp_name = team_name.get(fx["opp"], "")
                    title = FPL_TO_UNDERSTAT_TEAM.get(opp_name, opp_name)
                    xg5, deep5 = self._opp_rolling(
                        team_hist.get(title, []), today
                    )
                    ptail = poisson_tail(
                        rate * mins_avg / 90.0, GROUP_THRESHOLD[GROUP[pos]]
                    )
                    x = [ptail, rate, mins_avg / 90.0, xg5, deep5,
                         1.0 if fx["home"] else 0.0]
                    p = self.p_threshold(GROUP[pos], x)
                    total += p * CAP_POINTS * p60
                xs.append(round(total, 3))
            out[pid] = xs
        return out


def bps_adjustment(player_id: int, horizon: int) -> list[float]:
    """TODO (GW8): 2026-27 retuned the BPS table and no data exists yet
    on how it shifts bonus points. Deliberately returns zeros until at
    least ~8 GWs of observed bonus data are available to fit against."""
    return [0.0] * horizon


# -- calibration backtest --------------------------------------------------


def calibrate(client: FPLClient) -> dict:
    """Temporal validation on 2025-26: fit GW2-19, validate GW20-38.
    Reports decile calibration (predicted vs observed hit rate), Brier
    score and AUC per group."""
    dc = DefConModel(client)
    team_hist, team_name_by_id = dc._understat_2025()

    log.info("fitting on GW2-19...")
    dc.fit(2, 19)
    log.info("building validation samples GW20-38...")
    val = dc._build_samples(20, 38, team_hist, team_name_by_id)

    from sklearn.metrics import roc_auc_score

    report = {}
    for group, (X, y) in val.items():
        preds = [dc.p_threshold(group, x) for x in X]
        brier = sum((p - t) ** 2 for p, t in zip(preds, y)) / len(y)
        auc = roc_auc_score(y, preds)
        deciles = []
        pairs = sorted(zip(preds, y))
        n = len(pairs)
        for d in range(10):
            chunk = pairs[d * n // 10:(d + 1) * n // 10]
            if chunk:
                deciles.append({
                    "predicted": round(sum(p for p, _ in chunk) / len(chunk), 3),
                    "observed": round(sum(t for _, t in chunk) / len(chunk), 3),
                    "n": len(chunk),
                })
        report[group] = {
            "n": n, "base_rate": round(sum(y) / n, 3),
            "brier": round(brier, 4), "auc": round(auc, 3),
            "deciles": deciles,
        }
    # refit on the full season for production use
    log.info("refitting on full 2025-26 for production...")
    dc.fit(2, 38)
    return report


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(name)s %(levelname)s %(message)s")
    report = calibrate(FPLClient())
    for group, r in report.items():
        print(f"\n{group}: n={r['n']} base_rate={r['base_rate']} "
              f"brier={r['brier']} auc={r['auc']}")
        print(f"{'decile':>7}{'pred':>8}{'obs':>8}{'n':>7}")
        for i, d in enumerate(r["deciles"], 1):
            print(f"{i:>7}{d['predicted']:>8.3f}{d['observed']:>8.3f}{d['n']:>7}")


if __name__ == "__main__":
    main()
