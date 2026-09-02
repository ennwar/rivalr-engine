"""Projections: OpenFPL inference as the baseline model. No retraining.

Pipeline (mirrors vendor/OpenFPL/play.ipynb exactly):
  1. Build the 228-feature vector per (player, future fixture): rolling means
     over the previous {1,3,5,10,38} gameweeks of FPL stats (element-summary)
     and Understat stats (player pages + league team data).
  2. np.nan_to_num -> float32 -> xscaler.transform -> nan_to_num -> float32.
  3. Slice to the per-position feature subset (models/features.save).
  4. Predict with all 50 models per position (5 CV folds x 10 candidates),
     inverse-transform with yscaler (points = y * 33 - 7), take the median.

We run GK/DEF/MID/FWD only. The AM (assistant manager) models and their
league-rank features are skipped: we project players, not managers, so the
12 rank columns are left NaN exactly like OpenFPL's own samples.csv does
for player rows.

Multi-GW horizon: OpenFPL predicts a single GW. For GW n+k we keep the
player/team form aggregates as of *now* and swap in the opponent block and
fixture for that GW - double gameweeks sum both fixtures, blanks are 0.

Stable interface:
    project(player_id, horizon) -> [xPts per gw]
    project_all(client, horizon) -> {player_id: [xPts per gw]}

If OpenFPL or Understat is unavailable the failure is logged at ERROR and
we degrade: Understat-only failure -> those features go NaN (scaled like
OpenFPL's own missing values); OpenFPL-missing -> fall back to the FPL
site's ep_next, flat across the horizon, so the rest of the engine runs.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import FPLClient
from .understat import Understat
from . import vendors

log = logging.getLogger("rivalr.model")

WINDOWS = [1, 3, 5, 10, 38]
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}  # element_type -> OpenFPL dir

# Blanks floor bias (docs/backtest_findings.md, 2026-08-14): the model
# essentially never predicts below ~1.7 for a player who takes the pitch
# (25th-percentile prediction for played players = 1.72), so projections
# near that floor carry no real signal - the model cannot distinguish a
# 1.7 from a blank. We deliberately do NOT de-bias projections: the
# correction was measured on a backtest without availability flags, and
# in-season the live flags + expected-minutes scaling attack the same
# over-prediction (double-correction risk). Revisit at GW8 with the live
# ledger's own evidence. Until then, flag low-margin projections.
BLANKS_FLOOR = 1.7
LOW_CONFIDENCE_MARGIN = 0.5


def confidence_margin(next_gw_xpts: float, minutes_factor: float = 1.0) -> float:
    """Projection minus the expected-minutes-adjusted model floor."""
    return next_gw_xpts - BLANKS_FLOOR * minutes_factor


def is_low_confidence(next_gw_xpts: float, minutes_factor: float = 1.0) -> bool:
    """True when a projection sits within LOW_CONFIDENCE_MARGIN of the
    floor - i.e. the number is indistinguishable from the model's
    default for 'plays but does nothing'."""
    return confidence_margin(next_gw_xpts, minutes_factor) <= LOW_CONFIDENCE_MARGIN

# FPL element-summary history key per feature base name.
FPL_PLAYER_METRICS = {
    "player fpl points": "total_points",
    "player minutes played": "minutes",
    "player influence": "influence",
    "player creativity": "creativity",
    "player threat": "threat",
    "player goals scored": "goals_scored",
    "player penalties missed": "penalties_missed",
    "player assists": "assists",
    "player goals conceded": "goals_conceded",
    "player own goals": "own_goals",
    "player saves": "saves",
    "player penalties saved": "penalties_saved",
    "player yellow cards": "yellow_cards",
    "player red cards": "red_cards",
    "player bps": "bps",
    "player fpl bonus points": "bonus",
}

# Understat per-player match key per feature base name.
US_PLAYER_METRICS = {
    "player shots": "shots",
    "player xg": "xG",
    "player xgchain": "xGChain",
    "player xgbuildup": "xGBuildup",
    "player key passes": "key_passes",
    "player xa": "xA",
}

# Understat team-history extractors per team-scope metric base name.
US_TEAM_METRICS = {
    "goals scored": lambda m: float(m["scored"]),
    "goals conceded": lambda m: float(m["missed"]),
    "xg": lambda m: float(m["xG"]),
    "xga": lambda m: float(m["xGA"]),
    "deep": lambda m: float(m["deep"]),
    "deep allowed": lambda m: float(m["deep_allowed"]),
    "ppda att": lambda m: float(m["ppda"]["att"]),
    "ppda def": lambda m: float(m["ppda"]["def"]),
    "ppda allowed att": lambda m: float(m["ppda_allowed"]["att"]),
    "ppda allowed def": lambda m: float(m["ppda_allowed"]["def"]),
}


def _windowed(values: list[float]) -> dict[int, float]:
    """Mean over the last N observations for each window; NaN when empty."""
    out = {}
    for w in WINDOWS:
        tail = values[-w:]
        out[w] = sum(tail) / len(tail) if tail else math.nan
    return out


# -- fixture-slot windows (semantics verified against OpenFPL samples.csv,
# see docs/backtest_findings.md) ------------------------------------------
#
# A player's history is a timeline of fixture SLOTS: the previous season's
# rows (played or not) followed by the current season's rows. A window
# takes the trailing N slots of the timeline FIRST, then drops
# previous-season slots the player didn't play (they hold a timeline
# position but carry no observation; current-season 0-minute rows are
# real observations and stay), and averages what remains.


def _slot_dropped(slot: dict) -> bool:
    return bool(slot.get("_prev")) and float(slot["minutes"]) == 0


def slot_windowed(slots: list[dict], value_fn) -> dict[int, float]:
    out = {}
    for w in WINDOWS:
        vals = [value_fn(s) for s in slots[-w:] if not _slot_dropped(s)]
        out[w] = sum(vals) / len(vals) if vals else math.nan
    return out


def _slot_home(slot: dict) -> bool:
    return str(slot.get("was_home")) == "True"


def venue_windowed(slots: list[dict], home: bool) -> dict[int, float]:
    """'relevant fpl points': points in slots at the upcoming venue."""
    vslots = [s for s in slots if _slot_home(s) == home]
    return slot_windowed(vslots, lambda s: float(s["total_points"]))


def aligned_windowed(
    slots: list[dict], us_by_date: dict[str, dict], key: str
) -> dict[int, float]:
    """Understat player metrics date-aligned onto the fixture slots,
    0.0 where the player has no Understat match that day."""
    def val(slot: dict) -> float:
        m = us_by_date.get(str(slot["kickoff_time"])[:10])
        return float(m.get(key) or 0.0) if m else 0.0
    return slot_windowed(slots, val)


class OpenFPLModel:
    def __init__(
        self,
        client: FPLClient,
        understat: Understat | None = None,
    ) -> None:
        self.client = client
        bootstrap = client.bootstrap()
        self._elements = {el["id"]: el for el in bootstrap["elements"]}
        self._teams = bootstrap["teams"]
        self._team_name = {t["id"]: t["name"] for t in self._teams}
        season = int(bootstrap["events"][0]["deadline_time"][:4])
        self.understat = understat or Understat(season=season, cache_dir=client.cache_dir)
        self._played_fixture_ids: set[int] | None = None  # lazy, see _current_rows

        self._artifacts_loaded = False
        self._models: dict[str, list] = {}
        self._model_paths: dict[str, list] = {}
        self._xscaler = None
        self._yscaler = None
        self._features: dict[str, list[str]] = {}
        self._feature_order: list[str] = []

        self._us_ready = False
        self._us_team_hist: dict[str, list[dict]] = {}   # understat title -> history
        self._us_player_map: dict[int, str] = {}          # fpl id -> understat id

        self._prev_loaded = False
        self._prev_rows_by_code: dict[str, list[dict]] = {}  # player code -> rows

    # -- previous-season fixture rows (vaastav dataset) --------------------

    PREV_SEASON_URL = (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        "master/data/{season}/{name}"
    )

    def _load_prev_season(self) -> None:
        """Previous season's per-fixture rows, keyed by permanent player
        code (element ids change between seasons). Finished-season data:
        downloaded once, cached forever. Loud failure -> empty (windows
        then truncate at the season boundary, logged)."""
        if self._prev_loaded:
            return
        self._prev_loaded = True
        import csv
        import urllib.request

        bootstrap = self.client.bootstrap()
        year = int(bootstrap["events"][0]["deadline_time"][:4])
        season = f"{year - 1}-{year % 100:02d}"
        cache = Path(self.client.cache_dir)
        try:
            files = {}
            for name in ("gws/merged_gw.csv", "players_raw.csv"):
                path = cache / f"vaastav_{season}_{name.replace('/', '_')}"
                if not path.exists():
                    url = self.PREV_SEASON_URL.format(season=season, name=name)
                    log.info("downloading previous-season data: %s", url)
                    urllib.request.urlretrieve(url, path)
                with path.open(encoding="utf-8") as f:
                    files[name] = list(csv.DictReader(f))
            id_to_code = {
                int(p["id"]): p["code"] for p in files["players_raw.csv"]
            }
            by_code: dict[str, list[dict]] = {}
            for r in files["gws/merged_gw.csv"]:
                code = id_to_code.get(int(r["element"]))
                if code:
                    r["_prev"] = True
                    by_code.setdefault(code, []).append(r)
            for rows in by_code.values():
                rows.sort(key=lambda r: r["kickoff_time"])
            self._prev_rows_by_code = by_code
            log.info("previous season (%s): fixture rows for %d players",
                     season, len(by_code))
        except Exception:
            log.error(
                "PREVIOUS-SEASON DATA UNAVAILABLE (%s) - windows will "
                "truncate at the season boundary", season, exc_info=True,
            )
            self._prev_rows_by_code = {}

    # -- artifact loading --------------------------------------------------

    def _load_artifacts(self) -> None:
        if self._artifacts_loaded:
            return
        import joblib

        root = vendors.require_openfpl()
        models_dir = root / "models"
        self._xscaler = joblib.load(models_dir / "xscaler.save")
        self._yscaler = joblib.load(models_dir / "yscaler.save")
        self._features = joblib.load(models_dir / "features.save")
        self._feature_order = list(self._xscaler.feature_names_in_)
        if len(self._feature_order) != 228:
            raise RuntimeError(
                f"expected 228 scaler features, got {len(self._feature_order)}"
            )
        for pos in ["GK", "DEF", "MID", "FWD"]:
            paths = []
            for cv in range(1, 6):
                cv_dir = models_dir / f"cv{cv}_{pos}"
                for candidate in sorted(p for p in cv_dir.iterdir() if p.is_dir()):
                    files = list(candidate.glob("*.joblib"))
                    if files:
                        paths.append(files[0])
            if len(paths) != 50:
                log.warning("%s: found %d model files (expected 50)", pos, len(paths))
            self._model_paths[pos] = paths
        self._artifacts_loaded = True
        log.info("OpenFPL artifacts indexed (%d model files)",
                 sum(len(v) for v in self._model_paths.values()))

    def _models_for(self, pos: str) -> list:
        """Load a position's 50-model bundle. With RIVALR_LOW_MEM=1 the
        bundle is NOT cached - the caller predicts and releases, keeping
        peak memory to one position's models (matters on small cloud
        containers; the ensemble itself is unchanged)."""
        import os

        import joblib

        if pos in self._models:
            return self._models[pos]
        bundle = [joblib.load(p) for p in self._model_paths[pos]]
        if os.environ.get("RIVALR_LOW_MEM") != "1":
            self._models[pos] = bundle
        return bundle

    # -- understat aggregates ----------------------------------------------

    def _load_understat(self) -> None:
        if self._us_ready:
            return
        try:
            self._us_team_hist = self.understat.teams_data()
            self._us_player_map = self.understat.map_fpl_players(
                list(self._elements.values()), self._teams
            )
        except Exception:
            log.error(
                "UNDERSTAT UNAVAILABLE - proceeding with FPL-only features. "
                "Team/opponent and player xG features will be NaN.",
            )
            self._us_team_hist = {}
            self._us_player_map = {}
        if self._us_team_hist:
            from .understat import FPL_TO_UNDERSTAT_TEAM
            for t in self._teams:
                title = FPL_TO_UNDERSTAT_TEAM.get(t["name"], t["name"])
                if not self._us_team_hist.get(title):
                    log.error(
                        "UNDERSTAT TEAM UNMAPPED: FPL '%s' -> '%s' has zero "
                        "matches - all its team/opponent features will be "
                        "NaN. Add a FPL_TO_UNDERSTAT_TEAM entry.",
                        t["name"], title,
                    )
        self._us_ready = True

    def _team_block(self, fpl_team_id: int, scope: str) -> dict[str, float]:
        from .understat import FPL_TO_UNDERSTAT_TEAM

        name = self._team_name.get(fpl_team_id, "")
        title = FPL_TO_UNDERSTAT_TEAM.get(name, name)
        history = self._us_team_hist.get(title, [])
        feats: dict[str, float] = {}
        for base, extract in US_TEAM_METRICS.items():
            series = [extract(m) for m in history]
            for w, val in _windowed(series).items():
                feats[f"{scope} {base} {w}"] = val
        return feats

    # -- feature building --------------------------------------------------

    def _current_rows(self, pid: int) -> list[dict]:
        """Current-season element-summary rows for FINISHED fixtures only.
        The API pre-creates all-zero rows for unplayed fixtures; feeding
        one into a form window reads 'hasn't kicked off yet' as 'blanked'."""
        if self._played_fixture_ids is None:
            from . import gameweek
            self._played_fixture_ids = gameweek.played_fixture_ids(self.client)
        return [
            h for h in self.client.element_summary(pid).get("history", [])
            if h.get("fixture") in self._played_fixture_ids
        ]

    def _player_slots(self, pid: int) -> list[dict]:
        """Fixture-slot timeline: previous-season rows (vaastav, by player
        code) + current-season element-summary rows, kickoff order."""
        self._load_prev_season()
        el = self._elements[pid]
        prev = self._prev_rows_by_code.get(str(el.get("code")), [])
        return prev + self._current_rows(pid)  # both already kickoff-sorted

    def _player_features(
        self, pid: int
    ) -> tuple[dict[str, float], dict[int, float], dict[int, float]]:
        """(venue-independent features, relevant-home windows,
        relevant-away windows). 'relevant fpl points' depends on the
        upcoming fixture's venue, so both variants are returned and the
        caller picks per fixture."""
        el = self._elements[pid]
        slots = self._player_slots(pid)
        feats: dict[str, float] = {}

        for base, key in FPL_PLAYER_METRICS.items():
            for w, val in slot_windowed(slots, lambda s, k=key: float(s[k])).items():
                feats[f"{base} {w}"] = val

        rel_home = venue_windowed(slots, home=True)
        rel_away = venue_windowed(slots, home=False)

        us_id = self._us_player_map.get(pid)
        matches = self.understat.player_matches(us_id) if us_id else []
        us_by_date = {m["date"][:10]: m for m in matches}
        for base, key in US_PLAYER_METRICS.items():
            for w, val in aligned_windowed(slots, us_by_date, key).items():
                feats[f"{base} {w}"] = val

        status = el.get("status", "a")
        chance = el.get("chance_of_playing_next_round")
        if status == "a":
            avail = 1.0
        elif status in ("i", "s", "n", "u"):
            avail = 0.0 if chance is None else chance / 100.0
        else:
            avail = (chance if chance is not None else 50) / 100.0
        feats["status player availability"] = avail
        return feats, rel_home, rel_away

    def _upcoming_fixtures(self, horizon: int) -> tuple[int, dict[int, dict[int, list[dict]]]]:
        """next_gw, {team_id: {gw: [ {opponent, home} ]}} over the horizon."""
        next_gw = self.client.next_gw()
        gws = range(next_gw, min(39, next_gw + horizon))
        by_team: dict[int, dict[int, list[dict]]] = {}
        for f in self.client.fixtures():
            if f.get("event") not in gws or f.get("finished"):
                continue
            h, a, gw = f["team_h"], f["team_a"], f["event"]
            by_team.setdefault(h, {}).setdefault(gw, []).append(
                {"opponent": a, "home": True}
            )
            by_team.setdefault(a, {}).setdefault(gw, []).append(
                {"opponent": h, "home": False}
            )
        return next_gw, by_team

    # -- inference ---------------------------------------------------------

    def _predict_position(self, pos: str, rows: pd.DataFrame) -> np.ndarray:
        import gc
        import os

        X = rows[self._feature_order].to_numpy()
        X = np.nan_to_num(X).astype("float32")
        X = np.nan_to_num(self._xscaler.transform(X)).astype("float32")
        idx = [self._feature_order.index(f) for f in self._features[pos]]
        X = X[:, idx]
        bundle = self._models_for(pos)
        preds = []
        for model in bundle:
            p = model.predict(X)
            p = self._yscaler.inverse_transform(p.reshape(-1, 1)).reshape(-1)
            preds.append(p)
        result = np.median(np.vstack(preds), axis=0)
        if os.environ.get("RIVALR_LOW_MEM") == "1":
            del bundle, preds
            gc.collect()
        return result

    def default_pool(self) -> list[int]:
        """Players worth projecting: not marked unavailable/left."""
        return [
            el["id"] for el in self._elements.values()
            if el.get("status") not in ("u", "n")
        ]

    def project_all(
        self, horizon: int = 5, pool: list[int] | None = None
    ) -> dict[int, list[float]]:
        self._load_artifacts()
        self._load_understat()
        pool = pool or self.default_pool()
        next_gw, fixtures = self._upcoming_fixtures(horizon)
        gws = list(range(next_gw, min(39, next_gw + horizon)))

        # Opponent blocks are shared across players: precompute per team.
        opp_block = {t["id"]: self._team_block(t["id"], "opponent") for t in self._teams}
        own_block = {t["id"]: self._team_block(t["id"], "team") for t in self._teams}

        rows: list[dict] = []
        meta: list[tuple[int, int, bool]] = []  # (player_id, gw, home)
        season_matches: dict[int, int] = {}
        for pid in pool:
            el = self._elements.get(pid)
            if el is None or el["element_type"] not in POSITIONS:
                continue
            base, rel_home, rel_away = self._player_features(pid)
            season_matches[pid] = len(self._current_rows(pid))
            base.update(own_block[el["team"]])
            team_fixtures = fixtures.get(el["team"], {})
            for gw in gws:
                for fx in team_fixtures.get(gw, []):
                    row = dict(base)
                    rel = rel_home if fx["home"] else rel_away
                    for w, val in rel.items():
                        row[f"player relevant fpl points {w}"] = val
                    row.update(opp_block[fx["opponent"]])
                    row["_pos"] = POSITIONS[el["element_type"]]
                    rows.append(row)
                    meta.append((pid, gw, fx["home"]))

        if not rows:
            return {}
        df = pd.DataFrame(rows)
        for col in self._feature_order:
            if col not in df.columns:
                df[col] = math.nan  # rank features etc.

        results: dict[int, list[float]] = {
            pid: [0.0] * len(gws) for pid in pool if pid in self._elements
        }
        self.last_venue: dict[int, list[float]] = {
            pid: [0.0] * len(gws) for pid in results
        }
        for pos in ["GK", "DEF", "MID", "FWD"]:
            mask = df["_pos"] == pos
            if not mask.any():
                continue
            preds = self._predict_position(pos, df[mask])
            for (pid, gw, home), val in zip(
                [m for m, keep in zip(meta, mask.tolist()) if keep], preds
            ):
                vadj = self._venue_adjustment(pid, home)
                idx = gw - next_gw
                results[pid][idx] += float(val) + vadj  # DGW fixtures sum
                self.last_venue[pid][idx] += vadj

        blended = self._cold_start_blend(results, season_matches)
        return self._apply_form_credit(blended, next_gw, gws)

    # -- early-season form credit ------------------------------------------
    # Retiring ep_next (2026-09-02) left the model blind to current-season
    # form for ~5 GWs, because OpenFPL's form windows still lean on last
    # season this early - it favoured unproven season-openers over proven
    # in-form players (Joao Pedro, Tzolis). This restores a SMALL, capped
    # slice of current-season form WITHOUT ep_next's fluke-chasing:
    #   - non-penalty xG + xA per 90 (real underlying, not points), so a
    #     penalty-assisted haul on low npxG (the Bruno case) earns little
    #     while two genuinely high-npxG games (Joao Pedro) earn credit
    #   - a dead-band trusts the model within a bonus/conversion allowance
    #     of the xG-implied floor, so an established elite performing to
    #     expectation (Haaland) is NOT dinged
    #   - hard cap +/-1.0 per GW: no single haul can swing a captaincy
    #   - only while < 5 current-season matches; phases to zero by match 5
    # Falsifiable expectation (docs/fixture_layer_design.md), scored at
    # GW8: improves played-RMSE and ordering for < 5-match players over
    # GW3-GW8, without resurrecting fluke-chasing, or it comes out.
    FORM_APPEAR = 2.0
    FORM_GOAL_PTS = {1: 6, 2: 6, 3: 5, 4: 4}
    FORM_ASSIST = 3.0
    FORM_CAP = 1.0
    FORM_FORGIVE = 2.0          # bonus/conversion the raw-xG floor can't see
    FORM_FULL_TRUST = 5         # current-season matches before credit is 0
    FORM_SEASON_START = "-08-01"

    def _current_us_form(self, pid: int) -> tuple[int, float, float]:
        """(current-season understat matches, npxG/90, xA/90). Empty when
        the player has no understat mapping or no current-season match."""
        uid = self._us_player_map.get(pid)
        if not uid:
            return 0, 0.0, 0.0
        season = str(self.understat.season)
        cutoff = season + self.FORM_SEASON_START
        cur = [m for m in (self.understat.player_matches(uid) or [])
               if str(m.get("date", "")) >= cutoff]
        mins = sum(float(m["time"]) for m in cur)
        if not cur or mins <= 0:
            return len(cur), 0.0, 0.0
        npxg90 = sum(float(m.get("npxG") or 0.0) for m in cur) / mins * 90.0
        xa90 = sum(float(m.get("xA") or 0.0) for m in cur) / mins * 90.0
        return len(cur), npxg90, xa90

    def _apply_form_credit(
        self, proj: dict[int, list[float]], next_gw: int, gws: list[int],
    ) -> dict[int, list[float]]:
        self.last_form: dict[int, list[float]] = {
            pid: [0.0] * len(gws) for pid in proj
        }
        for pid, xs in proj.items():
            el = self._elements.get(pid)
            if el is None:
                continue
            m, npxg90, xa90 = self._current_us_form(pid)
            if m == 0 or m >= self.FORM_FULL_TRUST:
                continue
            gp = self.FORM_GOAL_PTS.get(el["element_type"], 4)
            floor = self.FORM_APPEAR + npxg90 * gp + xa90 * self.FORM_ASSIST
            for i, model_gw in enumerate(xs):
                w = max(0.0, (self.FORM_FULL_TRUST - (m + i)) / self.FORM_FULL_TRUST)
                if w == 0.0:
                    continue
                if model_gw < floor:
                    cr = min(self.FORM_CAP, w * (floor - model_gw))
                elif model_gw > floor + self.FORM_FORGIVE:
                    cr = -min(self.FORM_CAP,
                              w * (model_gw - floor - self.FORM_FORGIVE))
                else:
                    cr = 0.0
                xs[i] = round(model_gw + cr, 3)
                self.last_form[pid][i] = round(cr, 3)
        return proj

    def form_report(self) -> dict[int, list[float]]:
        """Per-player per-GW form credit already folded into the returned
        projection, for ledger attribution."""
        return getattr(self, "last_form", {})

    # Venue term: the model's trained weights price home advantage at
    # ~zero while 2025-26 (29,757 hindsight-free player-GW rows,
    # scripts/fixture_layer_calib.py venue-only fit) measures the real
    # home-away residual at +0.375 (MID/FWD) and +0.440 (GK/DEF) points
    # per match. One parameter per group, sign known in advance, applied
    # to the MODEL output pre-blend (never to ep_next). Falsifiable
    # expectation recorded in docs/fixture_layer_design.md: it must
    # improve home-vs-away ordering accuracy over GW3-GW8 or it comes
    # out, same as DefCon's standing commitment.
    VENUE_HOME_COEF = {"GKDEF": 0.440, "MIDFWD": 0.375}

    def _venue_adjustment(self, pid: int, home: bool) -> float:
        et = self._elements.get(pid, {}).get("element_type")
        group = "GKDEF" if et in (1, 2) else "MIDFWD"
        half = self.VENUE_HOME_COEF[group] / 2.0
        return half if home else -half

    def venue_report(self) -> dict[int, list[float]]:
        """Per-player per-GW venue adjustment (pre-blend) from the most
        recent project_all call. Multiply by the blend weight for the
        post-blend contribution."""
        return getattr(self, "last_venue", {})

    # Season matches before the model stands alone.
    # 5 -> 3 on 2026-09-01: hindsight-free GW1+GW2 scoring
    # (scripts/blend_eval.py) had model-only beating the deployed blend
    # on played-RMSE both weeks (3.64 vs 3.71; 3.01 vs 3.18).
    # 3 -> 2 on 2026-09-02: ep_next retired at 2+ matches. Verified
    # corr(ep_next, form) = 0.993 across 357 players - at this point of
    # the season ep_next IS the form chart (a trailing 2-game points
    # average), which injected one 23-point haul straight into the
    # captain ordering. ep's only legitimate remaining job (correcting
    # the model's constant early-season under-prediction) cannot change
    # orderings, while its form-chasing demonstrably corrupts them.
    # Magnitude de-bias is a GW8 decision, as a constant, never via a
    # form chart. ep_next still covers GW1-2 where the model is blind.
    COLD_START_FULL_TRUST = 2

    def _cold_start_blend(
        self, results: dict[int, list[float]], season_matches: dict[int, int]
    ) -> dict[int, list[float]]:
        """Early-season the FPL history features are empty and the model
        under-predicts everyone, so blend with the FPL site's ep_next
        (which carries last-season priors), weighted by matches played:
        pure ep_next at GW1, pure OpenFPL from ~GW6.

        The per-player split is kept on self.last_blend so the UI can
        show how much of a displayed number is FPL's ep_next vs the
        model's own output (raw, pre-minutes/defcon)."""
        blended_players = 0
        self.last_blend: dict[int, dict] = {}
        for pid, xs in results.items():
            n = season_matches.get(pid, self.COLD_START_FULL_TRUST)
            w = min(n / self.COLD_START_FULL_TRUST, 1.0)
            ep = float(self._elements[pid].get("ep_next") or 0.0)
            self.last_blend[pid] = {
                "model_raw": round(xs[0], 3) if xs else None,
                "ep_next": ep,
                "model_weight": round(w, 2),
            }
            if w >= 1.0:
                continue
            results[pid] = [round(w * x + (1.0 - w) * ep, 3) for x in xs]
            blended_players += 1
        if blended_players:
            log.info(
                "cold-start blend applied to %d players (< %d season matches)",
                blended_players, self.COLD_START_FULL_TRUST,
            )
        return results

    def blend_report(self) -> dict[int, dict]:
        """{pid: {model_raw, ep_next, model_weight}} for the most recent
        project_all call. model_raw is the next-GW OpenFPL output BEFORE
        blending (and before minutes/defcon adjustments)."""
        return getattr(self, "last_blend", {})

    def project(self, player_id: int, horizon: int = 5) -> list[float]:
        return self.project_all(horizon, pool=[player_id]).get(player_id, [])


# -- module-level stable interface ----------------------------------------

_default_model: OpenFPLModel | None = None


def _fallback_projections(client: FPLClient, horizon: int) -> dict[int, list[float]]:
    log.error(
        "OPENFPL UNAVAILABLE - falling back to FPL ep_next, flat across the "
        "horizon. Projections will be markedly worse. %s", vendors.SETUP_HINT,
    )
    bootstrap = client.bootstrap()
    return {
        el["id"]: [float(el.get("ep_next") or 0.0)] * horizon
        for el in bootstrap["elements"]
    }


def project_all(client: FPLClient, horizon: int = 5) -> dict[int, list[float]]:
    global _default_model
    try:
        if _default_model is None or _default_model.client is not client:
            _default_model = OpenFPLModel(client)
        return _default_model.project_all(horizon)
    except FileNotFoundError:
        return _fallback_projections(client, horizon)
    except ImportError as exc:
        log.error("OPENFPL DEPENDENCY MISSING: %s", exc)
        return _fallback_projections(client, horizon)


def blend_report() -> dict[int, dict]:
    """Blend split for the most recent module-level project_all call
    (empty when the fallback served or nothing has been projected)."""
    return _default_model.blend_report() if _default_model else {}


def venue_report() -> dict[int, list[float]]:
    """Venue adjustments (pre-blend) for the most recent module-level
    project_all call; empty when the fallback served."""
    return _default_model.venue_report() if _default_model else {}


def form_report() -> dict[int, list[float]]:
    """Early-season form credit folded into the most recent module-level
    project_all output; empty when the fallback served."""
    return _default_model.form_report() if _default_model else {}


def opponent_form(fpl_team_id: int, last: int = 5) -> dict | None:
    """A club's recent defensive record from understat: xGA per match,
    goals conceded, and clean sheets over the last `last` matches.
    Evidence for single-gameweek decisions - a September opponent and a
    March opponent are different teams. None when no data (or no
    project_all has loaded understat yet)."""
    m = _default_model
    if m is None or not getattr(m, "_us_ready", False):
        return None
    from .understat import FPL_TO_UNDERSTAT_TEAM

    name = m._team_name.get(fpl_team_id, "")
    hist = m._us_team_hist.get(FPL_TO_UNDERSTAT_TEAM.get(name, name), [])
    tail = hist[-last:]
    if not tail:
        return None
    # Understat PREMIER LEAGUE matches only: no Championship, no
    # friendlies. Promoted clubs therefore have only this season's
    # matches; established clubs' window can reach back into last
    # May's fixtures. Raw means over the matches actually present -
    # nothing is padded or shrunk. The counts below exist so the UI
    # can say which basis a number stands on.
    season_start = f"{m.understat.season}-07-01"
    cur = sum(1 for x in tail if str(x.get("date", "")) >= season_start)
    return {
        "matches": len(tail),
        "current_season_matches": cur,
        "prev_season_matches": len(tail) - cur,
        "thin": len(tail) < last,
        "xga_per_match": round(sum(float(x["xGA"]) for x in tail) / len(tail), 2),
        "conceded": sum(int(x["missed"]) for x in tail),
        "clean_sheets": sum(1 for x in tail if int(x["missed"]) == 0),
    }


def project(player_id: int, horizon: int) -> list[float]:
    """Stable signature: per-GW xPts for one player over the horizon."""
    global _default_model
    if _default_model is None:
        _default_model = OpenFPLModel(FPLClient())
    return _default_model.project(player_id, horizon)
