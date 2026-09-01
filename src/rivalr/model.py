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
        meta: list[tuple[int, int]] = []  # (player_id, gw)
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
                    meta.append((pid, gw))

        if not rows:
            return {}
        df = pd.DataFrame(rows)
        for col in self._feature_order:
            if col not in df.columns:
                df[col] = math.nan  # rank features etc.

        results: dict[int, list[float]] = {
            pid: [0.0] * len(gws) for pid in pool if pid in self._elements
        }
        for pos in ["GK", "DEF", "MID", "FWD"]:
            mask = df["_pos"] == pos
            if not mask.any():
                continue
            preds = self._predict_position(pos, df[mask])
            for (pid, gw), val in zip(
                [m for m, keep in zip(meta, mask.tolist()) if keep], preds
            ):
                results[pid][gw - next_gw] += float(val)  # DGW fixtures sum

        return self._cold_start_blend(results, season_matches)

    COLD_START_FULL_TRUST = 5  # season matches before the model stands alone

    def _cold_start_blend(
        self, results: dict[int, list[float]], season_matches: dict[int, int]
    ) -> dict[int, list[float]]:
        """Early-season the FPL history features are empty and the model
        under-predicts everyone, so blend with the FPL site's ep_next
        (which carries last-season priors), weighted by matches played:
        pure ep_next at GW1, pure OpenFPL from ~GW6."""
        blended_players = 0
        for pid, xs in results.items():
            n = season_matches.get(pid, self.COLD_START_FULL_TRUST)
            w = min(n / self.COLD_START_FULL_TRUST, 1.0)
            if w >= 1.0:
                continue
            ep = float(self._elements[pid].get("ep_next") or 0.0)
            results[pid] = [round(w * x + (1.0 - w) * ep, 3) for x in xs]
            blended_players += 1
        if blended_players:
            log.info(
                "cold-start blend applied to %d players (< %d season matches)",
                blended_players, self.COLD_START_FULL_TRUST,
            )
        return results

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


def project(player_id: int, horizon: int) -> list[float]:
    """Stable signature: per-GW xPts for one player over the horizon."""
    global _default_model
    if _default_model is None:
        _default_model = OpenFPLModel(FPLClient())
    return _default_model.project(player_id, horizon)
