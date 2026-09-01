"""Hindsight-free evaluation of the cold-start blend on GW1 + GW2.

Reconstructs OpenFPL-only projections AS OF each gameweek's snapshot:
features see only fixtures/understat matches BEFORE that GW's first
kickoff, and availability comes from the ledger snapshot, not today's
bootstrap. Compares against the deployed (blended) snapshot numbers and,
for GW1 (where the blend weight was 0 for everyone), pure ep_next.

Honesty notes:
- GW1 deployed == ep_next-only by construction, so GW1 is a clean
  model-vs-ep head-to-head.
- GW2 ep_next-only is NOT reconstructible: ep_next at snapshot time was
  never stored, and back-solving it through the (then-buggy) minutes
  factor would manufacture numbers. GW2 therefore compares deployed
  blend vs model-only.
- The deployed GW2 numbers carry the phantom-row minutes bug as shipped;
  the model-only reconstruction uses the fixed pipeline. That favours
  model-only slightly and is noted in the output.
"""

import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.WARNING)

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rivalr.fetch import FPLClient
from rivalr.model import POSITIONS, OpenFPLModel

LEDGER = Path(__file__).resolve().parents[1] / "data" / "predictions"


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return float("nan")
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def as_of_minutes_factor(rows: list[dict], past: list[dict], avail: dict) -> float:
    """Replicates minutes.estimate_minutes on as-of data."""
    status = avail.get("status", "a")
    chance = avail.get("chance_of_playing_next_round")
    if status in ("i", "s", "n"):
        mult = 0.0
    elif status == "d":
        mult = (chance if chance is not None else 50) / 100.0
    else:
        mult = 1.0

    recent = rows[-6:]
    if not recent:
        last_minutes = past[-1]["minutes"] if past else None
        if last_minutes is None:
            p_start = 0.6
        elif last_minutes >= 2400:
            p_start = 0.85
        elif last_minutes >= 1200:
            p_start = 0.65
        else:
            p_start = 0.45
        avg_start = 80.0
    else:
        p_start = sum(r["starts"] for r in recent) / len(recent)
        started = [r for r in recent if r["starts"]]
        avg_start = (sum(r["minutes"] for r in started) / len(started)
                     if started else 75.0)
    p_start = min(1.0, max(0.0, p_start)) * mult
    p_cameo = (1.0 - p_start) * 0.5 if mult > 0 else 0.0
    xmins = p_start * avg_start + p_cameo * 18.0
    return min(1.0, xmins / 90.0)


def model_only_asof(client: FPLClient, gw: int, cutoff: str,
                    avail: dict[str, dict], pool: list[int]) -> dict[int, float]:
    """Raw OpenFPL prediction for `gw` using only pre-cutoff data."""
    m = OpenFPLModel(client)
    m._load_artifacts()
    m._load_understat()
    m._load_prev_season()

    # understat team histories: only matches before the cutoff
    m._us_team_hist = {
        t: [e for e in hist if str(e.get("date", ""))[:10] < cutoff]
        for t, hist in m._us_team_hist.items()
    }
    # understat player matches: wrap with a date filter
    orig_pm = m.understat.player_matches
    m.understat.player_matches = lambda uid: [
        x for x in orig_pm(uid) if str(x.get("date", ""))[:10] < cutoff
    ]
    # current-season element rows: only rounds before this gw
    m._current_rows = lambda pid: [
        h for h in m.client.element_summary(pid).get("history", [])
        if h.get("round", 99) < gw
    ]
    # availability as recorded in the snapshot
    for pid_s, a in avail.items():
        el = m._elements.get(int(pid_s))
        if el is not None:
            el = dict(el)
            el["status"] = a.get("status", "a")
            el["chance_of_playing_next_round"] = a.get(
                "chance_of_playing_next_round")
            m._elements[int(pid_s)] = el

    fixtures = [f for f in client.fixtures() if f.get("event") == gw]
    by_team: dict[int, list[dict]] = {}
    for f in fixtures:
        by_team.setdefault(f["team_h"], []).append(
            {"opponent": f["team_a"], "home": True})
        by_team.setdefault(f["team_a"], []).append(
            {"opponent": f["team_h"], "home": False})

    opp_block = {t["id"]: m._team_block(t["id"], "opponent") for t in m._teams}
    own_block = {t["id"]: m._team_block(t["id"], "team") for t in m._teams}

    rows, meta = [], []
    for pid in pool:
        el = m._elements.get(pid)
        if el is None or el["element_type"] not in POSITIONS:
            continue
        try:
            base, rel_home, rel_away = m._player_features(pid)
        except Exception:
            continue
        base.update(own_block[el["team"]])
        for fx in by_team.get(el["team"], []):
            row = dict(base)
            rel = rel_home if fx["home"] else rel_away
            for w, val in rel.items():
                row[f"player relevant fpl points {w}"] = val
            row.update(opp_block[fx["opponent"]])
            row["_pos"] = POSITIONS[el["element_type"]]
            rows.append(row)
            meta.append(pid)

    df = pd.DataFrame(rows)
    for col in m._feature_order:
        if col not in df.columns:
            df[col] = math.nan

    out: dict[int, float] = {pid: 0.0 for pid in pool}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        mask = df["_pos"] == pos
        if not mask.any():
            continue
        preds = m._predict_position(pos, df[mask])
        for pid, val in zip(
            [p for p, keep in zip(meta, mask.tolist()) if keep], preds
        ):
            out[pid] += float(val)
    return out, m


def metrics(pred: dict[int, float], actual: dict[int, float],
            played: set[int]) -> dict:
    common = [p for p in pred if p in actual]
    pl = [p for p in common if p in played]

    def rmse(ids):
        if not ids:
            return float("nan")
        return math.sqrt(sum((pred[p] - actual[p]) ** 2 for p in ids) / len(ids))

    def mae(ids):
        if not ids:
            return float("nan")
        return sum(abs(pred[p] - actual[p]) for p in ids) / len(ids)

    top30 = sorted(common, key=lambda p: -pred[p])[:30]
    return {
        "rmse_all": round(rmse(common), 3),
        "rmse_played": round(rmse(pl), 3),
        "mae_played": round(mae(pl), 3),
        "spearman_top30": round(
            spearman([pred[p] for p in top30], [actual[p] for p in top30]), 3),
        "n_all": len(common),
        "n_played": len(pl),
    }


def main() -> None:
    client = FPLClient()
    cutoffs = {1: "2026-08-21", 2: "2026-08-28"}

    for gw in (1, 2):
        snap = json.loads((LEDGER / f"gw{gw}.json").read_text(encoding="utf-8"))
        base = {int(k): v for k, v in snap["layers"]["base"].items()}
        avail = snap.get("availability", {})
        live = {el["id"]: el["stats"] for el in client.event_live(gw)["elements"]}
        actual = {p: float(live[p]["total_points"]) for p in base if p in live}
        played = {p for p in actual if live[p]["minutes"] > 0}
        pool = [p for p in base if base[p] and base[p][0] is not None]

        raw, m = model_only_asof(client, gw, cutoffs[gw], avail, pool)
        model_only = {}
        for pid in pool:
            rows = m._current_rows(pid)
            past = client.element_summary(pid).get("history_past", [])
            f = as_of_minutes_factor(rows, past, avail.get(str(pid), {}))
            model_only[pid] = raw.get(pid, 0.0) * f

        deployed = {p: float(base[p][0]) for p in pool}

        print(f"\n===== GW{gw} (cutoff {cutoffs[gw]}, "
              f"{len(pool)} players, {len(played & set(pool))} played) =====")
        label_dep = ("deployed blend == PURE ep_next (w=0)" if gw == 1
                     else "deployed blend (w~0.2, as shipped incl. old bugs)")
        for name, pred in (("model-only (as-of)", model_only),
                           (label_dep, deployed)):
            mt = metrics(pred, actual, played)
            print(f"  {name:<45} {mt}")
        if gw == 2:
            print("  ep_next-only: NOT reconstructible (ep at snapshot time "
                  "never stored) - stated, not estimated.")

        # captain-relevance: top-10 by each prediction, their mean actual
        for name, pred in (("model-only", model_only), ("deployed", deployed)):
            top10 = sorted(pool, key=lambda p: -pred[p])[:10]
            mean_actual = sum(actual.get(p, 0.0) for p in top10) / 10
            print(f"  mean ACTUAL points of {name} top-10 picks: "
                  f"{mean_actual:.2f}")


if __name__ == "__main__":
    main()
