"""Early-season form credit: design search + validation BEFORE shipping.

Design under test (xG-based, capped, early-season-only, symmetric):
  goal_pts = {GK:6, DEF:6, MID:5, FWD:4}; assist = 3; appearance = 2
  cur: non-penalty xG90 + xA90 over CURRENT-season understat matches
  form_floor  = appearance + npxG90*goal_pts + xA90*3
  UP   when model_gw < form_floor           -> +min(CAP, w*(floor - model))
  DOWN when model_gw > form_floor + FORGIVE -> -min(CAP, w*(model - floor - FORGIVE))
  dead-band [floor, floor+FORGIVE]: trust the model (bonus/conversion the
    raw-xG floor doesn't see). w(m)=max(0,(5-m)/5), phased over the horizon
    by assumed matches m+i. Non-penalty xG is what separates Joao Pedro
    (two high-npxG games) from the Bruno case (one penalty-assisted haul).

Run: uv run python scripts/form_credit_eval.py
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rivalr.fetch import FPLClient
from rivalr import model as M, minutes, defcon

APPEAR = 2.0
GOAL_PTS = {1: 6, 2: 6, 3: 5, 4: 4}
ASSIST = 3.0
CAP = 1.0
FORGIVE = 2.0
FULL_TRUST = 5
SEASON_START = "2026-08-01"


def per90(matches, key):
    mins = sum(float(m["time"]) for m in matches)
    if mins <= 0:
        return 0.0
    return sum(float(m.get(key) or 0.0) for m in matches) / mins * 90.0


def form_floor(el, cur):
    gp = GOAL_PTS[el["element_type"]]
    return APPEAR + per90(cur, "npxG") * gp + per90(cur, "xA") * ASSIST


def credit(el, cur, model_gw, m, i):
    """Symmetric, capped, phased credit for one horizon index i."""
    w = max(0.0, (FULL_TRUST - (m + i)) / FULL_TRUST)
    if w == 0.0 or not cur:
        return 0.0
    floor = form_floor(el, cur)
    if model_gw < floor:
        return min(CAP, w * (floor - model_gw))
    if model_gw > floor + FORGIVE:
        return -min(CAP, w * (model_gw - floor - FORGIVE))
    return 0.0


def main():
    c = FPLClient()
    b = c.bootstrap()
    els = {e["web_name"]: e for e in b["elements"]}
    tn = {t["id"]: t["short_name"] for t in b["teams"]}
    raw = M.project_all(c, horizon=5)
    mdl = M._default_model
    est = {}
    dc = {}

    watch = ["João Pedro", "Tzolis", "B.Fernandes", "Haaland", "Gonzalo",
             "Tavernier", "Mbeumo", "Szoboszlai", "Calafiori"]
    ids = {n: els[n]["id"] for n in watch if n in els}
    est = {p: minutes.estimate_minutes(c, p) for p in ids.values()}
    dc = defcon.DefConModel(c).corrections(list(ids.values()), est, horizon=5)

    print(f"{'player':<13}{'m':>2}{'npxG90':>8}{'xA90':>7}{'floor':>7}"
          f"{'mdlG3':>7}{'credG3':>8}{'before':>8}{'after':>8}  fixtures")
    rows = {}
    for n, pid in ids.items():
        el = els[n]
        uid = mdl._us_player_map.get(pid)
        allm = mdl.understat.player_matches(uid) or []
        cur = [x for x in allm if str(x["date"]) >= SEASON_START]
        m = len(cur)
        # per-GW model (raw incl venue) * minutes + defcon; credit added on model pre-minutes?
        # apply credit on the FINAL scale (post-minutes/defcon), matching a display layer
        finals_before, finals_after, creds = [], [], []
        for i in range(5):
            base_final = raw[pid][i] * est[pid].factor + (dc.get(pid) or [0]*5)[i]
            cr = credit(el, cur, base_final, m, i)
            finals_before.append(base_final)
            creds.append(cr)
            finals_after.append(base_final + cr)
        rows[n] = {"before": finals_before, "after": finals_after,
                   "cred": creds, "m": m}
        print(f"{n:<13}{m:>2}{per90(cur,'npxG'):>8.2f}{per90(cur,'xA'):>7.2f}"
              f"{form_floor(el,cur):>7.2f}{finals_before[0]:>7.2f}{creds[0]:>+8.2f}"
              f"{sum(finals_before):>8.2f}{sum(finals_after):>8.2f}")

    print("\n== BRUNO CAPTAIN CHECK (GW3) ==")
    bf = rows["B.Fernandes"]; ha = rows["Haaland"]
    print(f"  before: Bruno {bf['before'][0]:.2f}  Haaland {ha['before'][0]:.2f}"
          f"  ({'BRUNO' if bf['before'][0]>ha['before'][0] else 'HAALAND'} +"
          f"{abs(bf['before'][0]-ha['before'][0]):.2f})")
    print(f"  after:  Bruno {bf['after'][0]:.2f}  Haaland {ha['after'][0]:.2f}"
          f"  ({'BRUNO' if bf['after'][0]>ha['after'][0] else 'HAALAND'} +"
          f"{abs(bf['after'][0]-ha['after'][0]):.2f})")
    swing = (bf['after'][0]-ha['after'][0]) - (bf['before'][0]-ha['before'][0])
    print(f"  Bruno's shift vs Haaland: {swing:+.2f}  "
          f"({'FAIL: Bruno leapfrogs by >1' if swing>1.0 else 'ok: stays close'})")

    print("\n== JOAO PEDRO vs GONZALO (5-GW planner) ==")
    jp = rows["João Pedro"]; go = rows["Gonzalo"]
    print(f"  before: JP {sum(jp['before']):.2f}  Gonzalo {sum(go['before']):.2f}"
          f" -> {'SELL JP' if sum(go['before'])>sum(jp['before']) else 'KEEP JP'}")
    print(f"  after:  JP {sum(jp['after']):.2f}  Gonzalo {sum(go['after']):.2f}"
          f" -> {'SELL JP' if sum(go['after'])>sum(jp['after']) else 'KEEP JP'}")
    tz = rows["Tzolis"]; tv = rows["Tavernier"]
    print(f"  Tzolis before {sum(tz['before']):.2f} after {sum(tz['after']):.2f} "
          f"| Tavernier {sum(tv['before']):.2f}")


if __name__ == "__main__":
    main()
