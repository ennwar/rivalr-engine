# rivalr-engine

Local FPL engine. No web app — everything runs from the CLI and outputs to
terminal + JSON. The differentiator is **mini-league intelligence**: it
optimises against your rivals, not just for raw points.

## Modules

| Module | What it does |
|---|---|
| `src/rivalr/fetch.py` | FPL API client, disk cache (6h bootstrap, 1h standings), 429 backoff, cache-miss logging |
| `src/rivalr/rivals.py` | rival squads, chips, banks, transfer history; mini-league effective ownership; SHIELD/SWORD/NEUTRAL classification; per-rival overlap and differentials → `rivals_report.json` |
| `src/rivalr/model.py` | OpenFPL trained-model inference (no retraining), 228 rolling-window features from FPL + Understat |
| `src/rivalr/understat.py` | Understat JSON API client (league + player endpoints), cross-season history, loud failure logging |
| `src/rivalr/minutes.py` | v0 expected-minutes: rolling starts, minutes trend, availability flag, congestion; news-based v1 hook |
| `src/rivalr/optimise.py` | wraps the FPL-Optimization-Tools HiGHS MILP with our projections; three objectives: points / chase / defend |
| `src/rivalr/report.py` | one-command gameweek brief (Telegram-friendly plain text) |
| `src/rivalr/ledger.py` + `score.py` | append-only prediction ledger, post-GW RMSE/MAE by outcome bucket + transfer counterfactual |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and git.

```bash
uv sync
uv run python scripts/setup_vendors.py   # clones OpenFPL (~750MB models) + FPL-Optimization-Tools into vendor/
```

## Run

The gameweek brief (the main command):

```bash
uv run python -m rivalr.report --team <your_entry_id> --league <classic_league_id> --mode chase --target <rival_entry_id>
```

- `--mode points` — plain xPts maximisation (no target needed)
- `--mode chase --target <id>` — attack a rival above you: differentials + variance
- `--mode defend --target <id>` — protect a lead: cover the chaser's squad, prefer shields

All three plans are always solved and shown side by side with honest
(unweighted) expected points and a next-GW swing-vs-target estimate.

Score a finished gameweek against the ledger:

```bash
uv run python -m rivalr.score --gw 5
```

Smoke tests (no team/league needed):

```bash
uv run python scripts/smoke.py            # fetch + model + minutes
uv run python scripts/smoke_optimise.py   # full MILP solve
```

Tests:

```bash
uv run pytest
```

## Upstream integrations (input formats)

**OpenFPL** (`vendor/OpenFPL`, daniegr/OpenFPL): 250 joblib models
(xgboost 3.0.2 / sklearn 1.5.2-era pickles; deps pinned in our
pyproject to their `plug.txt`). Input is a 228-feature vector per
(player, fixture): rolling means of FPL + Understat stats over the
previous {1,3,5,10,38} gameweeks, min-max scaled by `models/xscaler.save`,
sliced per position via `models/features.save`, predicted by 50 models
per position (5 CV folds x 10 candidates), inverse-scaled
(`points = y*33 - 7`), median-ensembled. Predicts one GW at a time; we
re-run per future fixture with the opponent block swapped.

**FPL-Optimization-Tools** (`vendor/FPL-Optimization-Tools`,
sertalpbilal): HiGHS MILP via `highspy`. We bypass its CLI and call
`dev.solver.prep_data` + `solve_multi_period_fpl` directly, writing our
projections into `vendor/.../data/rivalr_<mode>.csv` in its fplreview
format: `ID,Name,Pos,Value,Team,{gw}_Pts,{gw}_xMins,...` (ID = FPL
element id; Pos in G/D/M/F; the solver re-derives prices/teams from
bootstrap-static). Options come from its `comprehensive_settings.json`
plus our overrides (see `SOLVER_OVERRIDES` in `optimise.py`).

## Known v0 limitations

- **Cold start**: OpenFPL features use current-season FPL history; at GW1
  they're empty, so projections blend toward FPL's `ep_next` until a
  player has 5 season matches (logged when active). Understat features do
  cross the season boundary (previous season merged in).
- **Understat player matching** is name-based; new signings without an
  EPL history stay unmatched (count logged at WARNING) and lose their
  player-level xG features.
- **`player relevant fpl points`** (an OpenFPL paper feature) is not
  defined in their repo; we approximate it as
  `total - appearance - bonus`.
- **Minutes v0** is purely statistical (no team news). Plug a news source
  into `minutes.news_adjustment()`.
- **Pre-season**: before your GW1 picks exist the optimiser solves a
  fresh £100m draft instead of transfers; rival squad intel is empty
  until the first deadline passes.
- First projection run scrapes ~1 Understat page per player (politely,
  0.3s apart, cached 24h) — expect a few minutes.

## Ledger

Snapshots record **every element in bootstrap-static** — players the
model can't project carry a `null` projection. Pool filtering exists
only on the solver side; the scoring universe is never shrunk (that
would bias accuracy in our favour).

Naming: `gw{N}.json` first snapshot, `gw{N}_v2.json`... if one already
exists (logged loudly, never overwritten). `*_test.json` files are
plumbing artifacts and are never scored. `rivalr.score` uses the
highest-versioned real snapshot and reports RMSE/MAE in the OpenFPL
paper's buckets (Zeros 0 / Blanks 1-3 / Tickers 4-9 / Haulers 10+) plus
the recommended-vs-actual transfer counterfactual.

## Scheduled pre-deadline snapshot

A Windows Task Scheduler job (`rivalr-snapshot`, hourly) runs:

```bash
uv run python -m rivalr.snapshot --auto --team <id> --league <id>
```

It reads the next deadline from the API (`events -> deadline_time`),
fires once inside the 4h window before it, and guarantees the ledger
write even if projections, rivals or the solver fail (the entry is then
flagged `partial` with the failure reasons). Every run — including
skips — appends to `logs/predictions/run_log.jsonl`; failures also
write `logs/predictions/ALERT_gw{N}.txt` and attempt a Windows toast.
The machine must be awake for the window; check the run log after each
deadline.

## Small leagues

With fewer than 15 entries, mini-league EO is too quantised to weight
by (4 entries = 25% steps). The engine then drops the EO terms from
chase/defend and reports direct pairwise swings instead: per transfer,
the head-to-head points swing against each named rival ("+4.2 vs
Nicholas"), where buying a player a rival owns is neutral and selling a
player they own concedes ground.
