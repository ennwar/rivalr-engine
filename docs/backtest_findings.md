# OpenFPL reimplementation backtest — findings (2026-08-14)

Backtest: `tests/backtest_openfpl.py`, one-GW-ahead, 2024-25 GW32-38,
scored against OpenFPL paper (arXiv 2508.09992) Table 4.

## Result: FAIL

| Bucket  | n    | ours  | paper | deviation |
|---------|------|-------|-------|-----------|
| Zeros   | 3209 | 0.946 | 0.818 | +15.7%    |
| Blanks  | 1464 | 1.546 | 1.291 | +19.8%    |
| Tickers | 185  | 1.325 | 1.517 | −12.7%    |
| Haulers | 455  | 5.295 | 5.142 | +3.0%     |

Cold-start blend and minutes scaling were bypassed (direct
`_predict_position` calls). No DefCon logic exists in this codebase.

## Diagnosis (in the mandated order)

Ground truth: `vendor/OpenFPL/data/samples.csv` — 4 real player rows at
GW38 2024-25 with all 228 feature values, rebuilt with our pipeline and
diffed feature-by-feature.

- **(a) Feature order — CORRECT.** Our `_feature_order` is byte-identical
  to `xscaler.feature_names_in_`, which equals samples.csv column order.
- **(b) Window semantics — CORRECT for {1,3,5,10}.** Windows exclude the
  current fixture ("points 1" at GW38 = the GW37 value for all four
  sample players) and are means over the trailing fixture rows,
  including 0-minute rows. Exact match on every sample.
- **(c) NaN handling — CORRECT.** nan_to_num before and after scaling,
  as in play.ipynb. The samples' position-inapplicable NaNs vs our
  computed 0.0s are numerically identical after nan_to_num.
- **(d) Position slicing — CORRECT.** 196 GK / 206 outfield, subsets of
  the scaler names, matching each model's `n_features_in_`.
- **(e) Inverse transform — CORRECT.** yscaler round-trip confirms
  `points = y*33 − 7`.

## Actual deviations found (feature-level, all verified numerically)

1. **`player relevant fpl points` is misdefined in our pipeline.**
   Paper: *"FPL points achieved by the player at the venue of the
   upcoming match (i.e., home or away)"* — a venue-split points series.
   Ours: `total − appearance − bonus` (a documented guess, now falsified).
   Venue-split reproduces the samples EXACTLY at w=1/3/5/10 for all four
   players. 5 features per row, all positions. Largest single error
   source (e.g. Mateta w10: theirs 5.30, ours 0.80).

2. **The 38-window crosses the season boundary; ours truncates.**
   Verified per player: Becker's w38 sum needs +6 = exactly his final
   2023-24 match; Mateta +20 = exactly his final 2023-24 match (20 pts
   vs Villa); Mitoma +0 (0 minutes in his 2023-24 tail — unplayed
   previous-season matches are excluded); Huijsen +0 (no 2023-24 PL
   data). Effective rule: extend the per-match series into the previous
   season using played matches only. Affects every `* 38` feature
   (~46 of 228) all season, and shorter windows in early gameweeks.

3. **Understat player-level windows 10/38 composition differs** (1/3/5
   match exactly). Mechanism unresolved — plausibly the same
   cross-season extension and/or fixture alignment; needs one more
   sample-diff pass after fixing (2).

4. **Evaluation-universe differences (backtest inputs, not pipeline
   bugs, disclosed):**
   - availability set to 1.0 for everyone (vaastav has no point-in-time
     injury flags); the paper uses live 0/25/50/75/100% flags. Directly
     inflates our Zeros/Blanks error — injured players we predict
     points for actually score 0.
   - AM managers included in the paper's Table 4, excluded by us.
   - Tickers n=185 is small; the −12.7% may be sample noise.

## Not done (by instruction)

No pipeline code changed, no tuning. The failing test is committed
as-is. Proposed next step, pending decision: implement (1) venue-split
relevant points and (2) cross-season windows — both are documented
correctness fixes derived from the paper text and sample data, not
tuning — then rerun the backtest and report the new deltas as they come
out. Also pending: our ledger's scoring buckets (0 / 1-3 / 4-9 / 10+)
do not match the paper's definitions used here (did-not-play / ≤2 /
3-4 / ≥5); aligning them changes our weekly accuracy reports.
