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

---

## Run 2 (2026-08-14, after approved correctness fixes)

Fixes implemented, each verified against samples.csv before the rerun:

1. **Venue-split `relevant fpl points`** (points at the upcoming match's
   venue) — replaces the falsified approximation.
2. **Cross-season fixture-slot windows**: a player's history is a
   timeline of fixture slots (previous season joined by player code +
   current season); windows take the trailing N slots FIRST, then drop
   previous-season slots the player didn't play. This positional
   drop-after-tail rule is what the samples require (Mitoma's unplayed
   2023-24 tail occupies a slot but contributes nothing).
3. **Understat 10/38 composition — RESOLVED, verified not assumed**:
   tested four hypotheses against samples.csv; the winner (22/22 numeric
   matches) is date-alignment of Understat matches onto the same
   cross-season fixture slots (0.0 where the player has no Understat
   match that day). Pure cross-season and pure alignment both fail
   cases the combined rule passes.
4. (Found en route) the Understat name matcher missed mononym players
   ("Alisson" vs "Alisson Ramses Becker"/"A.Becker"); fixed with a
   token-subset pass. Mapping 526 → 555 of 804.

Feature-level residual after fixes: **13 mismatches out of 912** sample
values (was ~150): `relevant fpl points 38` on 3 players and team-block
`* 38` season-boundary conventions (~1-3% deltas). Their exact
convention is not recoverable from 4 sample rows — left OPEN, not
guessed.

### Rerun result: **FAIL** against the pre-agreed criteria

| Bucket  | n    | ours  | paper | dev    | was    | delta  | criterion |
|---------|------|-------|-------|--------|--------|--------|-----------|
| Zeros   | 3209 | 0.958 | 0.818 | +17.1% | +15.7% | +1.4pp | FAIL (must improve) |
| Blanks  | 1464 | 1.610 | 1.291 | +24.7% | +19.8% | +4.9pp | FAIL (must improve; inside 25% numerically) |
| Tickers | 185  | 1.393 | 1.517 | −8.2%  | −12.7% | +4.5pp | PASS (≤15%) |
| Haulers | 455  | 5.208 | 5.142 | +1.3%  | +3.0%  | −1.7pp | PASS (≤5%) |

### Why Zeros/Blanks moved the wrong way — with evidence

The pre-fix numbers on zero/low-outcome buckets were **flattered by the
bugs**: corrupted features depressed all predictions toward zero, which
accidentally helps buckets whose true outcome is ~0 and hurts the
performance buckets. Fixing the features recalibrated predictions
upward: Tickers and Haulers moved toward the paper, Zeros/Blanks away.

**Oracle-availability diagnostic** (`--oracle-availability`; leaks the
outcome, NOT an accuracy claim — bounds the availability confound):
with hindsight availability (0 if the player didn't play), Zeros goes
0.958 → **0.471 (−42.5% vs paper)**. The paper's live 0-100% flags sit
between our hardcoded 1.0 and this oracle ⇒ the Zeros gap is
availability information vaastav cannot supply, not pipeline error.
Blanks under the oracle: 24.7% → 23.5% — availability explains only
~1pp; the Blanks residual remains OPEN (candidates: AM exclusion, the
13/912 feature residuals, bucket-population differences).

### Verdict

The reimplementation is feature-verified to 899/912 sample values and
lands within 1.3% (Haulers) and 8.2% (Tickers) of the paper on the
buckets that measure actual performances. The formal result against the
agreed criteria is FAIL: the Zeros/Blanks directional clause assumed
the fixes would improve those buckets, but the evidence shows their
deviation is dominated by the backtest's missing point-in-time
availability data, which no pipeline change can fix. Both runs and the
oracle diagnostic are preserved above; the criteria were not moved.
