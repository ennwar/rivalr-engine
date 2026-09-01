# Fixture adjustment layer — CALIBRATED, FAILED ITS GATES, NOT SHIPPED

Status: approved to build 2026-09-01 with amendments (FDR dropped
entirely; recent-form windows; real opponent numbers replace FDR in the
captain board). Calibration ran the same day on the full 2025-26 season
(29,757 player-GW rows, scripts/fixture_layer_calib.py) and FAILED the
validation gates fixed below before any code was written. Per those
gates, the layer is NOT in production. The captain-board display change
shipped; the adjustment did not. Failing run preserved here.

## Pre-build check result (the question that had to be answered first)

Is the model underweighting a good signal, or is the signal weak?

- **Attacking channel (MID/FWD vs opponent last-5 xGA)**: the signal
  itself is weak at single-gameweek granularity. Actual points across
  opponent-form quintiles: 3.73, 4.27, 3.84, 3.79, 4.34 — essentially
  flat and non-monotone. The model's response (2.88 -> 3.32 across the
  same quintiles) is small because the true effect is small. The
  intuition that attackers reliably cash in on soft fixtures is not
  supported week-to-week; single-match variance swamps opponent
  strength. The model's narrow opponent-swap range is mostly CORRECT.
- **Clean-sheet channel (GK/DEF vs opponent last-5 xG)**: real slope
  exists (actual 4.19 -> 3.21 across quintiles); the model captures
  about two-thirds of it (2.89 -> 2.25). Mild underweighting.
- **Venue**: the one genuinely missed signal. Real home-away gap +0.35
  (MID/FWD) and +0.44 (GK/DEF); the model's gap is ~0.00.

## Calibration outcome (2025-26, OOF 5-fold by GW blocks)

Grid over window W in {4,5,6} x shrinkage m in {2,4,6}; every config
WORSENED overall OOF RMSE by ~0.5% and the Zeros bucket by ~3%.
Against the pre-agreed gates, best config (W=4, m=2):

- played-RMSE 2.9168 -> 2.9180 (-0.04%; gate required >= +0.5% cut) FAIL
- Zeros bucket +3.16% (gate: no bucket worse than 2%) FAIL
- top-30 Spearman 0.168 -> 0.177 (not falling) pass
- decile calibration slope 0.64 (gate 0.6-1.4) pass, barely, with
  non-monotone deciles

Post-hoc variants, reported for transparency (NOT shipped - they were
not the pre-agreed design): venue-only +0.10% played-RMSE (below gate,
Zeros +2.3%); defence-only -0.06%. Nothing clears the bar.

Harness caveats, disclosed: availability = 1.0 for everyone (vaastav
has no point-in-time flags) and no minutes scaling, which inflates the
Zeros contamination; but played-RMSE is unaffected by Zeros and is
flat, so the conclusion does not rest on the artifact.

## Decision

The gates said: if they fail, the layer does not ship and the failing
run is preserved. They failed. The single-GW fixture signal the layer
was meant to amplify is mostly noise for attackers, mildly present for
defenders, and the venue gap - though real - is worth ~0.1% RMSE, under
the bar. Revisit only with materially new evidence (e.g. multi-season
calibration corpus, or live 2026-27 evidence by GW8 that projections
systematically miss fixture effects).

What DID ship from this work: the captain board now shows each
opponent's actual last-5 defensive record (xGA/match, goals conceded,
clean sheets) instead of FPL's static FDR - the evidence a
single-gameweek call should rest on.

## Addendum 2026-09-02: the VENUE term shipped on its own

Approved separately after the GW3 captain investigation (Bruno-vs-
Haaland decomposition) showed the missing venue term flipping the
model's own top pick. Rationale for shipping despite the global layer's
failed gate: that gate was sized for a speculative multi-feature layer
with many ways to overfit; this is ONE parameter per position group,
sign known in advance, fitted on 29,757 rows, landing where football
knowledge predicts. Pricing a real measured effect at exactly zero is
itself a defect.

Shipped values (pre-blend, never applied to ep_next; centred so home
gets +coef/2 and away -coef/2):
- MID/FWD: home coefficient +0.375  (+/- 0.187 per match)
- GK/DEF:  home coefficient +0.440  (+/- 0.220 per match)

Recorded as the fourth ledger layer ("venue": pre-blend adjustment x
blend weight x minutes factor) so post-GW scoring attributes error to
it independently.

FALSIFIABLE EXPECTATION (recorded before shipping, judged at GW8):
over GW3-GW8, the venue adjustment must improve home-vs-away ordering
accuracy - measured as the fraction of (home player, away player)
pairs, both >= 60 minutes, where the higher-projected player scored
more, with-venue vs without-venue (the without-venue counterfactual is
exactly recoverable from the ledger's venue layer). If it does not
improve, the term comes out - same standing commitment as DefCon.

---

The original design as approved (with amendments) follows, for the
record.

## Problem, measured

After the understat mapping fix, the opponent-swap experiment (GW3
basis, all 19 opponents x 2 venues, raw model) gives:

- Haaland: range 1.55 pts (best Ipswich H 6.21, worst Arsenal A 4.66)
- B.Fernandes: range 2.03 pts
- Same-opponent home-minus-away: +0.01 (Haaland) to +0.32 (Bruno)

OpenFPL's trained weights barely respond to the opponent block or the
venue split. This is inherited from training; we cannot retrain (the
250 joblib models are fixed artifacts). Precedent: DefCon added an
additive layer for a scoring rule the model never saw; this layer does
the same for a signal the model underweights.

## Proposed design

Additive correction per player-fixture, applied to the MODEL component
only, BEFORE the cold-start blend:

    adj(p, f) = k_att[g] * A_opp(f) + k_def[g] * D_opp(f) + v[g] * home(f)  [+ c[g] * F(f)]

where g is the position group (GK/DEF vs MID/FWD — 2 groups, so ~8
parameters in total, deliberately tiny).

- **A_opp** — opponent softness for attacking returns: venue-split
  expected goals conceded per 90 vs league average, from the understat
  xGA windows (5- and 38-match), shrunk toward the league mean by
  matches observed: `x_hat = (n*x + m*mu) / (n + m)` with m = 6. A
  promoted club with 2 matches sits near the mean instead of at a
  noisy extreme.
- **D_opp** — opponent threat for the clean-sheet channel (GK/DEF):
  same construction on opponent xG per 90.
- **home(f)** — plus/minus venue indicator.
- ~~F(f) — FPL FDR bridge~~ — REMOVED at approval (2026-09-01): FDR is
  a season-long static rating and captaincy is a single-gameweek
  decision; September-Everton and March-Everton are different teams.
  Only current-form opponent measures are used. Recent form weighted
  heavily: the window grid was {4,5,6} matches (not season-long), all
  matches rather than venue-split because a venue-split last-5 leaves
  2-3 matches of signal; venue enters as its own additive term.

Caps and floors:
- |adj| <= 1.5 pts (the scale of inter-fixture spread the model
  currently misses).
- Opponent with <2 observed matches AND no prior-season data: A/D
  terms zeroed, FDR-only term capped at +/-0.5.

Interaction rules (to avoid double counting):
- Applied to OpenFPL raw output pre-blend. NEVER to ep_next — FPL's
  number already embeds their own fixture view; adjusting the blended
  total would count fixtures twice for the ep share.
- Minutes factor multiplies afterwards, DefCon adds afterwards —
  pipeline becomes: model raw -> +fixture -> blend with ep_next ->
  x minutes -> +defcon.

## Calibration plan (2025-26, same corpus as the OpenFPL backtest)

1. For every 2025-26 player-GW with minutes >= 60: residual
   r = actual_points - base_projection, where base is the as-of raw
   OpenFPL output from the existing backtest harness.
2. Per position group, ridge regression (small lambda) of r on
   [A_opp, D_opp, home, F]. Report coefficients with standard errors.
3. 5-fold cross-validation, folds = contiguous GW blocks (no
   within-fold leakage of adjacent form).

## Validation gates (fixed NOW, before any code — DefCon-style)

1. **Decile calibration**: bucket held-out player-GWs by predicted
   adj into deciles; mean actual residual per decile must be monotone
   increasing with fitted slope in [0.6, 1.4].
2. **RMSE**: base+adj must cut played-RMSE vs base on out-of-fold
   2025-26 by >= 0.5%, with NO paper bucket (Zeros/Blanks/Tickers/
   Haulers) worsening by > 2%.
3. **Ordering**: mean per-GW Spearman of top-30 projected vs actual
   must not fall.
4. **Live falsifiable expectation, recorded before shipping**: over
   GW3–GW8, (i) the captain board's top pick outscores the no-layer
   counterfactual pick on average, and (ii) played-RMSE improves in at
   least 4 of the 6 snapshot weeks. Failing this at the GW8 review
   removes the layer.

If gates 1–3 fail on 2025-26, the layer does not ship at all and the
failing run is preserved in docs (same discipline as
docs/backtest_findings.md).

## Ledger

A fourth layer per snapshot: `base`, `defcon`, `fixture`, with `final`
= sum, so post-GW scoring attributes error to each layer independently.

## Explicitly not doing

- No retraining or reweighting of OpenFPL artifacts.
- No multiplicative scaling of the whole projection (it would distort
  appearance points, which don't depend on the opponent).
- No bookmaker odds feeds (new dependency, licensing).
