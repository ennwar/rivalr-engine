# Fixture adjustment layer — DESIGN ONLY, not built

Status: awaiting sign-off. Nothing in this document is implemented; no
production code path references it. (2026-09-01)

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
- **F(f)** — FPL FDR bridge, `(3 - FDR) / 2`, as a third regressor.
  Expectation: FDR mostly collapses onto A/D once shrinkage works; if
  its fitted coefficient is ~0 it is dropped before shipping.

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
