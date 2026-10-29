# Research results and evidence

All values below are local historical research measurements, not official
competition standings, live forecasts or investment returns.

## Evaluation design

The fixed numerical roster contains 1,073 admitted cases: 691 training
(2005–2017), 226 validation (2018–2021), and 156 holdout (2022–2024, 33 origins).
Dates, rotating assets and 2–12 weekday horizons are selected independently of
future moves. Missing exact origin/target dates and split-crossing targets are
excluded and logged. Five shapes cover single rates, curves, single FX,
FX baskets and mixed panels. Different origins have non-overlapping target
windows; estimation histories and economic regimes remain dependent.

Most comparisons use three seeds and 1,000 scenarios; cross-panel comparisons
use five seeds. Scoring combines fair marginal CRPS, order-1/2 variogram and
tail pinball. The runner reconstructs fixed Gaussian/M0-like expected-error
divisors using published daily formulas and clips composite loss at 8. It uses
the historical installed scorer components, not a claim of the latest official
evaluation environment. Lower loss is better.

## Frozen volatility confirmation

Settings were selected on training, explored on validation, and frozen with
the protocol in commit `9113289` before scoring holdout. Scenario shocks are
paired within each candidate/control comparison.

| Comparison | Control | Candidate | Reduction | Difference | Adjusted 97.5% interval |
|---|---:|---:|---:|---:|---|
| PCA variance blend vs base PCA | 1.357877 | 1.305176 | 3.88% | -0.052702 | [-0.124580, -0.001389] |
| Gaussian recent-60 vs base Gaussian | 1.332513 | 1.278862 | 4.03% | -0.053651 | [-0.147028, +0.003102] |

Both improved at all seeds. Only PCA met the predefined rule that its interval
be entirely below zero. The analysis used 10,000 non-wrapping four-origin
moving-block bootstrap replicates, preserving case weights and grouping every
shape/seed at an origin. Intervals were adjusted for two primary comparisons.
PCA's upper bound was narrowly below zero; this is support on this roster,
not a universal accuracy claim.

| Year | PCA reduction vs own control | Gaussian reduction vs own control |
|---|---:|---:|
| 2022 | 7.72% | 7.90% |
| 2023 | 1.31% | 0.48% |
| 2024 available cases | -0.63% | 1.12% |

Negative reductions mean worse performance. Gains concentrate in 2022; PCA
slightly regresses in 2024. The holdout is consumed and cannot serve as untouched
confirmation after further tuning.

Sources: [protocol](frozen_holdout_protocol.md), [findings](frozen_holdout_results.md),
[tracked audit](../experiments/frozen/holdout_20261016_results.json).

## Exploratory direct comparison

The later comparison averages seeds within case and compares adjusted Gaussian
against adjusted PCA on the saved holdout records. Gaussian loss is 1.278862
versus PCA's 1.305176, a 2.02% relative reduction. Gaussian minus PCA is
-0.026314, with exploratory 95% origin-block interval [-0.077525, +0.013264].
Gaussian wins 76/156 cases and 17/33 origin averages. Sensitivity intervals
for block lengths 2, 6 and 8 also cross zero.

This does not establish superiority or equivalence. The models differ in
preprocessing, history, dynamics, dependence and shock distributions, so it
does not isolate PCA's value. It was performed after inspecting the holdout.
[Direct comparison details](direct_candidate_comparison.md).

## Other experiments and decisions

| Experiment | Evidence | Decision and boundary |
|---|---|---|
| Base PCA vs base Gaussian | All per-shape validation intervals cross zero | Keep both; not architecture attribution |
| Zero/half/full drift | Training selected zero; neither model improved validation | Retain full drift |
| Independent/half/full rates–FX dependence | Tiny validation effects that change sign across seeds | Do not enable these adjustments |
| Recent volatility | Selected rules improve validation by 0.98% PCA / 1.63% Gaussian; intervals cross zero | Freeze candidates for holdout test above |
| Band-SETAR | Validation loss rises 0.134%; all seeds regress | Keep AR(1); not a general rejection of threshold models |
| Macro conditioning | Full lexical-proxy held-out composite ratio 1.0017 vs numeric-only 1.0000 | Keep text disabled; not LLM-quality evidence |

Methods: [Gaussian](gaussian_comparison.md), [drift](drift_ablation.md),
[dependence](cross_panel_ablation.md), [volatility](volatility_ablation.md),
[SETAR](setar_ablation.md), [text proxy](conditioning_backtest_notes.md).
The text suite has a different roster and raw-score-ratio method; its values
must not be combined with the normalized numerical table.

## Engineering evidence versus accuracy

The documented Linux rehearsal passed official smoke gates on 104 public units.
A later license-only image rebuild passed the worked exemplar. The feature
refactor preserved forecast values/fallback flags on all 104 cached units at
seed 2026. The consolidated suite passed 70 tests. None establishes forecast
accuracy or byte identity between current source and the archived Docker image.

Limits include few independent origins, persistent regimes, approximate
bootstrap inference, missing-date exclusions, no actual Final family semantics,
and no established point-in-time vintages from cutoff filtering alone. Future
confirmation needs independent dates and settings fixed before scoring.
Production integration of adaptive volatility remains separate.
