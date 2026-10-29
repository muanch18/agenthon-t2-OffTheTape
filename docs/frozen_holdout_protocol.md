# Frozen historical holdout protocol — 2026-10-16

This protocol is committed and pushed before any holdout outcome is scored.
It evaluates the two volatility rules selected on 2005–2017 training, after
exploratory checks on 2018–2021 validation. The holdout is the existing fixed
2022–2024 manifest roster. No new origins, exclusions, shapes or seeds are
selected from holdout behavior. Missing-data exclusions are already frozen.

## Frozen candidates and controls

1. PCA control: production numeric logic, 504 changes, two factors, AR(1),
   five-day innovation blocks, separate panel seeds, rates differences and FX
   log changes, full historical drift, no text. Candidate: 50/50 current and
   20-observation-half-life EWMA variance blend, sd ratio capped [0.5, 2].
2. Gaussian control: trailing 300 level rows, full drift, joint native-level
   sample covariance and coherent horizons. Candidate: trailing-60-change
   standard deviation, sd ratio capped [0.5, 2].

Only these four arms are scored. No zero/half drift, alternative volatility
blend, family routing, or cross-panel adjustment is added. Both candidate
implementations must match the prior experiment exactly on synthetic tests.
Root seeds remain 2026, 17 and 83; 1,000 scenarios per arm. Random shocks are
paired within each model/control comparison. Fixed full-Gaussian expected-error
divisors, official installed component functions and composite clipping at 8
are unchanged. These remain local historical research scores, not official
Final leaderboard scores.

## Frozen analysis and decision rule

Primary estimands: equal-case-weighted mean normalized composite loss for
each candidate minus its own control across the entire holdout. Average seeds
within case; do not count seeds as independent market observations. There are
two primary comparisons. Draw 10,000 moving-block bootstrap replicates with
seed 901, using contiguous non-wrapping blocks of four ordered origins and
keeping every shape and seed at an origin together. Truncate the concatenated
sample to the original origin count and retain original case weighting.

For each primary comparison report the two-sided 97.5% percentile interval
(quantiles 0.0125 and 0.9875), a Bonferroni adjustment for two comparisons.
Label the candidate supported on this holdout only if the adjusted interval
is entirely below zero; otherwise label it not confirmed. This bootstrap
interval is approximate, especially with few origins and persistent market
regimes; the multiplicity adjustment does not make it an exact test.

Report means, paired deltas, relative changes and per-seed sensitivity.
Breakdowns by the existing shapes and calendar year are diagnostic only,
with unadjusted 95% block intervals; do not use them to select routing or
retune candidates. Cross-model ranking is descriptive, not a third primary
claim. Do not automatically modify production or build/upload a new Docker
submission as part of evaluating this protocol.

## Freeze and audit

`experiments/frozen/holdout_20261016.json` pins candidate identifiers, seeds,
sample sizes, manifest hash, code hashes, Python/package versions, bootstrap
settings and decision rule. Code hashes normalize text line endings to LF so
Windows Git checkouts do not invalidate equivalent source. Manifest and data
hashes cover exact bytes. Verify all hashes and versions before scoring.
Source panel hashes from the historical manifest must still match. Failures
stop the run; no difficult case is silently skipped or replaced. The runner
checks cutoffs and output validity and refuses to overwrite an existing result.

```powershell
python -m experiments.frozen_holdout
```

The result is `reports/frozen_holdout_20261016.json`. Once evaluated, this
holdout is consumed and must never again be called untouched. Any new tuning
after these scores are known needs independent prospective evidence. This
test lacks historical text, real Final family semantics and live 2026 market
conditions. It cannot establish actual Final competition performance.
