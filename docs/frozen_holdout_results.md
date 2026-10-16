# Frozen historical holdout results — 2026-10-16

Candidates and analysis were frozen and pushed in commit `9113289` before
outcomes were scored. The code and data hashes and Python/package versions
matched the protocol. All 156 cases across 33 origins were evaluated at
three seeds: 468 paired case/seed records, 1,872 arm evaluations, no missing
or duplicate case/seed records and no model failures. Three synthetic tests
passed before evaluation, including exact candidate/control parity with the
earlier experiment. The result's protocol hash matches the frozen record.

## Primary results

Lower normalized composite loss is better. Each candidate is compared with
its own unchanged control. The two primary intervals are the predeclared
97.5% moving-block bootstrap intervals, adjusted for two comparisons.

| Candidate | Own control | Candidate | Relative improvement | Candidate minus control, adjusted interval | Frozen conclusion |
|---|---:|---:|---:|---|---|
| PCA current/EWMA variance blend | 1.357877 | 1.305176 | 3.88% | -0.052702 [-0.124580, -0.001389] | Supported on this holdout |
| Gaussian recent-60 volatility | 1.332513 | 1.278862 | 4.03% | -0.053651 [-0.147028, +0.003102] | Not confirmed |

Both candidates improved at every root seed. PCA passes the frozen criterion,
with its upper interval bound only slightly below zero. Gaussian's point
estimate improves more, but its wider interval crosses zero. The Gaussian
candidate's lower absolute score is descriptive; this was not a third
predeclared primary comparison against PCA and does not justify switching
the model on its own.

## Diagnostic results and limits

These breakdowns were predeclared as diagnostics, not as selection rules.

| Calendar period | PCA relative improvement | Gaussian relative improvement |
|---|---:|---:|
| 2022 | 7.72% | 7.90% |
| 2023 | 1.31% | 0.48% |
| 2024 through available October cases | -0.63% | 1.12% |

Much of the overall gain comes from 2022. PCA is slightly worse in 2024.
Both improve the average in rates, rates curves, FX baskets and mixed cases;
single-FX forecasts regress by 0.065% for PCA and 1.60% for Gaussian. Do not
use these observations to invent routing or retune this frozen candidate.

There are only 33 origins, with overlapping model estimation histories and
persistent market regimes. Four-origin bootstrap blocks and multiplicity
adjustment mitigate some uncertainty; they do not establish performance in
live 2026 markets. This suite has no contemporaneous text and no validated
Final family semantics. Scores reconstruct published normalization using the
installed component scorers; this is not an official Final evaluation.

## Decision and artifact state

The PCA volatility blend is the preferred candidate for a separate production
integration and contract-validation task: it retains the existing model and
passed the predeclared holdout test. Keep its rule unchanged if proceeding.
The Gaussian candidate remains a research comparator. Neither the production
candidate configuration nor the Docker image/submission was changed here.

**The 2022–2024 holdout is now consumed.** It is no longer an untouched source
of confirmation for subsequent tuning. Any new optimization requires new
independent or prospective evidence. No further candidate search was performed
after these results were read.

Tracked audit summary: `experiments/frozen/holdout_20261016_results.json`.
The summary contains primary/diagnostic results, the freeze commit and protocol
hash, and the SHA-256 of the detailed local result. Full per-case components
remain in `reports/frozen_holdout_20261016.json` under the existing ignore rule.
