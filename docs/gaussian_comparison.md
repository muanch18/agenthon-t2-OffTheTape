# PCA versus a simple joint Gaussian model

This experiment compares the frozen production PCA numeric logic with a
deliberately simpler Gaussian random walk. It uses the frozen historical
manifest from `final_historical_cases.md`. Only training and validation cases
are supported by the CLI; holdout cases remain unscored.

Run in the project evaluation environment from the repository root:

```powershell
python -m experiments.gaussian_comparison
```

Output: `reports/gaussian_comparison.json`, an ignored research artifact with
per-case/per-seed scores, normalized summaries and paired confidence intervals.
Default root seeds are 2026, 17 and 83; both models use 1,000 draws. The manifest
hash and source panel hashes bind the run to the selected inputs. A changed
panel file causes the runner to stop. Errors stop the run rather than silently
dropping difficult cases or substituting another model.

## Models

PCA uses the same 504-change lookback, two factors, AR(1), five-day historical
innovation blocks, FX log-price transformation, per-panel separation and seed
derivation as production. A regression test compares its draws byte-for-byte
with `Track2Agent` on a synthetic mixed case.

Gaussian uses each asset's trailing 300 level rows, first differences, the
published long-gap filter, date alignment, full sample mean and covariance,
and coherent cross-horizon covariance `min(h1,h2) * Sigma`. All assets share
one covariance matrix, including rates/FX on mixed cases. FX is modeled in
native level space, matching the daily-level M0 specification. There is no
PCA, AR term, drift shrinkage, family adjustment or text inference. It adds
1.1e-9 diagonal jitter, with a diagonal covariance fallback if factorization
fails. The benchmark does not implement monthly or return-target M0 modes,
and uses our seeds and draw count rather than M0's exact ensemble.

This is a comparison of two complete numeric alternatives. Any difference
cannot yet be attributed solely to PCA: history window, FX representation,
and cross-panel dependence also differ. Do not replace production solely
because one aggregate validation statistic improves.

## Scores and uncertainty

Official installed `_composite` computes fair marginal CRPS, order-1/2
variogram and tail pinball components. The local runner reconstructs
expected-error divisors from the Gaussian mean/covariance using the published
daily M0 formulas, with weights 0.5/0.3/0.2 (single cell 5/7, 0, 2/7) and
composite clipping at 8. These are normalized research scores, not official
leaderboard scores or private card scales.

Reference specification:
https://github.com/Agenthon-2026/track2-forecasting-public/blob/main/docs/M0-BASELINE.md

The joint divisor uses the folded-normal fractional moment, with a two-term
large-mean expansion above |mean difference|/sd = 100 to avoid subtracting
nearly equal numbers. Analytic checks at zero mean and Monte Carlo checks at
nonzero means validate this helper. No outcome enters a divisor.

Report each split and shape separately. Lower scores are better. The paired
difference is PCA minus Gaussian, so positive favors Gaussian. Seeds average
within each origin before 2,000 paired bootstrap resamples. The interval is
an exploratory 95% origin-bootstrap interval, not a guarantee or a multiple-
comparison-adjusted claim. Origin windows do not overlap, but financial
regimes and overlapping estimation histories can still create temporal
dependence; these intervals can understate uncertainty. Cases at the same
origin across shapes are not independent replications. Seed variation is
Monte Carlo variation, not additional market observations.

This suite has no historical text and does not validate actual F1–F4 family
semantics. Numerical success alone does not establish Final performance.

## Evaluation environment

Initial run: Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, SciPy 1.18.1,
qfbench2-common 2.3.1 and qfbench2-track-forecasting 3.1.0, in the existing
offline verifier image. Installed component functions are reused; the newer
published expected-error normalization and clipping are applied by this
research runner. This is not a claim that the installed verifier is the latest
organizer release or that the full Final verification contract was rehearsed.

## Initial results (2026-10-16)

Completed 917 cases (691 training, 226 validation), each with three seeds for
both models: 5,502 model/case/seed evaluations. No model failures occurred.
The 156 holdout cases were not scored. Validation normalized composite means:

| Shape | Origins | PCA | Gaussian | PCA minus Gaussian, exploratory 95% interval |
|---|---:|---:|---:|---|
| Single rates | 47 | 0.9756 | 0.9452 | +0.0305 [-0.0229, +0.1098] |
| Rates curve | 45 | 1.0813 | 1.0695 | +0.0117 [-0.0318, +0.0651] |
| Single FX | 47 | 1.1059 | 1.1104 | -0.0045 [-0.0649, +0.0359] |
| FX basket | 44 | 1.0270 | 1.0414 | -0.0145 [-0.0764, +0.0349] |
| Mixed rates/FX | 43 | 1.0673 | 1.0492 | +0.0181 [-0.0114, +0.0467] |

Every validation interval spans zero. This is weak evidence for a small
Gaussian advantage in rates and mixed cases, not grounds to replace the
submission. Training favored PCA on mixed cases, so that particular advantage
did not persist into validation. FX basket PCA wins on average but only at
34% of validation origins; a minority of larger wins drives its average.
Gaussian's small all-case validation advantage has the same sign at all three
seeds, but seed consistency does not establish market generalization.

Keep both implementations for further controlled tests. The next useful
ablation is drift shrinkage with other settings fixed; do not select per-shape
production routing from these few validation groups. No production model or
submission descriptor was changed.
