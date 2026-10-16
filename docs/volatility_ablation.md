# Recent volatility experiment

Predeclared rules: current volatility, trailing-60-change standard deviation,
a 50/50 blend of current and trailing-60 variance, and a 50/50 blend of current
and exponentially weighted variance with a 20-observation half-life. All
noncurrent standard-deviation ratios are capped at [0.5, 2]. No rule is tuned
to validation, holdout, particular assets or particular event outcomes.

```powershell
python -m experiments.volatility_ablation
```

Use the same 691 training and 226 validation cases, seeds 2026/17/83 and 1,000
draws as prior experiments. Holdout is never scored. Output goes to
`reports/volatility_ablation.json`. Source and manifest hashes are recorded.

Each model retains its previous comparison settings: PCA's independent panel
fits, 504-change history, AR dynamics, blocks and FX log transformation;
Gaussian's joint native-level covariance and trailing-300-row history. Both
retain full drift. Do not combine any earlier tentative experimental winners.

Rescale the same scenario deviations about each model's deterministic
conditional location, using one ratio per asset shared across all horizons.
For PCA, reconstruct its zero-innovation conditional factor path from the
fitted loadings, slopes, factor means and last factor state. Scale rates in
level space and FX in log space. This preserves PCA's conditional log-space
location; its arithmetic FX expectation can naturally change with log variance.
For Gaussian, preserve the analytic level-space mean. Correlations and random
shocks in the respective modeling spaces are unchanged. Current-rule draws
are returned directly, avoiding logarithm round-trip changes.

Recent variance uses 60 changes, centered at their own mean, with sample
denominator. Exponential variance uses all changes in the original window,
normalized exponential weights, a weighted mean, and the unbiased denominator
1 minus the sum of squared weights. Current PCA variance uses its 504 changes;
Gaussian uses the 299 changes from its 300 rows. Gaps in the original inputs
are already aligned by the historical case builder; this suite does not
impute missing observations. Positive current variance is required.

Scoring reuses fixed full-drift Gaussian/M0-like expected-error scales,
official installed component functions and clipping at 8. The score denominator
does not change with a volatility rule. See `gaussian_comparison.md` for
formula and evaluation-version limitations. These are local research scores.

Select one global rule for each model by the training mean only. Evaluate that
selection on validation; report all arms for transparency. No shape-specific
routing is selected. The 2,000 exploratory bootstrap resamples cluster all
cases and seeds at each origin while preserving case weighting. Intervals
do not fully account for serial regime dependence or multiple comparisons.
Repeated work on the same validation period is exploratory; only the still
untouched holdout can provide a fresh final comparison after candidate freeze.
Production settings and the Docker submission remain unchanged by this test.
