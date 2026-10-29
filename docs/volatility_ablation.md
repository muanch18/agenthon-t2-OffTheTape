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

## Results (2026-10-16)

Completed 917 cases, three seeds, two models and four rules: 22,008 evaluations.
Three new tests passed, covering exact controls, fixed transformed-space
locations/correlations, positive FX levels, and response to a synthetic
pre-cutoff volatility regime. All 5,502 current-rule scores exactly matched
the preceding PCA/Gaussian experiment (maximum difference zero). No holdout
cases were scored.

| Model | Rule | Training | Validation |
|---|---|---:|---:|
| PCA | Current | 1.001153 | 1.051194 |
| PCA | Recent 60 | 0.975833 | 1.035643 |
| PCA | 50/50 current/recent variance | 0.977196 | 1.034442 |
| PCA | 50/50 current/EWMA variance | **0.975669** | 1.040901 |
| Gaussian | Current | 1.007362 | 1.042826 |
| Gaussian | Recent 60 | **0.983055** | 1.025795 |
| Gaussian | 50/50 current/recent variance | 0.987249 | 1.029814 |
| Gaussian | 50/50 current/EWMA variance | 0.985513 | 1.034723 |

Bold identifies the training-selected rule, not a choice made on validation.
PCA selected the current/EWMA blend; Gaussian selected recent-60 volatility.

| Training-selected candidate | Validation change vs current | Relative improvement | Exploratory 95% origin interval |
|---|---:|---:|---|
| PCA current/EWMA blend | -0.010293 | 0.98% | [-0.026753, +0.007687] |
| Gaussian recent 60 | -0.017030 | 1.63% | [-0.042718, +0.008980] |

Both selected rules improve validation at all three seeds. Gaussian recent-60
improves the point estimate in all five shapes; PCA's selected blend improves
four shapes and slightly worsens single rates (+0.003250). Every selected
per-shape validation interval still includes zero. All three noncurrent rules
improve the overall validation point estimate for both models, but all overall
validation intervals also include zero.

The PCA current/recent blend happens to look better on validation than the
training-selected EWMA blend. Switching to it now would be an additional
selection on validation. Do not treat that alternative as independently
confirmed or change the rule to maximize the displayed validation result.

Decision: retain the frozen production submission for now, but keep these
two training-selected volatility rules as the strongest candidates from this
research sequence. Unlike drift and cross-panel dependence, the improvement
is larger in score terms and consistent across seeds, although
generalization remains unconfirmed. Freeze candidate definitions and the
holdout evaluation protocol before inspecting the untouched 2022–2024 cases.
Do not retune after reading holdout results.

Implementation and tests were pushed in commit `60dcd8c` on
`codex/final-validation-experiments`. This findings record is committed
separately. Detailed generated JSON stays local under the existing `reports/`
ignore rule; code, methods, tests and these findings are tracked in Git.

## Subsequent frozen confirmation

The two training-selected rules were later frozen in commit `9113289` and
evaluated on the previously untouched 2022–2024 roster. See
`frozen_holdout_results.md`. PCA's blend passed the predeclared criterion;
Gaussian's recent-60 improvement was not confirmed by its adjusted interval.
That holdout is now consumed. No volatility settings were retuned afterward.
