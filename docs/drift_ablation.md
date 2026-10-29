# Historical drift shrinkage experiment

This experiment changes only the historical sample-mean contribution to each
model's forecast. The predeclared fractions are zero, one half, and full drift.
Use the same 917 training/validation cases, root seeds 2026/17/83, 1,000 draws,
source hashes, normalized scoring and case manifest as the Gaussian comparison.
No holdout cases are scored. The production implementation and submission
configuration are unchanged.

```powershell
python -m experiments.drift_ablation
```

Results go to `reports/drift_ablation.json`, an ignored research artifact.
The CLI has no option to score holdout. It stops on failures or changed source
hashes rather than silently dropping cases. It records per-seed scores and
components, the manifest hash, summaries, and training-selected settings.

## What changes

Gaussian: subtract `(1 - fraction) * horizon * trailing_mean_change` from
every full-drift draw. Keep the 300-row window, covariance, diagonal jitter,
native-level FX representation, and simulated shocks fixed. Zero drift centers
the theoretical Gaussian forecast at the last observed level.

PCA: subtract `(1 - fraction) * horizon * mean_of_last_504_changes` in rates
space, and the corresponding mean log change in FX log space. All PCA/AR
parameters, innovations, initial factor state, five-day blocks, panel separation
and root seeds remain fixed. This removes the sample-mean term, not the
conditional factor dynamics; zero PCA drift does not imply an exactly flat
conditional forecast. FX log-space shifts preserve relative dispersion; the
absolute level-space standard deviation scales with the multiplicative shift.

Each fraction uses the same full-drift scenario realization. Do not regenerate
innovations between fractions. Full drift is exactly the preceding experiment's
control. Regression tests check production control parity, additive rates
shifts, multiplicative FX shifts, and clustered summary weights.

## Scoring and selection

Normalization remains fixed to the full-drift Gaussian/M0-like distribution
for every variant. It must not change with the setting being tested. Scoring
uses official installed components, locally reconstructed expected-error
divisors and clipping to 8; see `gaussian_comparison.md` for version and formula
details. Scores remain local research measurements, not official Final scores.

Choose one global fraction per model by the lowest training mean, with no
per-asset or per-shape routing. Validation scores test that frozen selection.
All three settings are reported for transparency; picking a different winner
after inspecting validation would require another independent test.

Summaries average seeds within cases and retain equal case weighting.
The paired delta is variant minus full drift: negative favors shrinkage.
The 2,000 exploratory bootstrap resamples keep all cases and seeds at a given
origin together. Confidence intervals do not account for all temporal regime
dependence or multiple comparisons. Training windows overlap even though
future target windows at different origins do not. No historical text is used.

## Results (2026-10-16)

Completed all 917 cases, three seeds, two models and three drift fractions:
16,506 evaluations. The 691 training cases span 147 origins; the 226 validation
cases span 50 origins. Six regression checks passed. All 5,502 full-drift
scores matched the preceding comparison exactly (maximum absolute difference
zero). No holdout rows were evaluated.

| Model | Training zero | Training half | Training full | Training-selected |
|---|---:|---:|---:|---|
| PCA | 0.98839 | 0.99293 | 1.00115 | Zero |
| Gaussian | 0.99169 | 0.99615 | 1.00736 | Zero |

Lower scores are better. The training-selected zero-drift setting failed to
improve validation for either model:

| Model | Validation zero | Validation half | Validation full |
|---|---:|---:|---:|
| PCA | 1.05191 | 1.04977 | 1.05119 |
| Gaussian | 1.04692 | 1.04162 | 1.04283 |

| Model / fraction | Validation delta vs full | Exploratory 95% origin interval |
|---|---:|---|
| PCA zero | +0.00071 | [-0.01353, +0.01507] |
| PCA half | -0.00142 | [-0.00865, +0.00572] |
| Gaussian zero | +0.00409 | [-0.01897, +0.03119] |
| Gaussian half | -0.00120 | [-0.01308, +0.01264] |

Every validation interval, including each individual shape, spans zero.
Half drift's approximately 0.1% improvement is too small and uncertain to
support adoption. It was not the training-selected winner, so choosing it
after inspecting validation would be a new selection rather than confirmation.
There is also no basis for per-shape routing: the observed improvements and
regressions differ across rates, FX and mixed cases.

Decision: retain the frozen production settings. Keep shrinkage as a research
option, with no evidence yet that it helps on unseen dates. Next evaluate
joint rates/FX covariance with representation, window, drift and random shocks
controlled, so that the effect of cross-panel dependence is isolated.
