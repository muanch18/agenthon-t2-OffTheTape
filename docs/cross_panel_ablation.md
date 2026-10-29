# Cross-panel dependence experiment

Compare independent, half-strength and full-strength rates/FX dependence on
the fixed mixed historical cases: 133 training and 43 validation origins.
All other case shapes are unaffected by this change and are not rerun.
Holdout cases are never scored. Production configuration stays unchanged.

```powershell
python -m experiments.cross_panel_ablation
```

Output: `reports/cross_panel_ablation.json`. Source and manifest hashes bind
inputs to the prior experiments. Root seeds are fixed in advance at
2026, 17, 83, 9 and 123; each forecast has 1,000 draws. Select one dependence
strength per model on training only, then assess that choice on validation.

## Controlled alternatives

Gaussian: keep full historical drift, trailing-300 native-level moments,
all variances and within-panel covariance, horizon relationships, diagonal
jitter, and base standard-normal draws fixed. Multiply rates/FX off-block
covariance by 0, 0.5 or 1 before Cholesky sampling. Means and theoretical
marginal distributions are identical. Actual FX scenarios change when the
covariance transform changes, so finite-sample marginal scores can differ.
The full-strength control exactly matches the previous Gaussian experiment.

PCA: keep independent per-panel fits, 504-change lookback, AR dynamics,
five-day blocks, full drift, FX log representation and historical innovation
pools fixed. At strength zero, sample with the production panel-specific seeds.
At strength one, FX uses the rates panel's block-start schedule, so factor
shocks and residuals from matching historical dates are paired across panels.
Half strength uses a preseeded independent path-wise switch: half the paths
in expectation share dates; the remainder keep independent schedules. Whole
paths and all horizons move together. Rates draws are identical in every arm.
The strength-zero control exactly matches production PCA draws.

This PCA approach does not refit joint factors or add cross-panel AR dynamics.
It tests common historical shocks and regime dates. All historical dates must
match between panels; mismatches stop the experiment. Historical alignment
is based on pre-cutoff inputs. The optional `block_starts` argument in the
PCA forecast API is used only for research; its default retains the prior RNG
behavior. The runner verifies schedule bounds and integer types.

Sampling shared historical blocks preserves each panel's theoretical
distribution but changes the finite FX ensemble. Thus an observed composite
change is not purely a change in the variogram term. Component scores and
five-seed sensitivity are retained to distinguish dependence effects from
sampling noise. A positive result would still need follow-up validation.

## Scoring and interpretation

Reuse official installed component functions and the same full-covariance
Gaussian expected-error scales as earlier experiments. Divisors do not change
with candidate dependence. Compare paired normalized composite differences
against independence, lower is better. Keep root seeds together within each
origin for 2,000 exploratory bootstrap resamples; seed variation does not add
independent market observations. Intervals do not fully capture temporal regime
dependence or multiple comparisons. See `gaussian_comparison.md` for scoring
versions and limitations. No historical text or actual F1–F4 semantics are
evaluated. No Final performance claim follows from these research cases.

## Results (2026-10-16)

Completed 176 mixed cases, five seeds, two models and three dependence strengths:
5,280 evaluations. Both models selected independence on training. Three new
cross-panel tests and four existing PCA tests passed. All 1,056 historical
control scores matching the preceding experiment's three seeds were exactly
equal (maximum difference zero). No holdout cases were evaluated.

| Model | Training independent | Training half | Training full |
|---|---:|---:|---:|
| PCA | 1.015705 | 1.017239 | 1.015942 |
| Gaussian | 1.033257 | 1.033527 | 1.033277 |

Validation results (43 origins):

| Model | Independent | Half | Full |
|---|---:|---:|---:|
| PCA | 1.067671 | 1.067811 | 1.067799 |
| Gaussian | 1.052152 | 1.052439 | 1.052236 |

| Model / strength | Validation delta vs independent | Exploratory 95% origin interval |
|---|---:|---|
| PCA half | +0.000140 | [-0.002202, +0.002325] |
| PCA full | +0.000127 | [-0.003670, +0.003572] |
| Gaussian half | +0.000286 | [-0.001057, +0.001565] |
| Gaussian full | +0.000084 | [-0.001443, +0.001613] |

All intervals span zero, and all nonzero variants' validation deltas change
sign across seeds. The measured effects are smaller than typical seed effects.
The component audit found no average improvement in the unbounded normalized
joint contribution either. Five validation case/seed evaluations per variant
were clipped at 8, so unbounded component deltas do not sum to the clipped
composite delta. `reports/cross_panel_diagnostics.json` records component
contributions and the clipping adjustment explicitly.

Decision: do not enable cross-panel dependence in production on this evidence.
This rejects these two simple dependence adjustments on this narrow numeric
suite; it does not establish that rates/FX relationships are universally
unhelpful. The test covers only the manifest's fixed mixed basket, short
horizons and historical regimes, with no text-informed conditioning.

The next original priority is recent-volatility versus longer-window
volatility. That experiment should keep dependence and drift fixed and select
any volatility rule on training only.
