# Band-SETAR dynamics experiment — 2026-10-17

This exploratory test changes only the conditional factor dynamics in the
original PCA/AR(1) model. It does not combine SETAR with the volatility blend,
change production settings, or score the consumed 2022–2024 holdout.

One fixed specification: threshold at the 80th percentile of absolute lagged
factor values in the pre-cutoff training window, two coefficient sets (middle
versus tails), each shrunk halfway toward pooled AR(1). Require at least 50
lag/target pairs in each regime and positive lag variation; otherwise use the
pooled coefficients for that factor. Clip each fitted slope to [-0.95, 0.95]
before shrinkage. This bound is a numerical safeguard, not a general proof
of stationary or well-calibrated forecast distributions.

This is a symmetric band threshold specification: negative and positive
extremes share coefficients. It is not the conventional signed single-cutoff
two-regime SETAR. The classifier is invariant to arbitrary PCA factor signs.
Each factor uses its own threshold and coefficients; each simulated path
re-evaluates its own regime at every step. Regimes are defined from daily
change factors, not from asset levels or economic crisis labels.

Retain the 504-change PCA fit, means, loadings, scales, initial factor state,
independent panel seeds, FX log representation and five-day innovation blocks.
Crucially, both arms use the original pooled AR(1) factor shocks and same-date
asset residuals, with exactly the same sampled block indices. Reconstruct and
check the original daily transforms exactly on every panel/case/seed before
scoring. This isolates conditional mean dynamics. It does not estimate a new
SETAR error law, switch shock variance, or condition the shock pool on regime.

Use only the fixed 691 training and 226 validation cases at root seeds
2026/17/83, 1,000 draws. No threshold or shrinkage search, extra origins,
per-shape routing, or holdout scoring is allowed. The runner has no holdout
option. Use unchanged expected-error normalization, official installed scoring
components and clipping at 8. Errors stop the experiment rather than removing
unfavorable cases.

```powershell
python -m experiments.setar_ablation
```

Output: `reports/setar_ablation.json`, including coefficients, regime counts,
simulated stress occupancy and maximum factor magnitudes. Source panel hashes,
manifest hash, experiment code hash and scorer versions are recorded.
Compare SETAR minus AR(1), so negative favors threshold dynamics. Report
case-weighted paired means, seed sensitivity and 10,000 four-origin moving-
block bootstrap 95% intervals, with all cases and seeds at an origin together.
Breakdowns by shape are diagnostic. Earlier training/validation periods have
already been explored; these intervals do not constitute fresh confirmation.

Inspect numerical validity and descriptive regime occupancy before proposing
any further extension. If promising, a separate experiment must address how
to combine nonlinear dynamics with the volatility blend without incorrectly
reusing its linear conditional-location calculation. Any confirmation requires
new independent/prospective evidence. Production and the Docker image remain
unchanged regardless of this exploratory run's point estimate.

## Results — 2026-10-17

Implementation and specification were committed and pushed as `c160c3f`
before running historical scores. Three behavioral tests passed. All 917
cases completed at three seeds: 2,751 paired records and 5,502 arm evaluations.
All 2,751 AR(1) control scores matched the earlier Gaussian comparison exactly;
all 3,279 panel/case/seed daily-innovation reconstruction checks passed. No
case was dropped, no factor needed the small-regime fallback on these cases,
and all simulated values were finite. No holdout cases were evaluated.

| Period | AR(1) | Band-SETAR | SETAR minus AR(1) | Relative loss increase |
|---|---:|---:|---:|---:|
| Training | 1.001153 | 1.001751 | +0.000598 | 0.060% |
| Validation | 1.051194 | 1.052604 | +0.001410 | 0.134% |

The exploratory validation block interval for SETAR minus AR(1) is
[+0.000165, +0.002805]. Every seed regressed on both training and validation.
Validation point estimates also regressed in all five shapes:

| Shape | SETAR loss increase vs AR(1) |
|---|---:|
| Single rates | 0.403% |
| Rates curve | 0.117% |
| Single FX | 0.065% |
| FX basket | 0.064% |
| Mixed rates/FX | 0.031% |

Only the single-rates shape's exploratory interval excludes zero; the other
shape intervals span zero. None of these reused development-period intervals
is independent confirmation or a general finding against threshold models.
This specific conservative magnitude-threshold mean adjustment did not help.
Its fixed pooled AR(1) innovation law is a deliberate limitation of this
ablation, so the result does not evaluate a fully re-estimated SETAR error law.

Decision: do not adopt this variant or add it to the volatility candidate.
Do not tune thresholds/shrinkage against the consumed holdout. The previously
frozen PCA volatility blend remains the supported candidate for a separate
production integration task. The production code and Docker image are unchanged.

Tracked summary and audit: `experiments/results/setar_20261017.json`, including
the manifest and experiment hashes, scorer versions, per-shape summaries,
seed sensitivity and the hash of `reports/setar_ablation.json`. Detailed local
per-case scores and coefficients remain ignored under `reports/`.
