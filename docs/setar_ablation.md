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
