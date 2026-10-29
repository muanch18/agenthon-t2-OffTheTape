# Shared-input comparison: results

The protocol, implementation and input/environment hashes were committed and
pushed as `58ae81b` before new scores were run. This is an exploratory study
on previously examined validation dates, not a fresh confirmation study.
All 226 cases across 50 origins completed at three seeds: 678 paired score
records, 1,356 scored arm evaluations, with no dropped cases or model failures.
Additional warm calls measured runtime. No holdout cases were scored.
All 678 PCA scores matched the earlier base-PCA validation scores exactly.
The merged suite passed 73 tests, including three new shared-input checks.
Result/protocol hashes, unique record counts and reported means were audited.

## Shared settings and remaining differences

Both arms use the same 504 observed changes, feature snapshots, full sample
drift, historical volatility setting, transformed-space representation
(yield differences / FX log changes), independent rates/FX panels, 1,000 draws
and three seeds. No adaptive volatility is applied. The Gaussian arm has
therefore changed from the original frozen native-level, 300-row comparator.
These scores are not replacements for the original candidate holdout table.

PCA retains AR(1) dynamics and empirical five-day block shocks; Gaussian uses
independent daily multivariate normal shocks. Conditional forecast locations
and variances can differ as a consequence of those dynamics. This controls
input choices and compares the remaining model packages, not PCA alone.

## Accuracy

| Measure | Result |
|---|---:|
| Shared-input Gaussian mean loss | 1.053374 |
| Shared-input PCA mean loss | 1.051194 |
| PCA minus Gaussian | -0.002179 |
| PCA relative loss reduction | 0.207% |
| Exploratory paired 95% interval | [-0.015319, +0.009936] |

PCA has a small mean advantage at all three seeds, but the interval includes
zero. This does not establish superiority or equivalence. Both use the earlier
fixed pre-outcome normalization, installed component scorers and clipping at 8.
Inference uses 10,000 moving-block replicates of four ordered origins, grouping
shapes/seeds and preserving case weights. Diagnostic shape intervals below
are exploratory and unadjusted.

| Shape | Gaussian | PCA | PCA minus Gaussian | Exploratory 95% interval |
|---|---:|---:|---:|---|
| Single rates | 0.975056 | 0.975632 | +0.000576 | [-0.015013, +0.015411] |
| Rates curve | 1.087448 | 1.081270 | -0.006178 | [-0.037392, +0.022633] |
| Single FX | 1.108157 | 1.105875 | -0.002281 | [-0.032506, +0.016117] |
| FX basket | 1.028529 | 1.026953 | -0.001576 | [-0.025004, +0.016186] |
| Mixed rates/FX | 1.068860 | 1.067349 | -0.001512 | [-0.010821, +0.012462] |

All shape intervals cross zero. Do not choose routing from these breakdowns.

## Runtime and complexity

| Warm API latency | Gaussian | PCA |
|---|---:|---:|
| Median across per-case medians | 1.316 ms | 5.642 ms |
| P90 across per-case medians | 2.307 ms | 9.734 ms |

PCA's median is approximately 4.29 times Gaussian's in this implementation.
Calls were warmed before three repetitions at seed 2026; arm order alternated.
Timing excludes shared preparation, data loading, scoring and file I/O. PCA
includes diagnostic construction; Gaussian has fewer diagnostics. These are
implementation latencies, not an isolated factor-fitting benchmark or an
end-to-end service SLA. Both are small in absolute terms. The run used the
existing Linux/WSL2 verifier image, NumPy 2.1.3, Python 3.13.15 and one BLAS/OMP
thread. Hardware load and software choices limit portability.

| Model responsibility | Gaussian | PCA |
|---|---|---|
| Estimation | Sample mean/covariance and eigen square root | Standardization, SVD, factor AR fits and empirical innovation pool |
| Simulation | Independent daily Gaussian shocks with within-panel covariance | Sequential factor states with paired block shocks and residuals |
| Horizon handling | Read from one cumulative simulated path | Same |
| Distribution assumption | Gaussian transformed daily increments | Empirical shocks plus factor dynamics |
| Additional model choices | Covariance estimation | Factor count, slope bounds and block length |

The Gaussian arm is structurally simpler. Neither model learns neural weights;
both estimate their parameters using only pre-cutoff data.

## Decision

The controlled inputs leave little observed average accuracy difference on
this validation roster. Gaussian is a credible simple reference implementation
for future work. There is no demonstrated accuracy gain here that justifies
choosing PCA on performance grounds alone, and no proof the models are equal.
PCA remains useful for studying factor dynamics and empirical shock structure.
No runtime candidate, Docker image or original frozen candidate was changed.

This does not invalidate PCA's earlier adaptive-volatility improvement against
its own control. That is a different comparison on a different split. Nor
does it identify which earlier preprocessing difference explained the original
cross-model ranking. Further component attribution or confirmation would need
a separately fixed study and independent data for a generalization claim.

## Audit and reproduction

- [Frozen protocol and run instructions](controlled_architecture_protocol.md)
- [Machine-readable freeze](../experiments/frozen/controlled_architecture_v1.json)
- [Tracked results summary](../experiments/results/controlled_architecture_v1.json)
- [Original frozen candidate profiles](frozen_candidates.md)

The detailed local result is `reports/controlled_architecture_v1.json`, SHA-256
`1558299da568ebca2975a32233d3750d4d77da3df1e1c6f924dc7f0b0daad337`.
It contains per-case components and per-case median timings; large reports stay
out of Git. Reproduction needs the recorded manifest/data and pinned environment.
The runner refuses an existing output rather than silently replacing evidence.
