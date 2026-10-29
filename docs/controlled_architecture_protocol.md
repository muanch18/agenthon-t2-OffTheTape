# Shared-input model comparison: frozen exploratory protocol

This study is defined before running its new scores. It uses the existing 226
validation cases from 2018–2021, already explored in prior research. It is not
fresh confirmation. Neither training nor the consumed holdout is scored.

## Fixed arms

Both arms use the same feature snapshots, aligned observations, 504 changes,
full historical sample-mean drift, historical sample volatility, root seeds
2026/17/83, 1,000 draws, asset order and horizon grid. Yields use differences;
FX uses log changes and exponential reconstruction. Rates and FX are separate
panels in both arms. No adaptive volatility, text, threshold dynamics, routing,
or parameter search is included.

Gaussian draws independent daily multivariate normal shocks using the full
sample covariance in each panel. A symmetric eigen square root handles positive
semidefinite covariance without additional diagonal jitter. PCA retains two
factors, clipped AR(1) dynamics and five-day paired empirical innovation blocks.
Identical seed labels improve reproducibility but do not imply matched shocks
across these different sampling algorithms.

The shared volatility setting means identical historical input sd and no
rescaling. AR dynamics and block dependence can change PCA's predictive
variance: equal realized forecast variance or equal conditional locations are
not imposed. This comparison isolates the remaining model package, not PCA
dimensionality reduction alone. Removing AR dynamics or changing shock laws
would be another study.

## Accuracy and uncertainty

Use unchanged full-native-level Gaussian/M0-like normalizers from the earlier
suite for both arms. They are pre-outcome and fixed, despite the Gaussian
comparison arm now using shared log-space FX preprocessing. Reuse installed
component scoring and composite clipping at 8. Report mean losses, PCA minus
Gaussian, relative change, seed sensitivity, and exploratory 95% paired
four-origin moving-block bootstrap intervals (10,000 replicates, seed 901).
Shape breakdowns are diagnostic, unadjusted and cannot select routing.
Average seeds within case; keep all shapes together by origin. No official
competition score or equivalence claim is intended.

## Runtime and complexity

Warm both model APIs on every case, then time three calls at seed 2026 with
alternating order. Report median per-case latency and its across-case median
and P90. Exclude loading, shared feature preparation, scoring and file I/O.
PCA's API includes diagnostics while the Gaussian implementation has fewer
diagnostics: this is measured implementation cost, not a pure algorithmic
speed claim. CPU, library and scheduler effects limit portability. Fix BLAS
threads to one for this run. No GPU is used.

Gaussian fits sample moments and a covariance square root. PCA adds SVD,
factor AR fits, an empirical shock pool and sequential simulation. Both learn
parameters from each pre-cutoff panel; neither trains neural weights. Neither
is automatically enabled or removed based on these results.

## Freeze and execution

The machine-readable protocol pins settings, environment, manifest/data hashes
and relevant source hashes. Generate with `python -m experiments.controlled_architecture --freeze`,
commit the protocol and code, and only then run
`python -m experiments.controlled_architecture`. The runner rejects hash or
environment changes, missing roster cases and existing result files. Exceptions
stop the run rather than silently skipping cases. Detailed output stays in
`reports/`; a compact audit summary is tracked under `experiments/results/`.
