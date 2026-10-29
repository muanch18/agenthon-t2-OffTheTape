# Two frozen research candidates

These candidate definitions were frozen before the 2022–2024 historical
holdout was scored. They are research variants, distinct from the default
numeric PCA runtime and the archived competition image. No new tuning is
introduced by this presentation.

| Choice | PCA adaptive-volatility candidate | Gaussian recent-volatility candidate |
|---|---|---|
| Base model | Two PCA factors, clipped AR(1), paired five-day empirical innovation/residual blocks | Multivariate Gaussian random walk with coherent cross-horizon covariance |
| History | 504 daily changes | 300 level rows, yielding up to 299 changes |
| Yield representation | Level differences | Level differences |
| FX representation | Log changes; exponential level reconstruction | Native-level differences |
| Rates/FX dependence | Separate per-panel models | Joint covariance across requested assets |
| Drift | Full historical mean plus conditional factor dynamics | Full historical mean |
| Volatility adjustment | 50/50 historical and EWMA variance; 20-observation half-life | Trailing 60-change sample standard deviation |
| Scaling safeguard | Candidate/base sd ratio capped to [0.5, 2] | Same cap |
| Location preservation | Rescale around deterministic conditional location in rates/log-FX space | Rescale around analytic level-space mean |
| Draws and root seeds | 1,000; 2026, 17, 83 | Same |
| Text conditioning | Disabled | Disabled |

## Why keep Gaussian?

Gaussian is a credible simpler alternative because it needs historical moments
and a covariance factorization rather than PCA factor fitting, AR dynamics and
an empirical block-shock simulation. Its assumptions are explicit and its
cross-horizon moments are analytic. Simplicity here describes model structure;
it does not guarantee accuracy, calibration or cheap execution in every setting.
Level-space FX can produce nonpositive draws in principle and does not impose
multiplicative dynamics. This benchmark covers daily level targets, not every
official M0 mode or monthly/return target.

Its adjusted historical holdout loss was 1.278862 versus PCA's 1.305176.
The exploratory direct paired comparison gives a 2.02% mean advantage, but
the 95% difference interval includes zero. It is competitive, not a proven
winner. The original cross-model comparison also bundles different windows,
FX representations and panel-dependence choices.

## What supports the PCA adjustment?

The predeclared test compared adjusted PCA with unchanged PCA. Loss fell from
1.357877 to 1.305176 (3.88%), with adjusted 97.5% paired interval
[-0.124580, -0.001389]. It passed that study's criterion, narrowly. Gaussian's
adjustment improved its own control by 4.03%, but its interval crossed zero.
These two conclusions do not rank the models against each other.

PCA's gain was concentrated in 2022 and slightly reversed in 2024. The study
has only 33 origins and is not evidence of live-market performance. Keep this
scope alongside any headline percentage.

## Reproduction and subsequent work

The original candidate implementation and confirmation are in
`experiments/frozen_holdout.py`; the freeze commit is `9113289`. Detailed
results were recorded afterward without retuning. Reproduce at the documented
historical version and environment: main contains later feature refactors and
must not be made compatible by rewriting old hashes.

See [holdout findings](frozen_holdout_results.md),
[direct comparison](direct_candidate_comparison.md), and
[shared-input comparison protocol](controlled_architecture_protocol.md).
The [completed controlled study](controlled_architecture_results.md) uses
different, shared settings. Its arms must not
be called these original frozen candidates, and its scores cannot be inserted
into the original holdout table.
