# Direct comparison of frozen candidates

This is an exploratory analysis of the already inspected 2022–2024 holdout,
performed on 2026-10-29. It is not an additional predeclared confirmation test,
an official competition result, or evidence specifically isolating PCA's value.
No model is rerun, retuned, integrated into production, or submitted.

Compare PCA with its frozen 50/50 current/EWMA variance blend against Gaussian
with its frozen recent-60 volatility. Both retain their original, different
windows, FX representations, dynamics and panel dependence choices.

## Method and reproducibility

```powershell
python -m experiments.direct_candidate_comparison
python -m unittest tests.test_direct_candidate_comparison
```

The script checks the detailed saved result against its tracked SHA-256
(`16a85928b1597ed539f6ea84b75105ccd955841f8c9f83c7b397793a3bb1b0b9`),
checks the protocol hash and complete 156-case/33-origin/three-seed roster,
and rejects duplicates or nonfinite scores. Detailed scores must already be
available at `reports/frozen_holdout_20261016.json`; they are ignored by Git.
The tracked original audit is `experiments/frozen/holdout_20261016_results.json`.
Output is `reports/direct_candidate_comparison.json`.

Seeds are averaged within case. Each case has equal weight. We draw 10,000
non-wrapping moving-block bootstrap replicates of four ordered origins with
seed 901, keeping every case at an origin together. Negative differences mean
Gaussian has lower loss. The 95% interval is exploratory and unadjusted;
97.5% intervals are also reported for reference, without treating this as part
of the original two-test family. Subgroup intervals are diagnostic, unadjusted,
and do not justify selecting market-specific routing.

## Results

| Measure | Result |
|---|---:|
| PCA mean normalized composite loss | 1.305176 |
| Gaussian mean normalized composite loss | 1.278862 |
| Gaussian minus PCA | -0.026314 |
| Gaussian relative loss reduction | 2.02% |
| Exploratory 95% interval | [-0.077525, +0.013264] |
| Exploratory 97.5% interval | [-0.084377, +0.019065] |
| Gaussian wins, seed-averaged cases | 76/156 (48.72%) |
| Gaussian wins, origin-average loss | 17/33 (51.52%) |

Gaussian's average loss is lower at every seed, but the interval crosses zero.
There is no clear overall winner on this evidence. Wins are close to evenly
split; larger Gaussian wins drive the average advantage. All sensitivity
intervals using block lengths 2, 6 and 8 also cross zero.

| Period / shape | Gaussian loss reduction relative to PCA |
|---|---:|
| 2022 | 4.61% |
| 2023 | 2.59% |
| 2024 available cases | -3.73% |
| Single rates | 1.33% |
| Rates curve | 6.00% |
| Single FX | -1.33% |
| FX basket | -0.47% |
| Mixed rates/FX | 2.49% |

Negative relative reductions favor PCA. Each year's 95% interval crosses zero.
Rates curves are the sole shape with a 95% interval excluding zero
([-0.202854, -0.003708]); its 97.5% interval crosses zero. Given exploratory
subgroup inspection, this is a follow-up hypothesis, not a confirmed exception.

## Interpretation

The original finding remains intact: PCA's volatility adjustment passed its
predeclared comparison against unchanged PCA. That does not imply PCA beats
Gaussian, and this direct analysis does not establish Gaussian superiority.
The results do not establish equivalence either. Only 33 origins are available;
persistent regimes and overlapping estimation histories limit inference.
These local scores lack historical text and are not live-market validation.
The holdout remains consumed. A fair architecture ablation or a new independent
comparison should fix its settings before inspecting new results.
