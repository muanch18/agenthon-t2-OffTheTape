"""Verify paired drift changes and unchanged full-drift production control."""

import unittest
import numpy as np

from experiments.drift_ablation import variants, summarize
from experiments.gaussian_comparison import pca_draws
from off_the_tape.gaussian_joint import GaussianJointForecaster
from test_gaussian_comparison import GaussianComparisonTest


class DriftAblationTest(unittest.TestCase):
    def test_common_shocks_and_full_control(self):
        card = GaussianComparisonTest().card()
        draws, scales = variants(card, 17)
        np.testing.assert_array_equal(draws[("pca", 1)], pca_draws(card, 17))
        np.testing.assert_array_equal(draws[("gaussian", 1)], GaussianJointForecaster(seed=17)(card))
        self.assertTrue(all(np.isfinite(v) and v > 0 for v in scales.values()))
        for model in ("gaussian", "pca"):
            full, half, zero = [draws[(model, f)] for f in (1, 0.5, 0)]
            # Rates are additive shifts, retaining identical dispersion.
            np.testing.assert_allclose(full[:, :2]-half[:, :2], half[:, :2]-zero[:, :2], atol=1e-14)
            np.testing.assert_allclose(full[:, :2].std(axis=0), zero[:, :2].std(axis=0), atol=1e-14)
            if model == "pca":
                # FX shifts are additive in log space, retaining relative dispersion.
                np.testing.assert_allclose(np.log(full[:, 2:])-np.log(half[:, 2:]),
                                           np.log(half[:, 2:])-np.log(zero[:, 2:]), atol=1e-14)

    def test_overall_clusters_keep_case_weights(self):
        rows = [{"split": "train", "model": "pca", "shape": shape,
                 "fraction": 0.5, "asof": day, "score": score, "full_score": 1.0}
                for shape, day, score in (("mixed", "2006-01-01", 2.0),
                                           ("fx_single", "2006-01-01", 2.0),
                                           ("mixed", "2006-02-01", 0.0))]
        result = next(s for s in summarize(rows) if s["shape"] == "all")
        self.assertEqual(result["origins"], 2)
        self.assertAlmostEqual(result["delta_vs_full"], 1/3)


if __name__ == "__main__":
    unittest.main()
