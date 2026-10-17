"""Behavioral tests for band switching, shrinkage, sign invariance and fixed shocks."""

import unittest

import numpy as np

from experiments.setar_ablation import advance, fit_band, paired_draws
from experiments.gaussian_comparison import pca_draws
import test_gaussian_comparison as fixtures


class SetarTest(unittest.TestCase):
    def test_path_can_switch_and_negative_tail_is_stressed(self):
        band = {"threshold": np.array([1.0]), "intercept": np.zeros((2, 1)), "slope": np.array([[0.5], [-0.5]])}
        state = np.array([[0.1]])
        regimes, values = [], []
        for shock in (2, 0, 0, 0):
            state, regime = advance(state, np.array([[shock]]), band)
            regimes.append(bool(regime[0, 0]))
            values.append(float(state[0, 0]))
        self.assertEqual(regimes, [False, True, True, False])
        np.testing.assert_allclose(values, [2.05, -1.025, 0.5125, 0.25625])

    def test_sign_invariance_and_pooled_fallback(self):
        factors = np.random.default_rng(3).normal(size=(504, 2))
        a, b = np.array([0.02, -0.03]), np.array([0.2, -0.1])
        positive = fit_band(factors, a, b)
        negative = fit_band(-factors, -a, b)
        np.testing.assert_array_equal(positive["counts"], negative["counts"])
        np.testing.assert_allclose(positive["slope"], negative["slope"])
        np.testing.assert_allclose(positive["intercept"], -negative["intercept"])
        fallback = fit_band(factors, a, b, min_regime=150)
        self.assertEqual(fallback["fallback_factors"], [0, 1])
        np.testing.assert_array_equal(fallback["slope"], np.tile(b, (2, 1)))
        pooled = fit_band(factors, a, b, weight=0)
        np.testing.assert_array_equal(pooled["intercept"], np.tile(a, (2, 1)))

    def test_threshold_control_matches_production_and_fixed_innovations(self):
        card = fixtures.GaussianComparisonTest().card()
        draws, diagnostic = paired_draws(card, 17)
        np.testing.assert_array_equal(draws["ar1"], pca_draws(card, 17))
        self.assertTrue(np.isfinite(draws["setar"]).all())
        self.assertTrue((draws["setar"][:, 2:] > 0).all())
        self.assertFalse(np.array_equal(draws["setar"], draws["ar1"]))
        self.assertTrue(all(d["control_innovations_exact"] for d in diagnostic.values()))


if __name__ == "__main__":
    unittest.main()
