"""Validate volatility-only changes and unchanged forecast controls."""

from dataclasses import replace
import unittest

import numpy as np

from experiments.volatility_ablation import variants, scale_ratios, rescale
from experiments.gaussian_comparison import pca_draws
from off_the_tape.gaussian_joint import GaussianJointForecaster
import test_gaussian_comparison as fixtures


class VolatilityTest(unittest.TestCase):
    def test_controls_exact_and_fx_positive(self):
        card = fixtures.GaussianComparisonTest().card()
        draws, _, ratios = variants(card, 17)
        np.testing.assert_array_equal(draws[("pca", "current")], pca_draws(card, 17))
        np.testing.assert_array_equal(draws[("gaussian", "current")], GaussianJointForecaster(seed=17)(card))
        for rule in ("recent60", "blend60", "ewma20_blend"):
            self.assertTrue((draws[("pca", rule)][:, 2:] > 0).all())
            self.assertTrue(((ratios["pca"][rule] >= 0.5) & (ratios["pca"][rule] <= 2)).all())

    def test_rescaling_preserves_location_and_transformed_correlations(self):
        rng = np.random.default_rng(4)
        noise = rng.normal(size=(10000, 4))
        noise = np.concatenate((noise, -noise))
        location = np.array([2.0, 3.0, 0.1, 0.2])
        original = location+noise*0.01
        original[:, 2:] = np.exp(original[:, 2:])
        ratio = np.array([0.7, 1.2, 0.8, 1.5])
        scaled = rescale(original, location, ratio, np.array([False, False, True, True]))
        scaled[:, 2:] = np.log(scaled[:, 2:])
        base = location+noise*0.01
        np.testing.assert_allclose(scaled.mean(axis=0), location, atol=1e-13)
        np.testing.assert_allclose(scaled.std(axis=0)/base.std(axis=0), ratio, atol=1e-12)
        np.testing.assert_allclose(np.corrcoef(scaled, rowvar=False), np.corrcoef(base, rowvar=False), atol=1e-12)

    def test_recent_volatility_responds_to_pre_cutoff_regime(self):
        card = fixtures.GaussianComparisonTest().card()
        frame = card.panel.copy()
        rng = np.random.default_rng(3)
        for asset in card.assets:
            mask = frame.asset == asset
            n = mask.sum()
            noise = rng.normal(size=n)*0.001
            noise[-60:] *= 8
            frame.loc[mask, "value"] = 2+noise.cumsum()
        card = replace(card, panel=frame)
        for model in ("pca", "gaussian"):
            ratios = scale_ratios(card, model)
            self.assertTrue((ratios["recent60"] > 1.5).all())
            self.assertTrue((ratios["blend60"] > 1).all())


if __name__ == "__main__":
    unittest.main()
