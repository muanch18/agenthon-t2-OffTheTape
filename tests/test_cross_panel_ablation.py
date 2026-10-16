"""Check covariance controls, synchronized dates, and unchanged PCA defaults."""

from dataclasses import replace
import unittest

import numpy as np

from experiments.cross_panel_ablation import paired_draws
from experiments.gaussian_comparison import pca_draws
from off_the_tape.gaussian_joint import GaussianJointForecaster
from off_the_tape.pca_joint import PCAJointForecaster
import test_gaussian_comparison as fixtures


class CrossPanelTest(unittest.TestCase):
    def test_controls_and_fixed_rates(self):
        card = fixtures.GaussianComparisonTest().card()
        draws, _ = paired_draws(card, 17)
        np.testing.assert_array_equal(draws[("pca", 0)], pca_draws(card, 17))
        np.testing.assert_array_equal(draws[("gaussian", 1)], GaussianJointForecaster(seed=17)(card))
        for model in ("pca", "gaussian"):
            for weight in (0.5, 1):
                np.testing.assert_array_equal(draws[(model, weight)][:, :2], draws[(model, 0)][:, :2])

    def test_shared_dates_recover_known_cross_panel_shock(self):
        card = fixtures.GaussianComparisonTest().card()
        frame = card.panel.copy()
        rates = frame.loc[frame.asset == "UST_2Y"].value.to_numpy()
        frame.loc[frame.asset == "EUR", "value"] = np.exp(0.1*rates)
        card = replace(card, panel=frame)
        draws, _ = paired_draws(card, 83)
        for model in ("pca", "gaussian"):
            independent = draws[(model, 0)]
            joint = draws[(model, 1)]
            self.assertLess(abs(np.corrcoef(independent[:, 0], independent[:, 2])[0, 1]), 0.15)
            self.assertGreater(np.corrcoef(joint[:, 0], joint[:, 2])[0, 1], 0.97)

    def test_supplied_schedule_matches_default_and_rejects_bad_indices(self):
        card = fixtures.GaussianComparisonTest().card()
        card = replace(card, assets=("UST_2Y",), panel=card.panel.loc[card.panel.asset == "UST_2Y"])
        model = PCAJointForecaster(mode="difference", seed=83)
        starts = np.random.default_rng(83+card.asof.toordinal()).integers(0, 499, size=(1000, 2))
        np.testing.assert_array_equal(model.forecast(card).draws, model.forecast(card, block_starts=starts).draws)
        with self.assertRaisesRegex(ValueError, "invalid historical"):
            model.forecast(card, block_starts=np.full((1000, 2), 9999))
        with self.assertRaisesRegex(ValueError, "invalid historical"):
            model.forecast(card, block_starts=np.zeros((1000, 2)))


if __name__ == "__main__":
    unittest.main()
