"""Check Gaussian path moments, expected-score scales and PCA production parity."""

from dataclasses import replace
from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.special import gamma

from experiments.gaussian_comparison import expected_scales, pca_draws, summary
from off_the_tape.agent import TaskSpec, Track2Agent
from off_the_tape.gaussian_joint import GaussianJointForecaster
from off_the_tape.historical import PseudoCard


class GaussianComparisonTest(unittest.TestCase):
    def card(self):
        rng = np.random.default_rng(8)
        dates = pd.bdate_range("2015-01-01", periods=550)
        changes = rng.normal(size=(550, 2)) * [0.01, 0.002]
        values = np.array([2.0, 1.2]) + changes.cumsum(axis=0)
        frame = pd.DataFrame([(d, a, values[i, j]) for i, d in enumerate(dates)
                              for j, a in enumerate(("UST_2Y", "EUR"))], columns=["date", "asset", "value"])
        return PseudoCard("T2-F3", "mixed", dates[-1].date(), frame,
                          ("UST_2Y", "EUR"), (2, 7), "level", ())

    def test_gaussian_moments_and_horizon_covariance(self):
        card = self.card()
        model = GaussianJointForecaster(n_draws=60000)
        mean, cov = model.moments(card)
        draws = model(card)
        np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.0004)
        np.testing.assert_allclose(np.diag(np.cov(draws, rowvar=False)), np.diag(cov), rtol=0.03)
        self.assertAlmostEqual(cov[0, 1], (cov[0, 0]-1.1e-9), places=12)
        future = replace(card, asof=date(2015, 1, 1))
        with self.assertRaisesRegex(ValueError, "post-asof"):
            model(future)

    def test_expected_scales_against_analytic_and_monte_carlo(self):
        scale = expected_scales(np.zeros(2), np.eye(2))
        root_mean = 2**0.25 * gamma(0.75)/np.sqrt(np.pi)
        variance = np.sqrt(2) * (np.sqrt(2/np.pi)-root_mean**2)
        self.assertAlmostEqual(scale["joint"], 2*variance, places=12)
        self.assertAlmostEqual(scale["marginal"], 1/np.sqrt(np.pi))
        for delta in (0.7, 110.0):
            rng = np.random.default_rng(91)
            diff = rng.normal(delta, np.sqrt(2), size=500000)
            mc = 2*np.var(np.sqrt(np.abs(diff)))
            actual = expected_scales(np.array([delta, 0]), np.eye(2))["joint"]
            self.assertAlmostEqual(actual/mc, 1.0, delta=0.015)

    def test_pca_draws_match_production_numeric_path(self):
        card = self.card()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for panel_id, asset in (("rates_daily", "UST_2Y"), ("g10_fx_daily", "EUR")):
                card.panel.loc[card.panel.asset == asset].to_parquet(path/f"{panel_id}.parquet")
            task = TaskSpec("synthetic", card.family, card.asof, card.assets, card.horizons,
                            "level", {"UST_2Y": "rates_daily", "EUR": "g10_fx_daily"})
            production = Track2Agent().forecast(task, path, path, card.asof, seed=17, numeric_only=True)
            np.testing.assert_array_equal(pca_draws(card, 17), production.draws)

    def test_summary_counts_origins_not_seeds(self):
        rows = [{"split": "validation", "shape": "mixed", "asof": origin,
                 "pca": pca, "gaussian": 1.0}
                for origin, pca in (("2020-01-01", 0.5), ("2020-01-01", 1.5),
                                    ("2020-02-01", 2.0), ("2020-02-01", 2.0))]
        result = summary(rows)[0]
        self.assertEqual(result["origins"], 2)
        self.assertEqual(result["pca_minus_gaussian"], 0.5)


if __name__ == "__main__":
    unittest.main()
