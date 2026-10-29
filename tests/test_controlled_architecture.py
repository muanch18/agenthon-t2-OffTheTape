from dataclasses import replace
import unittest

import numpy as np

from experiments.controlled_architecture import forecast, gaussian_daily, prepare
from tests import test_gaussian_comparison as fixtures
from off_the_tape.pca_joint import PCAJointForecaster
from off_the_tape.seeds import SeedManager


class ControlledArchitectureTest(unittest.TestCase):
    def test_shared_window_transforms_and_positive_fx(self):
        card = fixtures.GaussianComparisonTest().card()
        groups = prepare(card)
        self.assertEqual([g[2] for g in groups], ['difference', 'log_price'])
        for local, features, mode, train, anchor in groups:
            self.assertEqual(train.shape, (504, 1))
            np.testing.assert_array_equal(train, features.history(mode).changes[-504:])
        draws = forecast(card, groups, 'gaussian', 17)
        self.assertTrue((draws[:, 2:] > 0).all())
        np.testing.assert_array_equal(draws, forecast(card, groups, 'gaussian', 17))
        future = replace(card, asof=card.asof.replace(year=2014))
        with self.assertRaises(ValueError):
            prepare(future)

    def test_gaussian_daily_moments(self):
        train = np.random.default_rng(9).normal(size=(504, 2)) * [.02, .01]
        daily = gaussian_daily(train, 2, 100000, 22)
        np.testing.assert_allclose(daily.mean(axis=(0, 1)), train.mean(axis=0), atol=.0002)
        np.testing.assert_allclose(np.cov(daily.reshape(-1, 2), rowvar=False), np.cov(train, rowvar=False), atol=.000004)
        cumulative = daily.cumsum(axis=1)
        self.assertAlmostEqual(np.var(cumulative[:, 1, 0])/np.var(cumulative[:, 0, 0]), 2., delta=.03)

    def test_pca_control_parity(self):
        card = fixtures.GaussianComparisonTest().card()
        groups = prepare(card)
        actual = forecast(card, groups, 'pca', 2026)
        for local, features, mode, train, anchor in groups:
            expected = PCAJointForecaster(mode=mode, seed=SeedManager(2026).derive(f'pca:{local.panel_id}')).forecast(local).draws
            j = card.assets.index(local.assets[0])*len(card.horizons)
            np.testing.assert_array_equal(expected, actual[:, j:j+len(card.horizons)])


if __name__ == '__main__':
    unittest.main()
