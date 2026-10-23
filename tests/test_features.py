"""Verify shared feature cutoffs, transformations, isolation and volatility definitions."""

from datetime import date
import unittest

import numpy as np
import pandas as pd

from off_the_tape.features import FeatureLayer
from off_the_tape.historical import PseudoCard
from off_the_tape.pca_joint import PCAJointForecaster


def panel(n=150):
    rng = np.random.default_rng(5)
    values = 2+np.cumsum(rng.normal(0, 0.001, (n, 2)), axis=0)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame([(d, a, values[i, j]) for i, d in enumerate(dates)
                         for j, a in enumerate(("A", "B"))], columns=["date", "asset", "value"])


class FeatureTest(unittest.TestCase):
    def test_cutoff_filtering_alias_and_duplicate_policy(self):
        raw = panel()
        for invalid in (None, "2020-01-01T12:00:00", "2020-01-01T00:00:00Z"):
            with self.assertRaisesRegex(ValueError, "as-of"):
                FeatureLayer(raw, assets=("A",), asof=invalid)
        cutoff = raw.date.unique()[-4]
        duplicate = raw.iloc[[-8]].copy()
        duplicate["value"] = 9
        raw = pd.concat([raw, duplicate], ignore_index=True).rename(columns={"asset": "asset_id"})
        layer = FeatureLayer(raw, assets=("A", "B"), asof=cutoff, future_policy="filter", duplicate_policy="last")
        self.assertLessEqual(layer.frame.date.max(), cutoff)
        self.assertEqual(layer.frame.loc[(layer.frame.date == duplicate.date.iloc[0]) & (layer.frame.asset == "A"), "value"].iloc[0], 9)
        with self.assertRaisesRegex(ValueError, "as-of"):
            FeatureLayer(raw, assets=("A", "B"), asof=cutoff)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            FeatureLayer(raw, assets=("A", "B"), asof=cutoff, future_policy="filter")

    def test_transformations_match_batch_definitions(self):
        raw = panel()
        layer = FeatureLayer(raw, assets=("B", "A"), asof=raw.date.max().date())
        values = raw.pivot(index="date", columns="asset", values="value")[["B", "A"]].to_numpy()
        for mode, expected in (("difference", np.diff(values, axis=0)),
                               ("log_price", np.diff(np.log(values), axis=0)),
                               ("simple_return", np.log1p(values))):
            history = layer.history(mode)
            np.testing.assert_array_equal(history.changes, expected)
            self.assertEqual(len(history.dates), len(expected))
            self.assertIs(history, layer.history(mode))

    def test_volatility_matches_independent_batch_calculation(self):
        raw = panel()
        layer = FeatureLayer(raw, assets=("A", "B"), asof=raw.date.max().date())
        vol = layer.volatility("log_price", lookback_changes=100)
        x = layer.history("log_price").changes[-100:]
        weights = 0.5**(np.arange(100)[::-1]/20)
        weights /= weights.sum()
        for j in range(2):
            mean = sum(weights*x[:, j])
            ewma = np.sqrt(sum(weights*(x[:, j]-mean)**2)/(1-sum(weights**2)))
            self.assertAlmostEqual(vol.ewma_sd[j], ewma, places=14)
        np.testing.assert_array_equal(vol.historical_sd, x.std(axis=0, ddof=1))
        np.testing.assert_array_equal(vol.recent_sd, x[-60:].std(axis=0, ddof=1))
        np.testing.assert_allclose(vol.variance_blend(), np.sqrt(0.5*vol.historical_sd**2+0.5*vol.ewma_sd**2))
        self.assertIs(vol, layer.volatility("log_price", lookback_changes=100))
        with self.assertRaises(ValueError):
            layer.volatility("log_price", half_life=1e-300)

    def test_caller_mutations_and_other_units_cannot_change_cached_features(self):
        raw = panel()
        layer = FeatureLayer(raw, assets=("A",), asof=raw.date.max().date())
        expected = layer.history("difference").changes.copy()
        raw["value"] *= 10
        layer.frame["value"] = 0
        layer.wide["A"] = 0
        np.testing.assert_array_equal(layer.history("difference").changes, expected)
        with self.assertRaises(ValueError):
            layer.history("difference").changes[0, 0] = 0
        with self.assertRaises(AttributeError):
            layer.asof = date(2099, 1, 1)
        other = FeatureLayer(raw, assets=("A",), asof=raw.date.max().date())
        self.assertFalse(np.array_equal(layer.history("difference").changes, other.history("difference").changes))

    def test_missing_cells_are_not_imputed_and_bad_domains_are_rejected(self):
        raw = panel().iloc[1:]
        layer = FeatureLayer(raw, assets=("A", "B"), asof=raw.date.max().date())
        self.assertTrue(layer.wide.isna().any().any())
        with self.assertRaisesRegex(ValueError, "complete and finite"):
            layer.history("difference")
        raw = panel()
        raw.loc[0, "value"] = 0
        with self.assertRaisesRegex(ValueError, "positive levels"):
            FeatureLayer(raw, assets=("A",), asof=raw.date.max().date()).history("log_price")

    def test_prepared_snapshot_reuse_and_stale_input_rejection(self):
        raw = panel()
        layer = FeatureLayer(raw, assets=("A", "B"), asof=raw.date.max().date())
        card = PseudoCard("T2-F3", "test", layer.asof, layer.frame, layer.assets, (2, 5), "level", ())
        model = PCAJointForecaster(mode="difference")
        np.testing.assert_array_equal(model.forecast(card).draws, model.forecast(card, features=layer).draws)
        changed = raw.copy()
        changed["value"] += 1
        stale = FeatureLayer(changed, assets=layer.assets, asof=layer.asof)
        with self.assertRaisesRegex(ValueError, "do not match"):
            model.forecast(card, features=stale)


if __name__ == "__main__":
    unittest.main()
