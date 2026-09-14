"""Behavioral tests for coherent PCA factor paths and local diagnostics."""

import unittest

import numpy as np
import pandas as pd

from off_the_tape.historical import historical_cases
from off_the_tape.pca_joint import PCAJointForecaster


def coupled_panel(mode="rates"):
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2023-01-02", periods=145)
    common = rng.standard_t(4, size=len(dates)) * 0.025
    noise = rng.normal(0, 0.002, size=(len(dates), 3))
    moves = np.column_stack((common, 0.85 * common, -0.7 * common)) + noise
    start = np.array([3.0, 4.0, 5.0]) if mode == "rates" else np.array([1.1, 0.8, 1.4])
    if mode == "rates":
        levels = start + np.cumsum(moves, axis=0)
        panel_id = "rates_daily"
    else:
        levels = start * np.exp(np.cumsum(moves, axis=0))
        panel_id = "g10_fx_daily"
    frame = pd.DataFrame(
        (day, asset, value, panel_id)
        for day, row in zip(dates, levels)
        for asset, value in zip(("A", "B", "C"), row)
    ).rename(columns={0: "date", 1: "asset", 2: "value", 3: "panel_id"})
    return frame, dates[120].date(), panel_id


def make_case(mode="rates"):
    frame, asof, panel_id = coupled_panel(mode)
    case = historical_cases(
        frame, family="T2-F3", panel_id=panel_id, assets=("A", "B", "C"),
        horizons=(2, 5), target_type="level", origins=(asof,),
    )[0]
    return frame, case


class PCAJointForecasterTest(unittest.TestCase):
    def test_deterministic_joint_paths_and_no_future_leakage(self):
        frame, case = make_case()
        model = PCAJointForecaster("difference", lookback=100, block_size=4, seed=42)
        forecast = model.forecast(case.card)
        self.assertEqual(forecast.draws.shape, (1000, 6))
        self.assertEqual(forecast.paths.shape, (1000, 5, 3))
        self.assertEqual(forecast.daily_transforms.shape, (1000, 5, 3))
        np.testing.assert_array_equal(forecast.draws, model(case.card))
        for column, (asset, horizon) in enumerate(case.card.cells):
            asset_index = case.card.assets.index(asset)
            np.testing.assert_array_equal(
                forecast.draws[:, column], forecast.paths[:, horizon - 1, asset_index]
            )

        changed = frame.copy()
        changed.loc[changed.date > pd.Timestamp(case.card.asof), "value"] += 100
        changed_case = historical_cases(
            changed, family="T2-F3", panel_id="rates_daily", assets=("A", "B", "C"),
            horizons=(2, 5), target_type="level", origins=(case.card.asof,),
        )[0]
        np.testing.assert_array_equal(forecast.draws, model(changed_case.card))
        self.assertEqual(forecast.diagnostics["training_end"], case.card.asof.isoformat())

    def test_rates_reconstruction_diagnostics_and_correlation(self):
        _, case = make_case()
        forecast = PCAJointForecaster("difference", lookback=100).forecast(case.card)
        last = case.card.panel.pivot(index="date", columns="asset", values="value")
        last = last.loc[:, list(case.card.assets)].iloc[-1].to_numpy()
        np.testing.assert_allclose(
            forecast.paths[:, 0, :], last + forecast.daily_transforms[:, 0, :]
        )
        np.testing.assert_allclose(
            forecast.paths[:, 4, :], last + forecast.daily_transforms.sum(axis=1)
        )
        self.assertTrue(np.isfinite(forecast.draws).all())
        diag = forecast.diagnostics
        self.assertEqual(len(diag["explained_variance_ratio"]), 2)
        self.assertEqual(np.asarray(diag["residual_covariance"]).shape, (3, 3))
        self.assertEqual(len(diag["simulated_quantiles"]), 6)
        self.assertGreater(diag["historical_correlation"][0][1], 0.8)
        self.assertGreater(diag["simulated_correlation"][0][1], 0.5)
        self.assertLess(diag["simulated_correlation"][0][2], -0.4)

    def test_fx_paths_are_positive_reconstructed_levels(self):
        _, case = make_case("fx")
        forecast = PCAJointForecaster("log_price", lookback=100).forecast(case.card)
        last = case.card.panel.pivot(index="date", columns="asset", values="value")
        last = last.loc[:, list(case.card.assets)].iloc[-1].to_numpy()
        np.testing.assert_allclose(
            forecast.paths[:, 0, :], last * np.exp(forecast.daily_transforms[:, 0, :])
        )
        self.assertTrue((forecast.draws > 0).all())

    def test_factor_return_target_is_cumulative_log_return(self):
        dates = pd.bdate_range("2023-01-02", periods=45)
        frame = pd.DataFrame(
            (day, "MOM", 0.01 * np.sin(i)) for i, day in enumerate(dates)
        ).rename(columns={0: "date", 1: "asset", 2: "value"})
        case = historical_cases(
            frame, family="T2-F1", panel_id="factors_daily", assets=("MOM",),
            horizons=(2,), target_type="log_return", origins=(dates[35].date(),),
        )[0]
        forecast = PCAJointForecaster("simple_return", lookback=30).forecast(case.card)
        self.assertEqual(forecast.draws.shape, (1000, 1))
        np.testing.assert_allclose(
            forecast.paths[:, 1, 0], forecast.daily_transforms[:, :2, 0].sum(axis=1)
        )


if __name__ == "__main__":
    unittest.main()
