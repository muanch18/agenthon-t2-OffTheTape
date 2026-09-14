"""Regression tests for historical cutoffs, target construction, and draw scoring."""

import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import pandas as pd

from off_the_tape.historical import evaluate, historical_cases
from off_the_tape.cli import main, samples_from_frame
from off_the_tape.bootstrap import JointRowBootstrap
from off_the_tape.scoring import raw_track2_score
from qfbench2_track_forecasting.scoring import _composite


def panel(values):
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.DataFrame(
        (date, asset, value)
        for date, row in zip(dates, values)
        for asset, value in zip(("A", "B"), row)
    ).rename(columns={0: "date", 1: "asset", 2: "value"})


class HistoricalCasesTest(unittest.TestCase):
    def test_cutoff_and_asset_major_outcomes(self):
        frame = panel([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
        case = historical_cases(
            frame, family="T2-F3", panel_id="rates_daily", assets=["B", "A"], horizons=[2, 1],
            target_type="level", origins=["2024-01-02"],
        )[0]
        self.assertEqual(case.card.asof.isoformat(), "2024-01-02")
        self.assertEqual((case.card.family, case.card.panel_id), ("T2-F3", "rates_daily"))
        self.assertEqual(case.card.cells, (("B", 2), ("B", 1), ("A", 2), ("A", 1)))
        self.assertEqual(case.realized.tolist(), [40, 30, 4, 3])
        self.assertEqual(case.card.panel["date"].max(), pd.Timestamp("2024-01-02"))
        self.assertEqual(case.card.target_dates, (pd.Timestamp("2024-01-04").date(), pd.Timestamp("2024-01-03").date()))

    def test_cumulative_log_returns_exclude_origin(self):
        frame = panel([(0.9, 0), (0.5, 0), (0.1, 0.2), (0.2, -0.1)])
        case = historical_cases(
            frame, family="T2-F3", panel_id="factors_daily", assets=["A", "B"], horizons=[2],
            target_type="log_return", origins=["2024-01-02"],
        )[0]
        np.testing.assert_allclose(case.realized, [np.log1p(0.1) + np.log1p(0.2), np.log1p(0.2) + np.log1p(-0.1)])

    def test_rejects_incomplete_history_and_future(self):
        frame = panel([(1, 10), (2, 20), (3, 30)])
        with self.assertRaisesRegex(ValueError, "business-day target"):
            historical_cases(frame, family="T2-F3", panel_id="rates_daily", assets=["A", "B"], horizons=[2], target_type="level", origins=["2024-01-02"])
        frame = frame.drop(frame[(frame.date == pd.Timestamp("2024-01-02")) & (frame.asset == "B")].index)
        with self.assertRaisesRegex(ValueError, "every panel date"):
            historical_cases(frame, family="T2-F3", panel_id="rates_daily", assets=["A", "B"], horizons=[1], target_type="level", origins=["2024-01-01"])

    def test_forecaster_sees_only_history(self):
        case = historical_cases(
            panel([(1, 10), (2, 20), (3, 30)]), family="T2-F1", panel_id="rates_daily", assets=["A"],
            horizons=[1], target_type="level", origins=["2024-01-02"],
        )[0]
        def forecast(card):
            self.assertEqual(card.panel.value.max(), 2)
            self.assertFalse(hasattr(card, "realized"))
            return np.full((200, 1), 3.0)
        result = evaluate([case], forecast)[0]
        self.assertEqual(result["realized"], [3.0])
        self.assertAlmostEqual(result["composite"], 0.0)

    def test_observed_business_day_index_crosses_weekend(self):
        frame = panel([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)])
        case = historical_cases(
            frame, family="T2-F3", panel_id="rates_daily", assets=["A"],
            horizons=[1], target_type="level", origins=["2024-01-05"],
        )[0]
        self.assertEqual(case.card.target_dates[0].isoformat(), "2024-01-08")
        self.assertEqual(case.realized.tolist(), [6.0])

    def test_missing_business_day_target_does_not_shift_forward(self):
        frame = panel([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)])
        frame = frame.loc[frame.date != pd.Timestamp("2024-01-08")]
        with self.assertRaisesRegex(ValueError, "business-day target"):
            historical_cases(
                frame, family="T2-F3", panel_id="rates_daily", assets=["A"],
                horizons=[1], target_type="level", origins=["2024-01-05"],
            )

    def test_task_identity_is_validated(self):
        frame = panel([(1, 10), (2, 20)])
        frame["panel_id"] = "rates_daily"
        with self.assertRaisesRegex(ValueError, "different panel_id"):
            historical_cases(
                frame, family="T2-F3", panel_id="g10_fx_daily", assets=["A"],
                horizons=[1], target_type="level", origins=["2024-01-01"],
            )
        with self.assertRaisesRegex(ValueError, "Track 2 family"):
            historical_cases(
                frame, family="T2-F5", panel_id="rates_daily", assets=["A"],
                horizons=[1], target_type="level", origins=["2024-01-01"],
            )


class BootstrapTest(unittest.TestCase):
    def test_joint_rows_shape_and_reproducibility(self):
        frame = panel([(1, 10), (2, 20), (4, 40), (7, 70), (8, 80), (10, 100)])
        case = historical_cases(
            frame, family="T2-F3", panel_id="rates_daily", assets=["A", "B"],
            horizons=[1, 2], target_type="level", origins=["2024-01-04"],
        )[0]
        model = JointRowBootstrap("difference", lookback=3, seed=7)
        draws = model(case.card)
        self.assertEqual(draws.shape, (1000, 4))
        np.testing.assert_array_equal(draws, model(case.card))
        np.testing.assert_allclose(draws[:, 2] - 70, 10 * (draws[:, 0] - 7))
        np.testing.assert_allclose(draws[:, 3] - 70, 10 * (draws[:, 1] - 7))

        changed = frame.copy()
        changed.loc[changed.date > pd.Timestamp("2024-01-04"), "value"] = 9999
        changed_case = historical_cases(
            changed, family="T2-F3", panel_id="rates_daily", assets=["A", "B"],
            horizons=[1, 2], target_type="level", origins=["2024-01-04"],
        )[0]
        np.testing.assert_array_equal(draws, model(changed_case.card))

    def test_log_price_and_simple_return_modes(self):
        levels = panel([(10, 20), (11, 22), (12, 24), (13, 26), (14, 28)])
        level_case = historical_cases(
            levels, family="T2-F3", panel_id="g10_fx_daily", assets=["A", "B"],
            horizons=[1], target_type="level", origins=["2024-01-04"],
        )[0]
        level_draws = JointRowBootstrap("log_price")(level_case.card)
        self.assertEqual(level_draws.shape, (1000, 2))
        np.testing.assert_allclose(level_draws[:, 1], 2 * level_draws[:, 0])

        returns = panel([(0.01, 0.02), (0.02, 0.04), (0.03, 0.06), (0.04, 0.08)])
        return_case = historical_cases(
            returns, family="T2-F3", panel_id="factors_daily", assets=["A", "B"],
            horizons=[1], target_type="log_return", origins=["2024-01-03"],
        )[0]
        return_draws = JointRowBootstrap("simple_return")(return_case.card)
        self.assertEqual(return_draws.shape, (1000, 2))
        self.assertTrue(np.isfinite(return_draws).all())


class ScoringTest(unittest.TestCase):
    def test_deterministic_components(self):
        scored = raw_track2_score(np.tile([0.0, 1.0], (200, 1)), np.array([1.0, 3.0]))
        expected_joint = 2 * (np.sqrt(2) - 1) ** 2
        self.assertAlmostEqual(scored["marginal"], 1.5)
        self.assertAlmostEqual(scored["joint"], expected_joint)
        self.assertAlmostEqual(scored["tail"], 0.75)
        self.assertAlmostEqual(scored["composite"], 0.5 * 1.5 + 0.3 * expected_joint + 0.2 * 0.75)

    def test_single_cell_renormalizes_and_validates(self):
        draws = np.full((200, 1), 2.0)
        scored = raw_track2_score(draws, np.array([3.0]))
        self.assertEqual(scored["joint"], 0.0)
        self.assertAlmostEqual(scored["composite"], 5 / 7 * scored["marginal"] + 2 / 7 * scored["tail"])
        with self.assertRaisesRegex(ValueError, "at least 200"):
            raw_track2_score(draws[:199], np.array([3.0]))
        with self.assertRaisesRegex(ValueError, "shape"):
            raw_track2_score(draws, np.array([3.0, 4.0]))

    def test_raw_score_matches_canonical_composite(self):
        draws = np.column_stack((np.linspace(1, 3, 200), np.linspace(4, 8, 200)))
        realized = np.array([2.5, 5.0])
        expected = _composite(
            draws, realized, weights=(0.5, 0.3, 0.2),
            tail_levels=(0.01, 0.05, 0.95, 0.99), joint="variogram",
            tail_metric="pinball", ref_scale=None,
        )
        for name, value in expected.items():
            self.assertAlmostEqual(raw_track2_score(draws, realized)[name], value)


class SavedForecastTest(unittest.TestCase):
    def setUp(self):
        self.frame = panel([(1, 10), (2, 20), (3, 30)])
        self.card = historical_cases(
            self.frame, family="T2-F3", panel_id="rates_daily", assets=["B", "A"], horizons=[1],
            target_type="level", origins=["2024-01-02"],
        )[0].card
        self.draws = pd.DataFrame(
            (draw, asset, 1, value)
            for draw in range(200)
            for asset, value in (("A", 3.0), ("B", 30.0))
        ).rename(columns={0: "draw", 1: "asset", 2: "horizon", 3: "value"})

    def test_saved_draws_follow_card_grid(self):
        matrix = samples_from_frame(self.draws.sample(frac=1, random_state=4), self.card)
        self.assertEqual(matrix.shape, (200, 2))
        self.assertEqual(matrix[0].tolist(), [30.0, 3.0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            samples_from_frame(pd.concat([self.draws, self.draws.iloc[:1]]), self.card)
        with self.assertRaisesRegex(ValueError, "grid"):
            samples_from_frame(self.draws.iloc[:-1], self.card)

    def test_cli_writes_unranked_report(self):
        out = Path("report.json")
        with patch("off_the_tape.cli.pd.read_parquet", side_effect=[self.frame, self.draws]), \
             patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_text") as write:
            status = main([
                "--panel", "panel.parquet", "--panel-id", "rates_daily", "--family", "T2-F3", "--forecasts", "forecasts",
                "--assets", "B", "A", "--horizons", "1", "--target-type", "level",
                "--origins", "2024-01-02", "--output", str(out),
            ])
        self.assertEqual(status, 0)
        report = json.loads(write.call_args.args[0])
        self.assertFalse(report["rankable"])
        self.assertEqual(report["cases"][0]["realized"], [30.0, 3.0])


if __name__ == "__main__":
    unittest.main()
