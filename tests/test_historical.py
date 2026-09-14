"""Regression tests for historical cutoffs, target construction, and draw scoring."""

import unittest

import numpy as np
import pandas as pd

from off_the_tape.historical import evaluate, historical_cases
from off_the_tape.scoring import raw_track2_score


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
            frame, assets=["B", "A"], horizons=[2, 1],
            target_type="level", origins=["2024-01-02"],
        )[0]
        self.assertEqual(case.card.asof.isoformat(), "2024-01-02")
        self.assertEqual(case.card.cells, (("B", 2), ("B", 1), ("A", 2), ("A", 1)))
        self.assertEqual(case.realized.tolist(), [40, 30, 4, 3])
        self.assertEqual(case.card.panel["date"].max(), pd.Timestamp("2024-01-02"))
        self.assertEqual(case.card.target_dates, (pd.Timestamp("2024-01-04").date(), pd.Timestamp("2024-01-03").date()))

    def test_cumulative_log_returns_exclude_origin(self):
        frame = panel([(0.9, 0), (0.5, 0), (0.1, 0.2), (0.2, -0.1)])
        case = historical_cases(
            frame, assets=["A", "B"], horizons=[2],
            target_type="log_return", origins=["2024-01-02"],
        )[0]
        np.testing.assert_allclose(case.realized, [np.log1p(0.1) + np.log1p(0.2), np.log1p(0.2) + np.log1p(-0.1)])

    def test_rejects_incomplete_history_and_future(self):
        frame = panel([(1, 10), (2, 20), (3, 30)])
        with self.assertRaisesRegex(ValueError, "future observations"):
            historical_cases(frame, assets=["A", "B"], horizons=[2], target_type="level", origins=["2024-01-02"])
        frame = frame.drop(frame[(frame.date == pd.Timestamp("2024-01-02")) & (frame.asset == "B")].index)
        with self.assertRaisesRegex(ValueError, "every panel date"):
            historical_cases(frame, assets=["A", "B"], horizons=[1], target_type="level", origins=["2024-01-01"])

    def test_forecaster_sees_only_history(self):
        case = historical_cases(
            panel([(1, 10), (2, 20), (3, 30)]), assets=["A"],
            horizons=[1], target_type="level", origins=["2024-01-02"],
        )[0]
        def forecast(card):
            self.assertEqual(card.panel.value.max(), 2)
            self.assertFalse(hasattr(card, "realized"))
            return np.full((200, 1), 3.0)
        result = evaluate([case], forecast)[0]
        self.assertEqual(result["realized"], [3.0])
        self.assertAlmostEqual(result["composite"], 0.0)


class ScoringTest(unittest.TestCase):
    def test_single_cell_renormalizes_and_validates(self):
        draws = np.full((200, 1), 2.0)
        scored = raw_track2_score(draws, np.array([3.0]))
        self.assertEqual(scored["joint"], 0.0)
        self.assertAlmostEqual(scored["composite"], 5 / 7 * scored["marginal"] + 2 / 7 * scored["tail"])
        with self.assertRaisesRegex(ValueError, "at least 200"):
            raw_track2_score(draws[:199], np.array([3.0]))
        with self.assertRaisesRegex(ValueError, "shape"):
            raw_track2_score(draws, np.array([3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
