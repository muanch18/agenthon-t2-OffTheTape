"""Verify case selection is outcome-independent and model inputs respect cutoffs."""

import unittest

import pandas as pd

from experiments.final_cases import GROUPS, RATES, FX, build_manifest, hydrate, partition


class FinalCasesTest(unittest.TestCase):
    def panel(self):
        dates = pd.bdate_range("2015-01-01", "2017-03-01")
        return pd.DataFrame([(d, a, 1.0 + i / 1000) for i, d in enumerate(dates)
                             for a in RATES + FX], columns=["date", "asset", "value"])

    def test_selection_ignores_values_and_hydration_hides_future(self):
        panel = self.panel()
        kwargs = dict(start="2017-01-02", end="2017-02-01", min_history=20)
        manifest = build_manifest(panel, **kwargs)
        changed = panel.copy()
        changed.loc[changed.date > "2017-01-02", "value"] *= 50
        self.assertEqual(manifest, build_manifest(changed, **kwargs))
        for spec in manifest["cases"]:
            case = hydrate(panel, spec)
            self.assertLessEqual(case.card.panel.date.max().date(), case.card.asof)
            self.assertFalse(hasattr(case.card, "realized"))
            self.assertTrue(all(2 <= h <= 12 for h in spec["horizons"]))

    def test_boundary_purge_and_explicit_missing_targets(self):
        self.assertIsNone(partition(pd.Timestamp("2017-12-29"), pd.Timestamp("2018-01-02")))
        panel = self.panel()
        base = build_manifest(panel, start="2017-01-02", end="2017-01-02", min_history=20)
        target = base["cases"][0]["target_dates"][0]
        incomplete = panel.loc[panel.date != pd.Timestamp(target)]
        result = build_manifest(incomplete, start="2017-01-02", end="2017-01-02", min_history=20)
        self.assertTrue(any(r["reason"] == "missing_target" for r in result["excluded"]))


if __name__ == "__main__":
    unittest.main()
