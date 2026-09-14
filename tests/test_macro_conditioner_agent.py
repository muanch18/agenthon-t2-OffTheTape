"""Joint conditioning, guardrails, and end-to-end output contract tests."""

import json
import shutil
import unittest
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from off_the_tape.agent import TaskSpec, Track2Agent, main
from off_the_tape.historical import PseudoCard
from off_the_tape.macro_conditioner import MacroConditioner, NumericContext
from off_the_tape.macro_state import MacroState


class ConditioningTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        common = rng.normal(0, 0.04, size=(1000, 21, 1))
        daily = np.concatenate((common + rng.normal(0, 0.01, (1000, 21, 1)),
                                0.8 * common + rng.normal(0, 0.01, (1000, 21, 1))), axis=2)
        self.base = np.array([3.0, 4.0])[None, None, :] + np.cumsum(daily, axis=1)
        self.context = NumericContext(("difference", "difference"), np.array([3., 4.]), np.array([.04, .04]))
        self.card = PseudoCard("T2-F1", "rates_daily", date(2024, 1, 31), pd.DataFrame(),
                               ("UST_2Y", "UST_10Y"), (5, 21), "level", ())
        self.state = MacroState(policy_stance=.8, policy_stance_change=.6, market_stress=.6,
                                shock_probability=.8, expected_skew=.8, confidence=.9)

    def test_neutral_low_confidence_and_guardrails(self):
        conditioner = MacroConditioner()
        neutral = conditioner.condition(self.base, MacroState.neutral(), self.card, self.context, 7)
        self.assertTrue(np.array_equal(neutral.paths, self.base))
        high = conditioner.condition(self.base, self.state, self.card, self.context, 7)
        low = conditioner.condition(self.base, replace(self.state, confidence=.05), self.card, self.context, 7)
        self.assertGreater(high.draws.mean(), neutral.draws.mean())
        self.assertLess(np.linalg.norm(low.draws - neutral.draws), np.linalg.norm(high.draws - neutral.draws))
        self.assertLessEqual(high.diagnostics["vol_multiplier"], 1.4)
        self.assertLessEqual(max(abs(v) for v in high.diagnostics["mean_shift_daily"].values()),
                             .35 * .04 / np.sqrt(21) + 1e-12)

    def test_family_order_shock_tail_and_deterministic_paths(self):
        c = MacroConditioner()
        f1 = c.condition(self.base, self.state, self.card, self.context, 11)
        f2 = c.condition(self.base, self.state, replace(self.card, family="T2-F2"), self.context, 11,
                         volatility=False, skew=False, shock=False, joint=False)
        f1mean = c.condition(self.base, self.state, self.card, self.context, 11,
                             volatility=False, skew=False, shock=False, joint=False)
        self.assertGreater(np.linalg.norm(f2.draws - self.base[:, [4, 20], :].transpose(0, 2, 1).reshape(1000, 4)),
                           np.linalg.norm(f1mean.draws - self.base[:, [4, 20], :].transpose(0, 2, 1).reshape(1000, 4)))
        f4card = replace(self.card, family="T2-F4")
        f4 = c.condition(self.base, self.state, f4card, self.context, 11)
        f4_again = c.condition(self.base, self.state, f4card, self.context, 11)
        self.assertTrue(np.array_equal(f4.draws, f4_again.draws))
        self.assertGreater(f4.diagnostics["shocked_draws"], 0)
        no_shock = c.condition(self.base, self.state, f4card, self.context, 11, shock=False)
        self.assertGreater(np.quantile(f4.draws[:, 1], .99), np.quantile(no_shock.draws[:, 1], .99))
        self.assertTrue(np.array_equal(f4.draws[:, 0], f4.paths[:, 4, 0]))
        self.assertTrue(np.array_equal(f4.draws[:, 1], f4.paths[:, 20, 0]))

    def test_f3_shared_factor_keeps_joint_draw_structure(self):
        c = MacroConditioner()
        f3 = c.condition(self.base, self.state, replace(self.card, family="T2-F3"), self.context, 7)
        self.assertEqual(f3.draws.shape, (1000, 4))
        self.assertGreater(np.corrcoef(f3.draws[:, 1], f3.draws[:, 3])[0, 1], .5)
        self.assertGreater(f3.diagnostics["factor_scale"], 0)

    def test_fx_levels_stay_positive_and_extreme_signals_are_capped(self):
        rng = np.random.default_rng(23)
        fx = np.array([1.1, 150.0])[None, None, :] * np.exp(
            np.cumsum(rng.normal(0, .004, (1000, 21, 2)), axis=1)
        )
        context = NumericContext(("log_price", "log_price"), np.array([1.1, 150.0]), np.array([.004, .004]))
        card = replace(self.card, family="T2-F4", panel_id="g10_fx_daily", assets=("EUR", "JPY"))
        extreme = replace(self.state, policy_stance=100, policy_uncertainty=100,
                          market_stress=100, shock_probability=100, confidence=100)
        result = MacroConditioner().condition(fx, extreme, card, context, 9)
        self.assertTrue(np.isfinite(result.draws).all())
        self.assertTrue((result.draws > 0).all())
        self.assertLessEqual(result.diagnostics["shock_probability"], .20)
        self.assertLessEqual(result.diagnostics["vol_multiplier"], 1.4)


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.root = Path("reports") / f"agent_fixture_{uuid.uuid4().hex}"
        (self.root / "unit" / "panels").mkdir(parents=True)
        (self.root / "unit" / "text").mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        dates = pd.bdate_range("2023-01-02", periods=300)
        rng = np.random.default_rng(17)
        rows = []
        for asset, start in (("UST_2Y", 3.0), ("UST_10Y", 4.0)):
            values = start + np.cumsum(rng.normal(0, .03, len(dates)))
            rows.extend(zip(dates, [asset] * len(dates), values))
        pd.DataFrame(rows, columns=["date", "asset", "value"]).to_parquet(
            self.root / "unit" / "panels" / "rates_daily.parquet", index=False)
        self.asof = dates[-1].date()
        (self.root / "unit" / "card.toml").write_text(
            f'[task]\nid="fixture"\n[metadata]\ncategory="T2-F3"\n'
            f'[provenance]\ndata_cutoff="{self.asof}"\n'
            '[panels]\npanel_ids=["rates_daily"]\n[panels.rates_daily]\n'
            'asset_ids=["UST_2Y","UST_10Y"]\n[targets]\n'
            'asset_ids=["UST_2Y","UST_10Y"]\nhorizons=[5,21]\ntarget_type="level"\n',
            encoding="utf-8",
        )

    def test_numeric_bypass_fallback_and_cli_files(self):
        task = TaskSpec.from_card(self.root / "unit" / "card.toml")
        agent = Track2Agent()
        args = (task, self.root / "unit" / "panels", self.root / "unit" / "text", self.asof)
        numeric = agent.forecast(*args, seed=5, numeric_only=True)
        ablated = agent.forecast(*args, seed=5, macro_override=MacroState.neutral())
        fallback = agent.forecast(*args, seed=5)
        self.assertTrue(np.array_equal(numeric.draws, ablated.draws))
        self.assertTrue(np.array_equal(numeric.draws, fallback.draws))
        self.assertEqual(fallback.macro.source, "fallback")
        fallback_out = self.root / "fallback" / "forecast.parquet"
        agent.write(fallback, fallback_out)
        self.assertTrue(fallback_out.is_file())
        self.assertIn("fallback", (fallback_out.parent / "forecast_rationale.md").read_text(encoding="utf-8"))
        out = self.root / "out" / "forecast.parquet"
        self.assertEqual(main(["forecast", "--panels", str(self.root / "unit" / "panels"),
                               "--text", str(self.root / "unit" / "text"),
                               "--asof", str(self.asof), "--out", str(out), "--numeric-only"]), 0)
        self.assertTrue(out.is_file())
        meta = json.loads((out.parent / "forecast_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta, {"unit_id": "fixture", "asof": str(self.asof),
                                "asset_ids": ["UST_2Y", "UST_10Y"], "horizons": [5, 21],
                                "representation": "samples", "n_draws": 1000})
        self.assertTrue((out.parent / "forecast_rationale.md").read_text(encoding="utf-8").strip())
        frame = pd.read_parquet(out)
        self.assertEqual(len(frame), 4000)
        self.assertEqual(frame.groupby("draw").size().unique().tolist(), [4])

    def test_agent_ignores_panel_rows_after_pseudo_asof(self):
        path = self.root / "unit" / "panels" / "rates_daily.parquet"
        frame = pd.read_parquet(path)
        origin = pd.Timestamp(frame["date"].sort_values().unique()[-30]).date()
        task = replace(TaskSpec.from_card(self.root / "unit" / "card.toml"), asof=origin)
        agent = Track2Agent()
        first = agent.forecast(task, path.parent, self.root / "unit" / "text", origin,
                               seed=91, numeric_only=True)
        frame.loc[pd.to_datetime(frame["date"]) > pd.Timestamp(origin), "value"] += 1000
        frame.to_parquet(path, index=False)
        second = agent.forecast(task, path.parent, self.root / "unit" / "text", origin,
                                seed=91, numeric_only=True)
        self.assertTrue(np.array_equal(first.draws, second.draws))


if __name__ == "__main__":
    unittest.main()
