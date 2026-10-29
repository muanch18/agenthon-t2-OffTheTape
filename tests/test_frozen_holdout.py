"""Check frozen-arm parity and clustered holdout summary behavior, using synthetic data."""

import unittest

import numpy as np

from experiments.frozen_holdout import ARM_RULES, frozen_draws, paired_summary, verify_protocol
from experiments.volatility_ablation import variants
import test_gaussian_comparison as fixtures


class FrozenHoldoutTest(unittest.TestCase):
    def test_only_frozen_arms_and_exact_experiment_parity(self):
        card = fixtures.GaussianComparisonTest().card()
        frozen, scales = frozen_draws(card, 17)
        previous, old_scales, _ = variants(card, 17)
        self.assertEqual(set(frozen), {"pca_current", "pca_candidate", "gaussian_current", "gaussian_candidate"})
        self.assertEqual(scales, old_scales)
        for model, rule in ARM_RULES.items():
            np.testing.assert_array_equal(frozen[model+"_current"], previous[(model, "current")])
            np.testing.assert_array_equal(frozen[model+"_candidate"], previous[(model, rule)])

    def test_sums_keep_case_weights_and_seed_grouping(self):
        rows = [{"case_id": case_id, "asof": origin, "seed": seed,
                 "pca_current": 1.0, "pca_candidate": candidate}
                for case_id, origin, candidate in (("a", "2022-01-01", 0.5),
                                                   ("b", "2022-01-01", 0.5),
                                                   ("c", "2022-02-01", 1.5)) for seed in (17, 83)]
        result = paired_summary(rows, "pca", block_length=1, reps=100, quantiles=[0.0125, 0.9875], seed=901)
        self.assertEqual((result["cases"], result["origins"]), (3, 2))
        self.assertAlmostEqual(result["delta"], -1/6)
        self.assertAlmostEqual(result["delta_by_seed"]["17"], -1/6)

    def test_tampered_candidates_rejected_before_reading_manifest(self):
        with self.assertRaisesRegex(ValueError, "candidate definition"):
            verify_protocol({"candidate_rules": {"pca": "blend60"}}, None)


if __name__ == "__main__":
    unittest.main()
