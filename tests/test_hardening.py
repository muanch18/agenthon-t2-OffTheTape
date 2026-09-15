"""Frozen candidate, seed, Docker, and final validation checks."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from off_the_tape.candidate import CURRENT_CANDIDATE
from off_the_tape.output_validation import validate_draw_matrix
from off_the_tape.seeds import SeedManager


class Task:
    assets = ("EUR",)
    horizons = (21,)
    asset_panels = {"EUR": "g10_fx_daily"}


class HardeningTest(unittest.TestCase):
    def test_candidate_is_frozen_to_supported_numeric_control(self):
        self.assertEqual(CURRENT_CANDIDATE.numeric_model, "pca_joint")
        self.assertEqual(CURRENT_CANDIDATE.n_draws, 1000)
        self.assertFalse(CURRENT_CANDIDATE.text_conditioning)
        self.assertFalse(any((CURRENT_CANDIDATE.mean_adjustment,
                              CURRENT_CANDIDATE.volatility_adjustment,
                              CURRENT_CANDIDATE.skew_adjustment,
                              CURRENT_CANDIDATE.shock_mixture,
                              CURRENT_CANDIDATE.joint_adjustment)))

    def test_seed_derivation_is_stable_and_component_specific(self):
        seeds = SeedManager(2026)
        self.assertEqual(seeds.derive("pca:rates"), SeedManager(2026).derive("pca:rates"))
        self.assertNotEqual(seeds.derive("pca:rates"), seeds.derive("conditioner"))

    def test_final_validator_rejects_draw_floor_constant_and_bad_fx(self):
        good = np.linspace(.9, 1.1, 200)[:, None]
        validate_draw_matrix(good, Task(), 200)
        with self.assertRaisesRegex(ValueError, "at least 200"):
            validate_draw_matrix(good[:199], Task(), 199)
        with self.assertRaisesRegex(ValueError, "constant"):
            validate_draw_matrix(np.ones((200, 1)), Task(), 200)
        with self.assertRaisesRegex(ValueError, "non-positive"):
            validate_draw_matrix(np.linspace(-1, 1, 200)[:, None], Task(), 200)

    def test_docker_contract_is_pinned_and_excludes_research_data(self):
        root = Path(__file__).parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        ignore = (root / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn('FROM python:3.13-slim-bookworm', dockerfile)
        self.assertIn('qfbench2.interface_version="2.0"', dockerfile)
        self.assertIn("83c6dc036bec03862b78e093bf8804d98964dad5", dockerfile)
        self.assertIn("refs/tags/v2.4.0", dockerfile)
        self.assertIn("reports", ignore)
        self.assertIn("*.parquet", ignore)


if __name__ == "__main__":
    unittest.main()
