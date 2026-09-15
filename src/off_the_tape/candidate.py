"""Frozen competition candidate chosen from the documented held-out ablation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateConfig:
    numeric_model: str = "pca_joint"
    n_draws: int = 1000
    text_conditioning: bool = False
    mean_adjustment: bool = False
    volatility_adjustment: bool = False
    skew_adjustment: bool = False
    shock_mixture: bool = False
    joint_adjustment: bool = False
    root_seed: int = 2026


CURRENT_CANDIDATE = CandidateConfig()
