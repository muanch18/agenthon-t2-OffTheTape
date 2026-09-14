"""Raw local diagnostics using the organizer's Track 2 scoring functions."""

from __future__ import annotations

import numpy as np
from qfbench2_common.scoring import crps
from qfbench2_track_forecasting.tail import tail_pinball


def raw_track2_score(draws: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    """Return unnormalized components and the published single-cell weight rule."""
    samples = np.asarray(draws, dtype=float)
    y = np.asarray(realized, dtype=float)
    if samples.ndim != 2 or y.ndim != 1 or samples.shape[1] != len(y):
        raise ValueError("draws must have shape (n_draws, n_assets * n_horizons)")
    if samples.shape[0] < 200:
        raise ValueError("Track 2 requires at least 200 joint draws")
    if not np.isfinite(samples).all() or not np.isfinite(y).all():
        raise ValueError("draws and realized values must be finite")
    marginal = float(crps.crps_marginal(samples, y))
    joint = float(crps.variogram_score(samples, y, p=0.5)) if len(y) > 1 else 0.0
    tail = float(tail_pinball(samples, y, (0.01, 0.05, 0.95, 0.99)))
    weights = (0.5, 0.3, 0.2) if len(y) > 1 else (5 / 7, 0.0, 2 / 7)
    composite = weights[0] * marginal + weights[1] * joint + weights[2] * tail
    return {"marginal": marginal, "joint": joint, "tail": tail, "composite": composite}
