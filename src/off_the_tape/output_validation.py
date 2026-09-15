"""Final forecast matrix and serialized-frame validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .agent_types import TaskSpecLike


def validate_draw_matrix(draws: np.ndarray, task: TaskSpecLike, n_draws: int) -> None:
    if n_draws < 200:
        raise ValueError("Track 2 requires at least 200 draws")
    values = np.asarray(draws)
    expected = (n_draws, len(task.assets) * len(task.horizons))
    if values.shape != expected:
        raise ValueError(f"draw matrix shape {values.shape} != {expected}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("draw matrix contains non-finite or non-numeric values")
    if np.any(np.abs(values) > 1e12):
        raise ValueError("forecast exceeds broad plausibility limit")
    for j, asset in enumerate(task.assets):
        columns = slice(j * len(task.horizons), (j + 1) * len(task.horizons))
        if "fx" in task.asset_panels[asset].lower() and np.any(values[:, columns] <= 0):
            raise ValueError(f"FX level forecast is non-positive for {asset}")
    if np.any(np.ptp(values, axis=0) <= np.maximum(1e-12, np.abs(values).mean(axis=0) * 1e-12)):
        raise ValueError("at least one marginal distribution is effectively constant")


def validate_forecast_frame(frame: pd.DataFrame, task: TaskSpecLike, n_draws: int) -> None:
    if tuple(frame.columns) != ("draw", "asset", "horizon", "value"):
        raise ValueError("forecast columns do not match Track 2 contract")
    if frame.duplicated(["draw", "asset", "horizon"]).any():
        raise ValueError("duplicate draw/asset/horizon rows")
    if frame["draw"].drop_duplicates().tolist() != list(range(n_draws)):
        raise ValueError("draw indices must be contiguous from zero")
    if set(frame["asset"]) != set(task.assets) or set(frame["horizon"]) != set(task.horizons):
        raise ValueError("output grid differs from task")
    expected_cells = {(a, h) for a in task.assets for h in task.horizons}
    for _, group in frame.groupby("draw", sort=False):
        if set(zip(group["asset"], group["horizon"])) != expected_cells:
            raise ValueError("a draw does not contain the complete grid")
    if frame["draw"].dtype != np.int32 or frame["horizon"].dtype != np.int32 or frame["value"].dtype != np.float64:
        raise ValueError("forecast dtypes do not match Track 2 contract")
