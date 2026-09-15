"""Minimal joint empirical fallback used when PCA or validation fails."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .historical import PseudoCard
from .macro_conditioner import NumericContext


@dataclass(frozen=True)
class SafeForecast:
    paths: np.ndarray
    context: NumericContext


def safe_joint_paths(card: PseudoCard, modes: tuple[str, ...], n_draws: int, seed: int) -> SafeForecast:
    """Rank-couple empirical asset shocks; inject a tiny common floor if needed."""
    rng = np.random.default_rng(seed)
    pools: list[np.ndarray] = []
    origins = []
    scales = []
    for asset, mode in zip(card.assets, modes):
        sub = card.panel.loc[card.panel["asset"] == asset, ["date", "value"]].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.sort_values("date").drop_duplicates("date", keep="last")
        values = pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        values = values[valid]
        dates = sub.loc[valid, "date"].reset_index(drop=True)
        if not len(values):
            raise ValueError(f"no finite history for {asset}")
        origins.append(float(values[-1]))
        if mode == "simple_return":
            transformed = np.log1p(np.clip(values, -0.999999, None))
        elif mode == "log_price":
            positive = np.maximum(values, np.finfo(float).tiny)
            transformed = np.diff(np.log(positive))
        else:
            transformed = np.diff(values)
        if mode != "simple_return" and len(dates) > 1:
            gaps = dates.diff().dt.days.to_numpy()[1:]
            typical = np.nanmedian(gaps) if len(gaps) else 1
            transformed = transformed[gaps <= max(5, typical * 10)]
        transformed = transformed[np.isfinite(transformed)]
        if not len(transformed):
            transformed = np.zeros(1)
        transformed = transformed[-504:] - np.mean(transformed[-504:])
        scale = float(np.std(transformed, ddof=1)) if len(transformed) > 1 else 0.0
        floor = max(abs(origins[-1]) * 1e-5, 1e-6)
        if not np.isfinite(scale) or scale < floor:
            scale = floor
            transformed = np.array([-scale, scale])
        pools.append(transformed)
        scales.append(scale)
    horizon = max(card.horizons)
    uniforms = rng.random((n_draws, horizon))
    daily = np.empty((n_draws, horizon, len(card.assets)))
    for j, pool in enumerate(pools):
        ordered = np.sort(pool)
        indices = np.minimum((uniforms * len(ordered)).astype(int), len(ordered) - 1)
        daily[:, :, j] = ordered[indices]
    cumulative = np.cumsum(daily, axis=1)
    paths = np.empty_like(cumulative)
    for j, mode in enumerate(modes):
        if mode == "difference":
            paths[:, :, j] = origins[j] + cumulative[:, :, j]
        elif mode == "log_price":
            paths[:, :, j] = origins[j] * np.exp(np.clip(cumulative[:, :, j], -50, 50))
        else:
            paths[:, :, j] = cumulative[:, :, j]
    return SafeForecast(paths, NumericContext(modes, np.asarray(origins), np.asarray(scales)))
