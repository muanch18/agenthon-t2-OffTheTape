"""A deliberately simple joint-row bootstrap for local Track 2 backtests.

Each simulated day samples one whole historical cross-asset row. This preserves
same-day dependence while making no claim about serial dependence or regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .historical import PseudoCard

ChangeMode = Literal["difference", "log_price", "simple_return"]


@dataclass(frozen=True)
class JointRowBootstrap:
    mode: ChangeMode
    n_draws: int = 1000
    lookback: int = 504
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.mode not in ("difference", "log_price", "simple_return"):
            raise ValueError("mode must be difference, log_price, or simple_return")
        if self.n_draws < 200 or self.lookback < 2:
            raise ValueError("n_draws must be at least 200 and lookback at least 2")

    def __call__(self, card: PseudoCard) -> np.ndarray:
        if self.mode == "simple_return" and card.target_type != "log_return":
            raise ValueError("simple_return mode requires a log_return target")
        if self.mode != "simple_return" and card.target_type != "level":
            raise ValueError("level targets require difference or log_price mode")
        if card.panel["date"].max().date() > card.asof:
            raise ValueError("forecast panel contains post-as-of rows")

        wide = card.panel.pivot(index="date", columns="asset", values="value")
        wide = wide.loc[:, list(card.assets)].sort_index()
        values = wide.to_numpy(dtype=float)
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("at least three complete finite historical dates are required")

        if self.mode == "difference":
            historical_rows = np.diff(values, axis=0)
        elif self.mode == "log_price":
            if np.any(values <= 0):
                raise ValueError("log_price mode requires positive levels")
            historical_rows = np.diff(np.log(values), axis=0)
        else:
            if np.any(values <= -1):
                raise ValueError("simple_return mode requires daily returns above -1")
            historical_rows = np.log1p(values)

        pool = historical_rows[-self.lookback :]
        rng = np.random.default_rng(self.seed + card.asof.toordinal())
        row_indices = rng.integers(0, len(pool), size=(self.n_draws, max(card.horizons)))
        cumulative = np.cumsum(pool[row_indices], axis=1)
        if self.mode == "difference":
            paths = values[-1][None, None, :] + cumulative
        elif self.mode == "log_price":
            paths = values[-1][None, None, :] * np.exp(cumulative)
        else:
            paths = cumulative
        return np.column_stack(
            [paths[:, horizon - 1, asset_index]
             for asset_index in range(len(card.assets)) for horizon in card.horizons]
        )
