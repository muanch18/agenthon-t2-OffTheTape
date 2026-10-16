"""A simple level-space Gaussian random walk for historical comparison.

Uses trailing-300-row moments, date-aligned changes and coherent horizons.
This daily-level benchmark is not an exact clone of every official M0 mode.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .historical import PseudoCard


@dataclass(frozen=True)
class GaussianJointForecaster:
    n_draws: int = 1000
    lookback: int = 300
    seed: int = 2026

    def moments(self, card: PseudoCard):
        if card.target_type != "level":
            raise ValueError("this benchmark supports daily level targets only")
        if card.panel.date.max().date() > card.asof:
            raise ValueError("post-asof rows in model input")
        if self.n_draws < 200 or self.lookback < 3:
            raise ValueError("need >=200 draws and >=3 history rows")
        series = []
        anchors = []
        for asset in card.assets:
            raw = card.panel.loc[card.panel.asset == asset].set_index("date").value.sort_index()
            if raw.index.has_duplicates or len(raw) < 3 or not np.isfinite(raw).all():
                raise ValueError("invalid asset history")
            anchors.append(float(raw.iloc[-1]))
            raw = raw.iloc[-self.lookback:]
            spacing = raw.index.to_series().diff().dt.days
            steps = raw.diff().where(spacing <= max(10 * spacing.median(), 5))
            series.append(steps.rename(asset))
        aligned = pd.concat(series, axis=1).dropna()
        if len(aligned) < 3:
            raise ValueError("insufficient aligned changes")
        mu = aligned.mean().to_numpy()
        sigma = np.atleast_2d(np.cov(aligned.to_numpy(), rowvar=False))
        hs = np.tile(card.horizons, len(card.assets))
        ai = np.repeat(np.arange(len(card.assets)), len(card.horizons))
        mean = np.asarray(anchors)[ai] + hs * mu[ai]
        cov = np.minimum.outer(hs, hs) * sigma[ai[:, None], ai[None, :]]
        cov += 1.1e-9 * np.eye(len(mean))
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = np.diag(np.diag(cov))
        return mean, cov

    def __call__(self, card):
        mean, cov = self.moments(card)
        rng = np.random.default_rng(self.seed + card.asof.toordinal())
        return mean + rng.standard_normal((self.n_draws, len(mean))) @ np.linalg.cholesky(cov).T
