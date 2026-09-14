"""PCA/AR(1) joint path forecaster with paired moving-block innovations.

Daily transformed observations are decomposed into common PCA factors and
residuals. The AR(1) factors and residuals are shocked with the *same* sampled
historical blocks, retaining cross-asset extremes and some serial structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import ChangeMode
from .historical import PseudoCard


@dataclass(frozen=True)
class PCAForecast:
    draws: np.ndarray  # [draw, asset-major/horizon-minor cell]
    paths: np.ndarray  # [draw, simulated day, asset]
    daily_transforms: np.ndarray  # changes, log-price returns, or factor log returns
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class PCAJointForecaster:
    mode: ChangeMode
    n_factors: int = 2
    n_draws: int = 1000
    lookback: int = 504
    block_size: int = 5
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.mode not in ("difference", "log_price", "simple_return"):
            raise ValueError("mode must be difference, log_price, or simple_return")
        if self.n_factors < 1 or self.n_draws < 200 or self.lookback < 20 or self.block_size < 1:
            raise ValueError("n_factors >= 1, n_draws >= 200, lookback >= 20, block_size >= 1")

    def __call__(self, card: PseudoCard) -> np.ndarray:
        return self.forecast(card).draws

    def forecast(self, card: PseudoCard) -> PCAForecast:
        if self.mode == "simple_return" and card.target_type != "log_return":
            raise ValueError("simple_return mode requires a log_return target")
        if self.mode != "simple_return" and card.target_type != "level":
            raise ValueError("level targets require difference or log_price mode")
        if card.panel.empty or card.panel["date"].max().date() > card.asof:
            raise ValueError("forecast panel must be nonempty and end on or before as-of")

        wide = card.panel.pivot(index="date", columns="asset", values="value")
        wide = wide.loc[:, list(card.assets)].sort_index()
        values = wide.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("training panel must be complete and finite")
        if self.mode == "difference":
            transformed = np.diff(values, axis=0)
            transformed_dates = wide.index[1:]
        elif self.mode == "log_price":
            if np.any(values <= 0):
                raise ValueError("log_price mode requires positive levels")
            transformed = np.diff(np.log(values), axis=0)
            transformed_dates = wide.index[1:]
        else:
            if np.any(values <= -1):
                raise ValueError("simple_return mode requires daily returns above -1")
            transformed = np.log1p(values)
            transformed_dates = wide.index

        training = transformed[-self.lookback :]
        if len(training) < max(20, self.n_factors + 3):
            raise ValueError("not enough pre-as-of observations to fit PCA/AR(1)")
        center = training.mean(axis=0)
        scale = training.std(axis=0, ddof=1)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("each asset needs positive training-window volatility")
        standardized = (training - center) / scale

        # X_t = B f_t + e_t, in standardized daily-transformation units.
        _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
        k = min(self.n_factors, standardized.shape[1])
        loadings = vt[:k].T  # [asset, factor]
        factors = standardized @ loadings
        residuals = standardized - factors @ loadings.T
        explained = (singular_values[:k] ** 2 / np.sum(singular_values ** 2)).tolist()

        lagged, current = factors[:-1], factors[1:]
        x_mean, y_mean = lagged.mean(axis=0), current.mean(axis=0)
        denominator = np.sum((lagged - x_mean) ** 2, axis=0)
        slope = np.divide(
            np.sum((lagged - x_mean) * (current - y_mean), axis=0),
            denominator,
            out=np.zeros(k),
            where=denominator > 1e-12,
        )
        slope = np.clip(slope, -0.95, 0.95)
        intercept = y_mean - slope * x_mean
        factor_shocks = current - intercept - slope * lagged
        # Pair factor shocks with same-date residuals before block sampling.
        innovations = np.concatenate([factor_shocks, residuals[1:]], axis=1)
        innovations -= innovations.mean(axis=0)

        horizon = max(card.horizons)
        block = min(self.block_size, len(innovations))
        n_blocks = (horizon + block - 1) // block
        rng = np.random.default_rng(self.seed + card.asof.toordinal())
        starts = rng.integers(0, len(innovations) - block + 1, size=(self.n_draws, n_blocks))
        indices = (starts[:, :, None] + np.arange(block)).reshape(self.n_draws, -1)[:, :horizon]
        sampled = innovations[indices]

        factor_state = np.broadcast_to(factors[-1], (self.n_draws, k)).copy()
        daily = np.empty((self.n_draws, horizon, len(card.assets)), dtype=float)
        for day in range(horizon):
            factor_state = intercept + slope * factor_state + sampled[:, day, :k]
            standardized_day = factor_state @ loadings.T + sampled[:, day, k:]
            daily[:, day, :] = center + scale * standardized_day

        cumulative = np.cumsum(daily, axis=1)
        if self.mode == "difference":
            paths = values[-1][None, None, :] + cumulative
        elif self.mode == "log_price":
            paths = values[-1][None, None, :] * np.exp(cumulative)
        else:
            paths = cumulative
        if not np.isfinite(paths).all() or (self.mode == "log_price" and np.any(paths <= 0)):
            raise ValueError("simulation produced invalid reconstructed levels")
        draws = np.column_stack(
            [paths[:, h - 1, asset_index]
             for asset_index in range(len(card.assets)) for h in card.horizons]
        )

        residual_native = residuals * scale
        simulated_daily = daily.reshape(-1, len(card.assets))
        quantile_levels = (0.01, 0.05, 0.95, 0.99)
        diagnostics: dict[str, object] = {
            "asof": card.asof.isoformat(),
            "assets": list(card.assets),
            "horizons": list(card.horizons),
            "mode": self.mode,
            "training_start": transformed_dates[-len(training)].date().isoformat(),
            "training_end": transformed_dates[-1].date().isoformat(),
            "training_observations": len(training),
            "n_draws": self.n_draws,
            "innovation_block_size": block,
            "n_factors": k,
            "explained_variance_ratio": explained,
            "loading_matrix_standardized": loadings.tolist(),
            "ar1_slope": slope.tolist(),
            "residual_covariance": np.atleast_2d(np.cov(residual_native, rowvar=False)).tolist(),
            "historical_marginal_volatility": dict(zip(card.assets, scale.tolist())),
            "simulated_marginal_volatility": dict(zip(
                card.assets, simulated_daily.std(axis=0, ddof=1).tolist()
            )),
            "historical_correlation": _correlation(training).tolist(),
            "simulated_correlation": _correlation(simulated_daily).tolist(),
            "simulated_quantiles": [
                {
                    "asset": asset,
                    "horizon": h,
                    "q01": float(np.quantile(draws[:, column], quantile_levels[0])),
                    "q05": float(np.quantile(draws[:, column], quantile_levels[1])),
                    "q95": float(np.quantile(draws[:, column], quantile_levels[2])),
                    "q99": float(np.quantile(draws[:, column], quantile_levels[3])),
                }
                for column, (asset, h) in enumerate(card.cells)
            ],
        }
        return PCAForecast(draws, paths, daily, diagnostics)


def _correlation(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    volatility = np.sqrt(np.diag(covariance))
    denominator = volatility[:, None] * volatility[None, :]
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
    )
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)
