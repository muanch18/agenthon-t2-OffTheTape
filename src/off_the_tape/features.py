"""Shared, cutoff-safe panel snapshots and deterministic numeric features.

Snapshots are scoped to one forecast input. There is no global cache or state
shared across competition units. NumPy results are read-only; frame accessors
return copies so callers cannot silently invalidate cached transformations.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

FEATURE_VERSION = "1"
Transform = Literal["difference", "log_price", "simple_return"]


def _readonly(array):
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TransformedHistory:
    assets: tuple[str, ...]
    asof: date
    mode: Transform
    values: np.ndarray
    changes: np.ndarray
    dates: pd.DatetimeIndex
    version: str = FEATURE_VERSION


@dataclass(frozen=True)
class VolatilityFeatures:
    assets: tuple[str, ...]
    asof: date
    mode: Transform
    lookback_changes: int
    observations: int
    recent_changes: int
    half_life: float
    historical_sd: np.ndarray
    recent_sd: np.ndarray
    ewma_sd: np.ndarray
    version: str = FEATURE_VERSION

    def variance_blend(self, *, source="ewma", recent_weight=0.5):
        if source not in {"recent", "ewma"} or not np.isfinite(recent_weight) or not 0 <= recent_weight <= 1:
            raise ValueError("invalid volatility blend")
        recent = self.recent_sd if source == "recent" else self.ewma_sd
        return _readonly(np.sqrt((1-recent_weight)*self.historical_sd**2+recent_weight*recent**2))


@dataclass(frozen=True, init=False, eq=False)
class FeatureLayer:
    def __init__(self, panel, *, assets, asof, future_policy="reject", duplicate_policy="reject"):
        if future_policy not in {"filter", "reject"} or duplicate_policy not in {"last", "reject"}:
            raise ValueError("invalid panel policy")
        cutoff = pd.Timestamp(asof)
        if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
            raise ValueError("as-of must be a non-null timezone-naive calendar date")
        object.__setattr__(self, "assets", tuple(assets))
        object.__setattr__(self, "asof", cutoff.date())
        if not self.assets or len(set(self.assets)) != len(self.assets):
            raise ValueError("assets must be nonempty and unique")
        frame = panel.copy(deep=True)
        if "asset_id" in frame.columns and "asset" not in frame.columns:
            frame = frame.rename(columns={"asset_id": "asset"})
        if not {"date", "asset", "value"}.issubset(frame.columns):
            raise ValueError("panel needs date, asset, value")
        frame["date"] = pd.to_datetime(frame["date"])
        if frame["date"].isna().any():
            raise ValueError("panel dates must be non-null")
        future = frame["date"] > cutoff
        if future.any() and future_policy == "reject":
            raise ValueError("forecast panel must end on or before as-of")
        frame = frame.loc[~future & frame["asset"].isin(self.assets), ["date", "asset", "value"]].copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.sort_values(["date", "asset"])
        if duplicate_policy == "reject" and frame.duplicated(["date", "asset"]).any():
            raise ValueError("duplicate date/asset rows")
        frame = frame.drop_duplicates(["date", "asset"], keep="last")
        if frame.empty or set(frame["asset"]) != set(self.assets):
            raise ValueError("target asset absent from panel")
        object.__setattr__(self, "_frame", frame)
        # Preserve missing cells; a caller's fallback must remain able to see gaps.
        object.__setattr__(self, "_wide", frame.pivot(index="date", columns="asset", values="value").sort_index().loc[:, list(self.assets)])
        object.__setattr__(self, "_transforms", {})
        object.__setattr__(self, "_volatility", {})

    @property
    def frame(self):
        return self._frame.copy(deep=True)

    @property
    def wide(self):
        return self._wide.copy(deep=True)

    def matches(self, panel, assets, asof):
        return self.assets == tuple(assets) and self.asof == asof and self._frame.equals(panel)

    def history(self, mode: Transform):
        if mode not in {"difference", "log_price", "simple_return"}:
            raise ValueError("unsupported daily transformation")
        if mode in self._transforms:
            return self._transforms[mode]
        values = self._wide.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("training panel must be complete and finite")
        if mode == "difference":
            changes, dates = np.diff(values, axis=0), self._wide.index[1:]
        elif mode == "log_price":
            if np.any(values <= 0):
                raise ValueError("log_price mode requires positive levels")
            changes, dates = np.diff(np.log(values), axis=0), self._wide.index[1:]
        else:
            if np.any(values <= -1):
                raise ValueError("simple_return mode requires daily returns above -1")
            changes, dates = np.log1p(values), self._wide.index
        result = TransformedHistory(self.assets, self.asof, mode, _readonly(values), _readonly(changes), dates.copy())
        self._transforms[mode] = result
        return result

    def volatility(self, mode, *, lookback_changes=504, recent_changes=60, half_life=20.0):
        if (isinstance(lookback_changes, bool) or not isinstance(lookback_changes, int)
                or isinstance(recent_changes, bool) or not isinstance(recent_changes, int)
                or lookback_changes < recent_changes or recent_changes < 2
                or not np.isfinite(half_life) or half_life <= 0):
            raise ValueError("invalid volatility windows or half-life")
        key = (mode, lookback_changes, recent_changes, half_life)
        if key in self._volatility:
            return self._volatility[key]
        pool = self.history(mode).changes[-lookback_changes:]
        if len(pool) < recent_changes:
            raise ValueError("insufficient history for requested recent volatility")
        w = 0.5**(np.arange(len(pool))[::-1]/half_life)
        w /= w.sum()
        denominator = 1-np.sum(w*w)
        if denominator <= 0:
            raise ValueError("half-life gives insufficient effective observations")
        mean = np.sum(w[:, None]*pool, axis=0)
        variance = np.sum(w[:, None]*(pool-mean)**2, axis=0)/denominator
        result = VolatilityFeatures(self.assets, self.asof, mode, lookback_changes, len(pool),
                                    recent_changes, half_life, _readonly(pool.std(axis=0, ddof=1)),
                                    _readonly(pool[-recent_changes:].std(axis=0, ddof=1)),
                                    _readonly(np.sqrt(variance)))
        self._volatility[key] = result
        return result
