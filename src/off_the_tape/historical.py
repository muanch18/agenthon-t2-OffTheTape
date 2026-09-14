"""Build leakage-safe pseudo-cards from a fully observed historical panel.

The forecaster receives only rows dated on or before each as-of date. Outcomes are
constructed separately from later rows and never attached to the pseudo-card.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

TargetType = Literal["level", "log_return"]


@dataclass(frozen=True)
class PseudoCard:
    family: str
    panel_id: str
    asof: date
    panel: pd.DataFrame
    assets: tuple[str, ...]
    horizons: tuple[int, ...]
    target_type: TargetType
    target_dates: tuple[date, ...]

    @property
    def cells(self) -> tuple[tuple[str, int], ...]:
        """Use the Track 2 scorer's asset-major, horizon-minor order."""
        return tuple((asset, horizon) for asset in self.assets for horizon in self.horizons)


@dataclass(frozen=True)
class HistoricalCase:
    card: PseudoCard
    realized: np.ndarray


def _validated_panel(panel: pd.DataFrame, assets: tuple[str, ...]) -> pd.DataFrame:
    required = {"date", "asset", "value"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel needs columns {sorted(required)}")
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("assets must be nonempty and unique")
    frame = panel.loc[panel["asset"].isin(assets), ["date", "asset", "value"]].copy()
    if set(frame["asset"]) != set(assets):
        raise ValueError("panel is missing at least one requested asset")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or (frame["date"] != frame["date"].dt.normalize()).any():
        raise ValueError("panel dates must be non-null calendar dates")
    if frame["date"].dt.dayofweek.ge(5).any():
        raise ValueError("daily panel dates must be weekdays")
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("panel has duplicate date/asset rows")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("panel values must be finite")
    counts = frame.groupby("date")["asset"].nunique()
    if not counts.eq(len(assets)).all():
        raise ValueError("requested assets must have a value on every panel date")
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def historical_cases(
    panel: pd.DataFrame,
    *,
    family: str,
    panel_id: str,
    assets: Iterable[str],
    horizons: Iterable[int],
    target_type: TargetType,
    origins: Iterable[date | str],
) -> list[HistoricalCase]:
    """Construct cases at exact weekday business-day offsets.

    A horizon of h uses ``asof + pandas.BDay(h)``. Every requested origin and
    target date must exist in the panel; a holiday or incomplete backtest fails
    instead of silently shifting its target or changing its denominator.
    """
    asset_ids = tuple(assets)
    if family not in {"T2-F1", "T2-F2", "T2-F3", "T2-F4"}:
        raise ValueError("family must be a Track 2 family")
    if not panel_id:
        raise ValueError("panel_id must be nonempty")
    if "panel_id" in panel.columns and not panel["panel_id"].eq(panel_id).all():
        raise ValueError("panel contains a different panel_id")
    steps = tuple(horizons)
    if not steps or len(set(steps)) != len(steps) or any(
        isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in steps
    ):
        raise ValueError("horizons must be unique positive business-day counts")
    if target_type not in ("level", "log_return"):
        raise ValueError("target_type must be level or log_return")
    frame = _validated_panel(panel, asset_ids)
    wide = frame.pivot(index="date", columns="asset", values="value").loc[:, list(asset_ids)]
    dates = list(wide.index)
    positions = {timestamp.date(): i for i, timestamp in enumerate(dates)}
    cases: list[HistoricalCase] = []
    seen: set[date] = set()
    for origin in origins:
        asof = pd.Timestamp(origin).date()
        if asof in seen:
            raise ValueError(f"duplicate origin: {asof}")
        seen.add(asof)
        if asof not in positions:
            raise ValueError(f"origin is absent from panel: {asof}")
        i = positions[asof]
        future_dates = tuple((dates[i] + pd.offsets.BDay(h)).date() for h in steps)
        if any(target not in positions for target in future_dates):
            raise ValueError(f"panel lacks a requested business-day target for {asof}")
        history = frame.loc[frame["date"] <= dates[i]].copy().reset_index(drop=True)
        card = PseudoCard(family, panel_id, asof, history, asset_ids, steps, target_type, future_dates)
        outcome = []
        for asset in asset_ids:
            series = wide[asset].to_numpy(dtype=float)
            for target in future_dates:
                end = positions[target]
                if target_type == "level":
                    outcome.append(float(series[end]))
                else:
                    daily = series[i + 1 : end + 1]
                    if np.any(daily <= -1):
                        raise ValueError("log_return requires daily simple returns greater than -1")
                    outcome.append(float(np.log1p(daily).sum()))
        cases.append(HistoricalCase(card, np.asarray(outcome, dtype=float)))
    if not cases:
        raise ValueError("at least one origin is required")
    return cases


def evaluate(
    cases: Iterable[HistoricalCase],
    forecast: Callable[[PseudoCard], np.ndarray],
) -> list[dict[str, object]]:
    """Score draw matrices; results are raw, local, and never leaderboard scores."""
    from .scoring import raw_track2_score

    results = []
    for case in cases:
        draws = np.asarray(forecast(case.card), dtype=float)
        result = raw_track2_score(draws, case.realized)
        results.append({
            "family": case.card.family,
            "panel_id": case.card.panel_id,
            "asof": case.card.asof.isoformat(),
            "target_dates": [d.isoformat() for d in case.card.target_dates],
            "assets": list(case.card.assets),
            "horizons": list(case.card.horizons),
            "target_type": case.card.target_type,
            "n_draws": int(draws.shape[0]),
            "realized": case.realized.tolist(),
            **result,
        })
    if not results:
        raise ValueError("at least one case is required")
    return results
