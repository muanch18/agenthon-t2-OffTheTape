"""Bounded, family-aware transformations of existing joint numeric paths.

All stochastic overlays are sampled once per draw and shared across assets and
horizons. The numeric forecaster is never refit or altered here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .historical import PseudoCard
from .macro_state import MacroState


@dataclass(frozen=True)
class ConditioningParams:
    f1_mean_scale: float = 0.08
    f1_vol_scale: float = 0.10
    f2_mean_scale: float = 0.18
    f2_vol_scale: float = 0.25
    f2_skew_scale: float = 0.15
    f3_factor_scale: float = 0.12
    f3_cov_scale: float = 0.18
    f4_shock_prob_scale: float = 0.30
    f4_shock_size_scale: float = 0.75
    f4_tail_df: float = 4.0


@dataclass(frozen=True)
class NumericContext:
    asset_modes: tuple[str, ...]  # difference, log_price, simple_return
    origins: np.ndarray  # last observed levels; unused for simple_return
    historical_volatility: np.ndarray  # native daily-change/log-return units


@dataclass(frozen=True)
class ConditionedForecast:
    paths: np.ndarray
    draws: np.ndarray
    diagnostics: dict[str, object]


def _exposure(asset: str, mode: str, state: MacroState) -> float:
    """Macro direction, not a forecast level. FX conventions follow asset IDs."""
    policy = state.policy_stance + 0.7 * state.policy_stance_change
    inflation = state.inflation_pressure + 0.6 * state.inflation_change
    growth = state.growth_outlook + 0.6 * state.growth_change
    stress = state.market_stress + state.shock_probability
    if mode == "difference":
        front = 1.25 if asset.endswith(("2Y", "5Y")) else 0.75
        return float(np.clip(front * policy + 0.6 * inflation + 0.35 * growth, -1, 1))
    if mode == "log_price":
        # USD-per-foreign pairs fall with USD hawkishness; JPY is JPY-per-USD.
        usd = -0.65 * policy - 0.45 * stress + 0.2 * growth
        return float(np.clip(-usd if asset in {"JPY", "CHF", "CAD"} else usd, -1, 1))
    factor = asset.upper()
    if factor in {"HML", "BAB"}:
        return float(np.clip(0.5 * inflation - 0.3 * stress, -1, 1))
    return float(np.clip(-0.6 * policy + 0.3 * growth - 0.8 * stress, -1, 1))


class MacroConditioner:
    def __init__(self, params: ConditioningParams | None = None):
        self.params = params or ConditioningParams()

    def condition(
        self, base_paths: np.ndarray, macro_state: MacroState,
        task: PseudoCard, numeric_context: NumericContext, seed: int,
        *, mean: bool = True, volatility: bool = True, skew: bool = True,
        shock: bool = True, joint: bool = True,
    ) -> ConditionedForecast:
        base = np.asarray(base_paths, dtype=float)
        if base.ndim != 3 or base.shape[1:] != (max(task.horizons), len(task.assets)):
            raise ValueError("base paths must cover every day and asset")
        modes = numeric_context.asset_modes
        origins = np.asarray(numeric_context.origins, dtype=float)
        sigma = np.asarray(numeric_context.historical_volatility, dtype=float)
        if len(modes) != len(task.assets) or origins.shape != sigma.shape or sigma.shape != (len(task.assets),):
            raise ValueError("numeric context does not match task assets")
        if not np.isfinite(base).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("numeric paths and volatilities must be finite and positive")
        if macro_state.confidence <= 0 or not any((mean, volatility, skew, shock, joint)):
            return self._result(base.copy(), task, {"text_impact": 0.0, "fallback": "neutral_or_ablated"})

        n, days, assets = base.shape
        daily = np.empty_like(base)
        for j, mode in enumerate(modes):
            if mode == "difference":
                daily[:, :, j] = np.diff(np.concatenate((np.full((n, 1), origins[j]), base[:, :, j]), axis=1), axis=1)
            elif mode == "log_price":
                if origins[j] <= 0 or np.any(base[:, :, j] <= 0):
                    raise ValueError("FX paths and origin must be positive")
                daily[:, :, j] = np.diff(np.log(np.concatenate((np.full((n, 1), origins[j]), base[:, :, j]), axis=1)), axis=1)
            elif mode == "simple_return":
                daily[:, :, j] = np.diff(np.concatenate((np.zeros((n, 1)), base[:, :, j]), axis=1), axis=1)
            else:
                raise ValueError(f"unsupported asset mode: {mode}")

        p = self.params
        conf = float(np.clip(macro_state.confidence, 0, 1))
        conflict = abs(macro_state.policy_stance + macro_state.policy_stance_change) < 0.15 and abs(macro_state.policy_stance) > 0.3
        gate = conf * (0.5 if conflict else 1.0)
        exposure = np.array([_exposure(asset, mode, macro_state) for asset, mode in zip(task.assets, modes)])
        family = task.family
        mean_scale = {"T2-F1": p.f1_mean_scale, "T2-F2": p.f2_mean_scale,
                      "T2-F3": p.f2_mean_scale * 0.7, "T2-F4": p.f2_mean_scale * 0.6}[family]
        vol_scale = p.f1_vol_scale if family == "T2-F1" else p.f2_vol_scale
        event = np.clip(macro_state.policy_uncertainty - 0.25 + macro_state.market_stress +
                        macro_state.shock_probability + abs(macro_state.expected_volatility_change), 0, 2)
        centered = daily - daily.mean(axis=(0, 1), keepdims=True)
        mean_shift = np.zeros(assets)
        vol_multiplier = 1.0
        if mean:
            mean_shift = np.clip(mean_scale * gate * exposure * sigma / np.sqrt(days),
                                 -0.35 * sigma / np.sqrt(days), 0.35 * sigma / np.sqrt(days))
            daily += mean_shift[None, None, :]
        if volatility:
            vol_multiplier = float(np.clip(1 + vol_scale * gate * event, 0.85, 1.4))
            daily += (vol_multiplier - 1) * centered
        if joint and family == "T2-F3" and assets > 1:
            # One standardized shared factor innovation per draw/day; no asset-wise sampling.
            common = (centered / sigma[None, None, :]).mean(axis=2)
            common -= common.mean(axis=(0, 1))
            common = np.clip(common, -4, 4)
            factor_scale = min(0.25, p.f3_factor_scale * gate * min(1, np.max(np.abs(exposure)) + event))
            cov_scale = min(0.35, p.f3_cov_scale * gate * event)
            daily += (factor_scale + cov_scale) * common[:, :, None] * sigma[None, None, :] * np.where(exposure == 0, 1, np.sign(exposure))[None, None, :]
        else:
            factor_scale = cov_scale = 0.0
        skew_scale = 0.0
        if skew and family != "T2-F1":
            skew_scale = float(min(0.25, p.f2_skew_scale * gate * abs(macro_state.expected_skew)))
            direction = np.sign(macro_state.expected_skew) * np.where(exposure == 0, 1, np.sign(exposure))
            daily += skew_scale * np.maximum(centered * direction[None, None, :], 0) * direction[None, None, :]

        shock_probability = 0.0
        shocked = 0
        if shock and family == "T2-F4":
            shock_probability = float(min(0.20, p.f4_shock_prob_scale * gate *
                                          (macro_state.shock_probability + 0.3 * macro_state.market_stress +
                                           0.2 * max(0, macro_state.policy_uncertainty - 0.25))))
            rng = np.random.default_rng(seed + task.asof.toordinal() + 91)
            mask = rng.random(n) < shock_probability
            shocked = int(mask.sum())
            if shocked:
                tdraw = np.minimum(np.abs(rng.standard_t(max(2.1, p.f4_tail_df), size=shocked)), 8.0)
                sign = np.where(exposure == 0, np.sign(macro_state.expected_skew), np.sign(exposure))
                if not np.any(sign):
                    sign = np.ones(assets)
                # Scale to the requested horizon, then cap each scenario to a
                # modest number of historical horizon-volatility units.
                size = np.minimum(p.f4_shock_size_scale * gate * tdraw, 2.0)
                displacement = size[:, None] * sigma[None, :] * np.sqrt(days) * sign[None, :]
                days_at = rng.integers(0, min(days, 10), size=shocked)
                daily[np.where(mask)[0], days_at, :] += displacement

        cumulative = np.cumsum(daily, axis=1)
        paths = np.empty_like(base)
        for j, mode in enumerate(modes):
            if mode == "difference":
                paths[:, :, j] = origins[j] + cumulative[:, :, j]
            elif mode == "log_price":
                paths[:, :, j] = origins[j] * np.exp(np.clip(cumulative[:, :, j], -50, 50))
            else:
                paths[:, :, j] = cumulative[:, :, j]
        return self._result(paths, task, {
            "text_impact": gate, "mean_shift_daily": dict(zip(task.assets, mean_shift.tolist())),
            "vol_multiplier": vol_multiplier, "factor_scale": factor_scale,
            "cov_scale": cov_scale, "skew_scale": skew_scale,
            "shock_probability": shock_probability, "shocked_draws": shocked,
            "confidence": conf, "conflict_gate": bool(conflict),
        })

    @staticmethod
    def _result(paths: np.ndarray, task: PseudoCard, diagnostics: dict[str, object]) -> ConditionedForecast:
        draws = np.column_stack([paths[:, h - 1, j] for j in range(len(task.assets)) for h in task.horizons])
        if not np.isfinite(draws).all():
            raise ValueError("conditioned draws are non-finite")
        return ConditionedForecast(paths, draws, diagnostics)
