"""Typed macro-state variables, evidence, and prior-state differences.

Positive policy means hawkish; positive inflation means upward price pressure;
positive growth/labor means strength. Positive volatility means *more* macro
uncertainty. Skew describes macro outcomes, never an asset return direction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

SCHEMA_VERSION = "1"
DIRECTIONAL = (
    "policy_stance", "policy_stance_change", "inflation_pressure", "inflation_change",
    "growth_outlook", "growth_change", "labor_strength", "expected_volatility_change",
    "expected_skew",
)
UNIT_INTERVAL = (
    "policy_uncertainty", "market_stress", "positioning_crowding",
    "shock_probability", "confidence",
)
SHOCK_DIRECTIONS = frozenset({
    "none", "inflationary", "disinflationary", "growth_positive", "growth_negative", "mixed"
})


@dataclass(frozen=True)
class MacroState:
    policy_stance: float = 0.0
    policy_stance_change: float = 0.0
    inflation_pressure: float = 0.0
    inflation_change: float = 0.0
    growth_outlook: float = 0.0
    growth_change: float = 0.0
    labor_strength: float = 0.0
    policy_uncertainty: float = 0.25
    market_stress: float = 0.0
    positioning_crowding: float = 0.0
    shock_probability: float = 0.0
    shock_direction: str = "none"
    expected_volatility_change: float = 0.0
    expected_skew: float = 0.0
    confidence: float = 0.0

    @classmethod
    def neutral(cls) -> MacroState:
        return cls()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MacroState:
        required = set(DIRECTIONAL) | set(UNIT_INTERVAL) | {"shock_direction"}
        if set(value) != required:
            raise ValueError("macro state must contain exactly the schema fields")
        parsed = {}
        for field in DIRECTIONAL + UNIT_INTERVAL:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                raise ValueError(f"{field} must be a finite number")
            low, high = (-1.0, 1.0) if field in DIRECTIONAL else (0.0, 1.0)
            parsed[field] = float(max(low, min(high, raw)))
        direction = value["shock_direction"]
        if not isinstance(direction, str) or direction not in SHOCK_DIRECTIONS:
            raise ValueError("invalid macro shock_direction")
        return cls(**parsed, shock_direction=direction)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def non_neutral_signals(self) -> tuple[str, ...]:
        baseline = self.neutral()
        fields = DIRECTIONAL + tuple(field for field in UNIT_INTERVAL if field != "confidence")
        changed = [
            field for field in fields
            if abs(getattr(self, field) - getattr(baseline, field)) > 1e-9
        ]
        if self.shock_direction != "none":
            changed.append("shock_direction")
        return tuple(changed)


@dataclass(frozen=True)
class SignalEvidence:
    signal: str
    value: float | str
    supporting_docs: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MacroDelta:
    policy_stance_delta: float
    inflation_pressure_delta: float
    growth_outlook_delta: float
    policy_uncertainty_delta: float
    shock_probability_delta: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compare_states(current: MacroState, previous: MacroState) -> MacroDelta:
    """Compare state *levels*; this differs from a document's own change language."""
    return MacroDelta(
        current.policy_stance - previous.policy_stance,
        current.inflation_pressure - previous.inflation_pressure,
        current.growth_outlook - previous.growth_outlook,
        current.policy_uncertainty - previous.policy_uncertainty,
        current.shock_probability - previous.shock_probability,
    )
