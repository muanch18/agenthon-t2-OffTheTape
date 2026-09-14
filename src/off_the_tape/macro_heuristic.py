"""Explicit offline keyword proxy for diagnostics when no model endpoint exists.

This is not the production LLM extractor and is never an automatic fallback.
It exercises selection, schema, provenance, comparison, and caching locally.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

from .macro_state import MacroState, SCHEMA_VERSION
from .text_selector import SelectedDocument

PHRASES = {
    "policy_stance": (
        ("restrictive", "tightening", "raise the target", "higher for longer", "hike"),
        ("easing", "accommodative", "lower the target", "reduce the target", "rate cut"),
    ),
    "policy_stance_change": (
        ("more restrictive", "further tightening", "raise the target", "higher for longer"),
        ("begin cutting", "lower the target", "reduce the target", "rate cut"),
    ),
    "inflation_pressure": (
        ("inflation remains elevated", "inflation is elevated", "price pressures", "persistent inflation"),
        ("disinflation", "inflation has eased", "inflation declined", "inflation fell"),
    ),
    "inflation_change": (
        ("inflation accelerated", "inflation increased", "price pressures intensified"),
        ("inflation slowed", "inflation eased", "inflation declined", "inflation fell"),
    ),
    "growth_outlook": (
        ("solid growth", "economic expansion", "economy remains resilient", "growth strengthened"),
        ("downside risks to growth", "economic slowdown", "recession", "growth weakened"),
    ),
    "growth_change": (
        ("growth strengthened", "growth accelerated", "activity picked up"),
        ("growth weakened", "growth slowed", "activity decelerated"),
    ),
    "labor_strength": (
        ("job gains remained strong", "labor market remains strong", "employment growth"),
        ("unemployment rose", "job gains slowed", "labor market weakened"),
    ),
    "policy_uncertainty": (
        ("uncertainty", "data dependent", "uncertain outlook", "risks have increased"), (),
    ),
    "market_stress": (
        ("financial market conditions have deteriorated", "credit crunch", "market stress", "liquidity strain", "financial crisis"), (),
    ),
    "positioning_crowding": (
        ("crowded", "record short", "record long", "speculative positions", "net short", "carry trade"), (),
    ),
    "shock_probability": (
        ("shock", "disruption", "crisis", "intervention", "abrupt", "sudden reversal"), (),
    ),
}


@dataclass(frozen=True)
class OfflineHeuristicModel:
    mode: str = "heuristic"

    def complete(
        self, *, system: str, user: str, documents: Sequence[SelectedDocument],
        model_name: str, seed: int
    ) -> str:
        defaults = MacroState.neutral().to_dict()
        values = dict(defaults)
        support: dict[str, list[str]] = {}
        for field, (positive, negative) in PHRASES.items():
            signed = 0.0
            cited = []
            for document in documents:
                text = document.excerpt.casefold()
                plus = sum(min(2, text.count(phrase)) for phrase in positive)
                minus = sum(min(2, text.count(phrase)) for phrase in negative)
                if plus or minus:
                    cited.append(document.doc_id)
                signed += plus - minus
            if field in ("policy_uncertainty", "market_stress", "positioning_crowding", "shock_probability"):
                values[field] = min(1.0, defaults[field] + 0.12 * max(0, signed))
            else:
                values[field] = math.tanh(0.20 * signed)
            support[field] = cited

        # Derived macro distribution attributes use the same cited text signals.
        stress = values["market_stress"]
        uncertainty = values["policy_uncertainty"] - defaults["policy_uncertainty"]
        shock = values["shock_probability"]
        values["shock_probability"] = min(1.0, shock + 0.20 * stress)
        support["shock_probability"] = list(dict.fromkeys(
            support["shock_probability"] + support["market_stress"]
        ))
        values["expected_volatility_change"] = min(1.0, 0.65 * stress + 0.35 * uncertainty + 0.35 * shock)
        support["expected_volatility_change"] = list(dict.fromkeys(
            support["market_stress"] + support["policy_uncertainty"] + support["shock_probability"]
        ))
        values["expected_skew"] = max(-1.0, min(1.0, 0.45 * values["growth_outlook"] - 0.45 * stress))
        support["expected_skew"] = list(dict.fromkeys(
            support["growth_outlook"] + support["market_stress"]
        ))
        if values["shock_probability"] > 0.2:
            if values["growth_outlook"] < -0.2:
                values["shock_direction"] = "growth_negative"
            elif values["inflation_pressure"] > 0.2:
                values["shock_direction"] = "inflationary"
            else:
                values["shock_direction"] = "mixed"
            support["shock_direction"] = list(dict.fromkeys(
                support["shock_probability"] + support["growth_outlook"] + support["inflation_pressure"]
            ))
        values["confidence"] = min(0.65, 0.25 + 0.04 * len(documents))

        for field in list(values):
            if field in ("confidence", "shock_direction"):
                continue
            if abs(values[field] - defaults[field]) > 1e-9 and not support.get(field):
                values[field] = defaults[field]
        if values["shock_direction"] != "none" and not support.get("shock_direction"):
            values["shock_direction"] = "none"
        state = MacroState.from_mapping(values)
        evidence = [
            {
                "signal": field,
                "value": getattr(state, field),
                "supporting_docs": support[field][:4],
                "summary": "Offline lexical proxy matched macro language in these dated documents.",
            }
            for field in state.non_neutral_signals()
        ]
        return json.dumps({"schema_version": SCHEMA_VERSION, "state": state.to_dict(), "evidence": evidence})
