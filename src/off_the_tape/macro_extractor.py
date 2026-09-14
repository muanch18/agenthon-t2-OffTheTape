"""Timestamp-safe, cacheable macro extraction parallel to numeric forecasting."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

from .macro_cache import MacroCache, cache_key
from .macro_llm import PROMPT_VERSION, SYSTEM_PROMPT, EndpointModel, MacroModel, build_user_prompt
from .macro_state import (
    DIRECTIONAL, SCHEMA_VERSION, MacroDelta, MacroState,
    SignalEvidence, compare_states,
)
from .text_corpus import load_corpus
from .text_selector import SELECTOR_VERSION, DocumentSelector


@dataclass(frozen=True)
class MacroExtraction:
    asof: date
    state: MacroState
    evidence: tuple[SignalEvidence, ...]
    selected_doc_ids: tuple[str, ...]
    source: str  # llm, heuristic, cache, fallback, ablated
    cache_key: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["asof"] = self.asof.isoformat()
        return result


@dataclass(frozen=True)
class PriorComparison:
    current: MacroExtraction
    previous: MacroExtraction
    window_days: int
    delta: MacroDelta


def _validate_payload(payload: Mapping[str, Any], doc_ids: set[str]) -> tuple[MacroState, tuple[SignalEvidence, ...]]:
    if set(payload) != {"schema_version", "state", "evidence"} or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("model output has wrong top-level schema")
    if not isinstance(payload["state"], dict) or not isinstance(payload["evidence"], list):
        raise ValueError("state must be an object and evidence an array")
    state = MacroState.from_mapping(payload["state"])
    evidence: list[SignalEvidence] = []
    used_signals = set()
    for item in payload["evidence"]:
        if not isinstance(item, dict) or set(item) != {"signal", "value", "supporting_docs", "summary"}:
            raise ValueError("invalid evidence record")
        signal = item["signal"]
        if signal not in state.to_dict() or signal == "confidence" or signal in used_signals:
            raise ValueError("invalid or duplicate evidence signal")
        used_signals.add(signal)
        cited = item["supporting_docs"]
        if not isinstance(cited, list) or not cited or any(not isinstance(x, str) for x in cited):
            raise ValueError("evidence needs supporting document IDs")
        if not set(cited).issubset(doc_ids):
            raise ValueError("evidence cites an unselected document")
        summary = item["summary"]
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 500:
            raise ValueError("evidence needs a short summary")
        observed = getattr(state, signal)
        reported = item["value"]
        if signal == "shock_direction":
            if reported != observed:
                raise ValueError("evidence value disagrees with state")
        else:
            if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(reported):
                raise ValueError("evidence value must be numeric")
            low, high = (-1, 1) if signal in DIRECTIONAL else (0, 1)
            reported = max(low, min(high, reported))
            if abs(reported - observed) > 1e-6:
                raise ValueError("evidence value disagrees with state")
        evidence.append(SignalEvidence(signal, observed, tuple(dict.fromkeys(cited)), summary.strip()))
    if not set(state.non_neutral_signals()).issubset(used_signals):
        raise ValueError("every non-neutral signal needs evidence")
    return state, tuple(evidence)


def _parse_output(raw: str, doc_ids: set[str]) -> tuple[MacroState, tuple[SignalEvidence, ...]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    return _validate_payload(payload, doc_ids)


def extract_macro_state(
    unit_dir: Path,
    *,
    asof: date,
    family: str | None = None,
    panel_id: str | None = None,
    text_enabled: bool = True,
    client: MacroModel | None = None,
    model_name: str | None = None,
    selector: DocumentSelector | None = None,
    cache_dir: Path | None = Path(".cache/macro_state"),
    seed: int = 2026,
) -> MacroExtraction:
    """Extract macro-only signals; invalid/unavailable model output is neutral."""
    if not text_enabled:
        return MacroExtraction(asof, MacroState.neutral(), (), (), "ablated")
    corpus = load_corpus(Path(unit_dir), asof)
    chosen_selector = selector or DocumentSelector()
    selected = chosen_selector.select(corpus, family=family, panel_id=panel_id)
    ids = tuple(doc.doc_id for doc in selected)
    if not selected:
        return MacroExtraction(asof, MacroState.neutral(), (), (), "fallback", reason="no_eligible_documents")
    model = client or EndpointModel.from_environment()
    name = model_name or os.environ.get("MODEL_NAME") or "unspecified"
    if model is None:
        return MacroExtraction(asof, MacroState.neutral(), (), ids, "fallback", reason="no_model_endpoint")
    key = cache_key({
        "information_hash": corpus.information_hash,
        "selected": [(doc.doc_id, doc.sha256, doc.excerpt) for doc in selected],
        "asof": asof.isoformat(),
        "family": family,
        "panel_id": panel_id,
        "model_name": name,
        "backend": model.mode,
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "selector": asdict(chosen_selector),
    })
    cache = MacroCache(Path(cache_dir)) if cache_dir is not None else None
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            try:
                state, evidence = _validate_payload(cached, set(ids))
                return MacroExtraction(asof, state, evidence, ids, "cache", key)
            except (TypeError, ValueError):
                pass
    user_prompt = build_user_prompt(
        asof=asof.isoformat(), family=family, panel_id=panel_id, documents=selected
    )
    try:
        raw = model.complete(
            system=SYSTEM_PROMPT, user=user_prompt, documents=selected,
            model_name=name, seed=seed,
        )
        state, evidence = _parse_output(raw, set(ids))
    except Exception:
        return MacroExtraction(asof, MacroState.neutral(), (), ids, "fallback", key, "invalid_or_failed_model_output")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": state.to_dict(),
        "evidence": [item.to_dict() for item in evidence],
    }
    if cache is not None:
        try:
            cache.put(key, payload)
        except OSError:
            pass  # A read-only inference mount must not make extraction fail.
    return MacroExtraction(asof, state, evidence, ids, model.mode, key)


def extract_with_prior(
    unit_dir: Path, *, asof: date, window_days: int, **kwargs: Any
) -> PriorComparison:
    if window_days < 1:
        raise ValueError("window_days must be positive")
    current = extract_macro_state(unit_dir, asof=asof, **kwargs)
    previous = extract_macro_state(unit_dir, asof=asof - timedelta(days=window_days), **kwargs)
    return PriorComparison(current, previous, window_days, compare_states(current.state, previous.state))
