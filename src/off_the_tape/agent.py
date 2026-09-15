"""One Track 2 agent: card parsing, numeric paths, macro state, output files."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import tempfile
from time import perf_counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .historical import PseudoCard, TargetType
from .candidate import CURRENT_CANDIDATE, CandidateConfig
from .macro_conditioner import ConditioningParams, MacroConditioner, NumericContext
from .macro_extractor import MacroExtraction, extract_macro_state
from .macro_llm import MacroModel
from .macro_state import MacroState
from .pca_joint import PCAJointForecaster
from .output_validation import validate_draw_matrix, validate_forecast_frame
from .safe_forecast import safe_joint_paths
from .seeds import SeedManager


@dataclass(frozen=True)
class TaskSpec:
    unit_id: str
    family: str
    asof: date
    assets: tuple[str, ...]
    horizons: tuple[int, ...]
    target_type: TargetType
    asset_panels: dict[str, str]

    @classmethod
    def from_card(cls, path: Path) -> TaskSpec:
        with Path(path).open("rb") as stream:
            card = tomllib.load(stream)
        assets = tuple(card["targets"]["asset_ids"])
        panels = {}
        for panel_id in card["panels"]["panel_ids"]:
            for asset in card["panels"][panel_id].get("asset_ids", ()):  # exemplar may omit mapping
                panels[asset] = panel_id
        if not all(asset in panels for asset in assets):
            raise ValueError("card must map each target asset to a panel")
        return cls(
            str(card["task"]["id"]), str(card["metadata"]["category"]),
            date.fromisoformat(card["provenance"]["data_cutoff"]), assets,
            tuple(int(h) for h in card["targets"]["horizons"]),
            card["targets"]["target_type"], panels,
        )


@dataclass(frozen=True)
class AgentForecast:
    task: TaskSpec
    draws: np.ndarray
    base_draws: np.ndarray
    macro: MacroExtraction
    diagnostics: dict[str, object]
    numeric_only: bool
    fallback_used: bool = False
    failure_category: str | None = None


@dataclass
class Track2Agent:
    params: ConditioningParams = field(default_factory=ConditioningParams)
    n_draws: int = 1000
    text_client: MacroModel | None = None
    candidate: CandidateConfig = CURRENT_CANDIDATE
    macro_cache_dir: Path | None = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "off_the_tape_macro_cache"
    )

    def forecast(
        self, task: TaskSpec, panels_path: Path, text_path: Path, asof: date,
        seed: int = 2026, numeric_only: bool | None = None,
        *, macro_override: MacroState | None = None,
        components: frozenset[str] = frozenset({"mean", "volatility", "skew", "shock", "joint"}),
    ) -> AgentForecast:
        started = perf_counter()
        if numeric_only is None:
            numeric_only = not self.candidate.text_conditioning
        if asof != task.asof:
            raise ValueError("as-of must match the trusted card cutoff")
        seeds = SeedManager(seed)
        base_paths, card, context, numeric_diag = self._numeric_paths(task, Path(panels_path), seeds)
        base_draws = MacroConditioner._result(base_paths, card, {}).draws
        try:
            validate_draw_matrix(base_draws, task, self.n_draws)
        except ValueError as exc:
            fallback = safe_joint_paths(card, context.asset_modes, self.n_draws,
                                        seeds.derive("final_numeric_fallback"))
            base_paths, context = fallback.paths, fallback.context
            base_draws = MacroConditioner._result(base_paths, card, {}).draws
            validate_draw_matrix(base_draws, task, self.n_draws)
            numeric_diag.update(numeric_model="safe_joint_fallback", fallback_used=True,
                                failure_category=classify_failure(exc),
                                fallback_reasons=numeric_diag.get("fallback_reasons", []) + [str(exc)])
        text_started = perf_counter()
        if numeric_only:
            macro = MacroExtraction(asof, MacroState.neutral(), (), (), "ablated")
        elif macro_override is not None:
            macro = MacroExtraction(asof, macro_override, (), (), "override")
        else:
            try:
                macro = extract_macro_state(
                    Path(text_path), asof=asof, family=task.family,
                    panel_id="+".join(dict.fromkeys(task.asset_panels[a] for a in task.assets)),
                    client=self.text_client,
                    cache_dir=self.macro_cache_dir,
                )
            except (OSError, ValueError, KeyError):
                macro = MacroExtraction(asof, MacroState.neutral(), (), (), "fallback", reason="text_unavailable")
        text_seconds = perf_counter() - text_started
        if numeric_only or macro.state == MacroState.neutral():
            return AgentForecast(task, base_draws, base_draws.copy(), macro,
                                 {"text_impact": 0.0, "numeric_model": numeric_diag["numeric_model"],
                                  **numeric_diag, "text_total_seconds": text_seconds,
                                  "model_calls": macro.model_calls,
                                  "input_tokens": macro.input_tokens,
                                  "output_tokens": macro.output_tokens,
                                  "llm_seconds": macro.latency_seconds,
                                  "text_selection_seconds": macro.selection_seconds,
                                  "llm_parsing_seconds": macro.parsing_seconds,
                                  "conditioning_seconds": 0.0, "total_seconds": perf_counter() - started},
                                 numeric_only, numeric_diag["fallback_used"], numeric_diag.get("failure_category"))
        condition_started = perf_counter()
        conditioned = MacroConditioner(self.params).condition(
            base_paths, macro.state, card, context, seeds.derive("conditioner"),
            mean="mean" in components, volatility="volatility" in components,
            skew="skew" in components, shock="shock" in components,
            joint="joint" in components,
        )
        try:
            validate_draw_matrix(conditioned.draws, task, self.n_draws)
        except ValueError:
            return AgentForecast(task, base_draws, base_draws.copy(), macro,
                                 {"text_impact": 0.0, **numeric_diag, "conditioning_fallback": True,
                                  "text_total_seconds": text_seconds,
                                  "model_calls": macro.model_calls,
                                  "input_tokens": macro.input_tokens,
                                  "output_tokens": macro.output_tokens,
                                  "llm_seconds": macro.latency_seconds,
                                  "text_selection_seconds": macro.selection_seconds,
                                  "llm_parsing_seconds": macro.parsing_seconds,
                                  "conditioning_seconds": perf_counter() - condition_started,
                                  "total_seconds": perf_counter() - started}, numeric_only, True,
                                 "forecast NaN")
        return AgentForecast(task, conditioned.draws, base_draws, macro,
                             {**numeric_diag, **conditioned.diagnostics,
                              "text_total_seconds": text_seconds,
                              "model_calls": macro.model_calls,
                              "input_tokens": macro.input_tokens,
                              "output_tokens": macro.output_tokens,
                              "llm_seconds": macro.latency_seconds,
                              "text_selection_seconds": macro.selection_seconds,
                              "llm_parsing_seconds": macro.parsing_seconds,
                              "conditioning_seconds": perf_counter() - condition_started,
                              "total_seconds": perf_counter() - started}, numeric_only,
                             numeric_diag["fallback_used"], numeric_diag.get("failure_category"))

    def _numeric_paths(self, task: TaskSpec, panels_path: Path, seeds: SeedManager) -> tuple[np.ndarray, PseudoCard, NumericContext, dict[str, object]]:
        if isinstance(seeds, int):  # Backward-compatible internal research API.
            seeds = SeedManager(seeds)
        loading_started = perf_counter()
        panel_dir = panels_path
        files = sorted(panel_dir.glob("*.parquet"))
        if not files:
            files = sorted(panel_dir.parent.glob("*.parquet"))
        paths: dict[str, np.ndarray] = {}
        origins: dict[str, float] = {}
        vol: dict[str, float] = {}
        modes: dict[str, str] = {}
        full_frame = []
        fit_seconds = 0.0
        simulation_seconds = 0.0
        fallback_used = False
        failure_categories = []
        fallback_reasons = []
        for panel_id in dict.fromkeys(task.asset_panels[a] for a in task.assets):
            match = next((path for path in files if path.stem == panel_id), None)
            if match is None:
                raise ValueError(f"missing panel {panel_id}")
            frame = pd.read_parquet(match)
            if "asset_id" in frame.columns and "asset" not in frame.columns:
                frame = frame.rename(columns={"asset_id": "asset"})
            if not {"date", "asset", "value"}.issubset(frame.columns):
                raise ValueError("panel needs date, asset, value")
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"])
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            if (frame["date"] > pd.Timestamp(task.asof)).any():
                frame = frame.loc[frame["date"] <= pd.Timestamp(task.asof)].copy()
            group_assets = tuple(a for a in task.assets if task.asset_panels[a] == panel_id)
            frame = frame.loc[frame["asset"].isin(group_assets), ["date", "asset", "value"]]
            frame = frame.sort_values(["date", "asset"]).drop_duplicates(["date", "asset"], keep="last")
            if frame.empty or set(frame["asset"]) != set(group_assets):
                raise ValueError("target asset absent from panel")
            full_frame.append(frame)
            mode = "simple_return" if task.target_type == "log_return" else (
                "log_price" if "fx" in panel_id.lower() else "difference"
            )
            local_card = PseudoCard(
                task.family, panel_id, task.asof, frame, group_assets, task.horizons,
                task.target_type, tuple((pd.Timestamp(task.asof) + pd.offsets.BDay(h)).date() for h in task.horizons),
            )
            wide = frame.pivot(index="date", columns="asset", values="value").sort_index()
            try:
                fit = PCAJointForecaster(
                    mode=mode, n_draws=self.n_draws,
                    seed=seeds.derive(f"pca:{panel_id}"),
                ).forecast(local_card)
                group_paths = fit.paths
                group_context = NumericContext(
                    tuple(mode for _ in group_assets),
                    np.array([float(wide[a].dropna().iloc[-1]) for a in group_assets]),
                    np.array([float(fit.diagnostics["historical_marginal_volatility"][a]) for a in group_assets]),
                )
                fit_seconds += float(fit.diagnostics.get("fit_seconds", 0.0))
                simulation_seconds += float(fit.diagnostics.get("simulation_seconds", 0.0))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                fallback_used = True
                failure_categories.append(classify_failure(exc))
                fallback_reasons.append(f"{type(exc).__name__}: {exc}")
                fallback = safe_joint_paths(
                    local_card, tuple(mode for _ in group_assets), self.n_draws,
                    seeds.derive(f"fallback:{panel_id}"),
                )
                group_paths, group_context = fallback.paths, fallback.context
            for j, asset in enumerate(group_assets):
                paths[asset] = group_paths[:, :, j]
                origins[asset] = float(group_context.origins[j])
                vol[asset] = float(group_context.historical_volatility[j])
                modes[asset] = mode
        joint_paths = np.stack([paths[asset] for asset in task.assets], axis=2)
        joint_card = PseudoCard(
            task.family, "+".join(dict.fromkeys(task.asset_panels[a] for a in task.assets)),
            task.asof, pd.concat(full_frame, ignore_index=True), task.assets,
            task.horizons, task.target_type,
            tuple((pd.Timestamp(task.asof) + pd.offsets.BDay(h)).date() for h in task.horizons),
        )
        context = NumericContext(tuple(modes[a] for a in task.assets),
                                 np.array([origins[a] for a in task.assets]),
                                 np.array([vol[a] for a in task.assets]))
        return joint_paths, joint_card, context, {
            "numeric_model": "safe_joint_fallback" if fallback_used else "PCAJointForecaster",
            "fallback_used": fallback_used,
            "failure_category": ",".join(sorted(set(failure_categories))) or None,
            "fallback_reasons": fallback_reasons,
            "input_loading_and_preprocessing_seconds": perf_counter() - loading_started,
            "numeric_fit_seconds": fit_seconds,
            "numeric_simulation_seconds": simulation_seconds,
        }

    @staticmethod
    def write(forecast: AgentForecast, out: Path) -> None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        task, draws = forecast.task, forecast.draws
        frame = pd.DataFrame({
            "draw": np.repeat(np.arange(len(draws), dtype=np.int32), draws.shape[1]),
            "asset": np.tile(np.repeat(task.assets, len(task.horizons)), len(draws)),
            "horizon": np.tile(np.tile(np.asarray(task.horizons, dtype=np.int32), len(task.assets)), len(draws)),
            "value": draws.reshape(-1).astype(np.float64),
        })
        validate_draw_matrix(draws, task, len(draws))
        validate_forecast_frame(frame, task, len(draws))
        frame.to_parquet(out, index=False)
        meta = {"unit_id": task.unit_id, "asof": task.asof.isoformat(),
                "asset_ids": list(task.assets), "horizons": list(task.horizons),
                "representation": "samples", "n_draws": len(draws)}
        (out.parent / "forecast_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        active = not forecast.numeric_only and forecast.diagnostics.get("text_impact", 0) > 0
        signals = ", ".join(f"{item.signal}={item.value:.2f}" if isinstance(item.value, float)
                            else f"{item.signal}={item.value}" for item in forecast.macro.evidence[:6]) or "none"
        rationale = (
            f"# Forecast for {task.unit_id}\n\n"
            f"The numeric base is a PCA joint path model with paired historical innovations and {len(draws)} draws. "
            f"Text conditioning was {'active' if active else 'inactive'}. "
            f"Extracted macro signals: {signals}. "
            f"When active, bounded changes to daily means, dispersion, shared factors, skew, and shock tails "
            f"depend on card family and confidence. "
            f"Text status: {forecast.macro.source}{' (' + forecast.macro.reason + ')' if forecast.macro.reason else ''}.\n"
        )
        (out.parent / "forecast_rationale.md").write_text(rationale, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Track 2 forecasting agent")
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--asof", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--text-conditioned", action="store_true")
    args = parser.parse_args((argv or sys.argv[1:])[1:] if (argv or sys.argv[1:])[:1] == ["forecast"] else argv)
    unit = args.panels.parent if args.panels.name == "panels" else args.panels
    task = TaskSpec.from_card(unit / "card.toml")
    if args.numeric_only and args.text_conditioned:
        parser.error("choose at most one of --numeric-only and --text-conditioned")
    numeric_only = False if args.text_conditioned else True
    result = Track2Agent().forecast(task, args.panels, args.text, args.asof, args.seed, numeric_only)
    Track2Agent.write(result, args.out)
    return 0


def classify_failure(exc: Exception) -> str:
    text = str(exc).casefold()
    if "complete" in text or "missing" in text or "absent" in text:
        return "sparse assets"
    if "history" in text or "observations" in text:
        return "short history"
    if "finite" in text or "nan" in text or "volatility" in text:
        return "forecast NaN"
    if "panel" in text:
        return "unsupported panel"
    if "target" in text:
        return "unsupported target type"
    if isinstance(exc, np.linalg.LinAlgError) or "singular" in text:
        return "singular covariance"
    return "numeric model"


if __name__ == "__main__":
    raise SystemExit(main())
