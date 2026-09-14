"""Chronological pseudo-card ablations; never use practice-card sealed outcomes.

Candidate units are chosen mechanically by family and era. Each retrospective
origin is 30 weekdays before its unit cutoff, with 10/21-weekday outcomes that
remain inside that same panel. This is a sparse, curated-text proxy benchmark,
not a leaderboard estimate. Only the pre-2018 train split selects strength.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from off_the_tape.agent import TaskSpec, Track2Agent
from off_the_tape.macro_conditioner import ConditioningParams, MacroConditioner
from off_the_tape.macro_heuristic import OfflineHeuristicModel
from off_the_tape.scoring import raw_track2_score

ABLATIONS = {
    "A numeric": frozenset(),
    "B mean": frozenset({"mean"}),
    "C +vol": frozenset({"mean", "volatility"}),
    "D +skew": frozenset({"mean", "volatility", "skew"}),
    "E +shock": frozenset({"mean", "volatility", "skew", "shock"}),
    "F full": frozenset({"mean", "volatility", "skew", "shock", "joint"}),
    "G neutral": frozenset({"mean", "volatility", "skew", "shock", "joint"}),
}
SPLITS = {"train": (None, 2018), "validation": (2018, 2022), "heldout": (2022, None)}


def _strength(params: ConditioningParams, multiplier: float) -> ConditioningParams:
    values = asdict(params)
    for key in values:
        if key != "f4_tail_df":
            values[key] *= multiplier
    return ConditioningParams(**values)


def _case(unit: Path, task: TaskSpec) -> tuple[TaskSpec, np.ndarray] | None:
    origin = (pd.Timestamp(task.asof) - pd.offsets.BDay(30)).date()
    horizons = (10, 21)
    target_dates = [(pd.Timestamp(origin) + pd.offsets.BDay(h)).date() for h in horizons]
    if target_dates[-1] > task.asof:
        return None
    realized = []
    for asset in task.assets:
        panel = task.asset_panels[asset]
        file = unit / f"{panel}.parquet"
        if not file.is_file():
            file = unit / "panels" / f"{panel}.parquet"
        frame = pd.read_parquet(file)
        col = "asset" if "asset" in frame.columns else "asset_id"
        frame = frame.loc[frame[col] == asset, ["date", "value"]].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        series = frame.set_index("date")["value"]
        if origin not in series.index or any(d not in series.index for d in target_dates):
            return None
        if task.target_type == "log_return":
            for end in target_dates:
                days = series.loc[(series.index > origin) & (series.index <= end)].to_numpy(dtype=float)
                if np.any(days <= -1):
                    return None
                realized.append(float(np.log1p(days).sum()))
        else:
            realized.extend(float(series.loc[d]) for d in target_dates)
    return replace(task, asof=origin, horizons=horizons), np.array(realized)


def _selected_units(root: Path) -> list[tuple[str, Path, TaskSpec, np.ndarray]]:
    catalog: dict[tuple[str, str], list[tuple[Path, TaskSpec, np.ndarray]]] = {}
    for unit in sorted((root / "units").iterdir()):
        if not (unit / "card.toml").is_file():
            continue
        try:
            task = TaskSpec.from_card(unit / "card.toml")
        except (KeyError, ValueError):
            continue
        if not set(task.asset_panels.values()) <= {"rates_daily", "g10_fx_daily", "factors_daily"}:
            continue
        split = "train" if task.asof.year < 2018 else "validation" if task.asof.year < 2022 else "heldout"
        case = _case(unit, task)
        if case is None:
            continue
        pseudo, realized = case
        catalog.setdefault((task.family, split), []).append((unit, pseudo, realized))
    selected = []
    for (family, split), cases in sorted(catalog.items()):
        cases.sort(key=lambda x: (x[1].asof, x[0].name))
        picks = [cases[0]] if len(cases) == 1 else [cases[0], cases[-1]]
        selected.extend((split, *item) for item in picks)
    return selected


def _score(draws: np.ndarray, realized: np.ndarray, task: TaskSpec) -> dict[str, object]:
    overall = raw_track2_score(draws, realized)
    per_horizon = {}
    for k, horizon in enumerate(task.horizons):
        columns = [j * len(task.horizons) + k for j in range(len(task.assets))]
        per_horizon[str(horizon)] = raw_track2_score(draws[:, columns], realized[columns])
    return {**overall, "by_horizon": per_horizon}


def run(root: Path) -> dict[str, object]:
    cases = _selected_units(root)
    if not cases:
        raise ValueError("no retrospective cases with complete weekday targets")
    proxy = OfflineHeuristicModel()
    agent = Track2Agent(text_client=proxy)
    results = []
    for split, unit, task, realized in cases:
        panels = unit / "panels"
        text = unit / "text"
        base = agent.forecast(task, panels, text, task.asof, seed=2026, numeric_only=True)
        full = agent.forecast(task, panels, text, task.asof, seed=2026)
        neutral = agent.forecast(task, panels, text, task.asof, seed=2026,
                                 macro_override=type(full.macro.state).neutral())
        paths, card, context = agent._numeric_paths(task, panels, 2026)
        if not np.array_equal(base.draws, neutral.draws):
            raise AssertionError("neutral control differs from numeric base")
        state = full.macro.state
        scores = {}
        for strength in (0.0, 0.5, 1.0):
            conditioner = MacroConditioner(_strength(agent.params, strength))
            for name, flags in ABLATIONS.items():
                if strength != 1.0 and name != "F full":
                    continue
                if name == "A numeric":
                    draws = base.draws
                elif name == "G neutral":
                    draws = neutral.draws
                else:
                    chosen = type(state).neutral() if name == "G neutral" else state
                    draws = conditioner.condition(
                        paths, chosen, card, context, 2026,
                        mean="mean" in flags, volatility="volatility" in flags,
                        skew="skew" in flags, shock="shock" in flags, joint="joint" in flags,
                    ).draws
                key = name if strength == 1.0 else f"F full x{strength:g}"
                scores[key] = _score(draws, realized, task)
                if name == "F full" and strength == 1.0 and not np.array_equal(draws, full.draws):
                    raise AssertionError("ablation full differs from assembled agent")
        results.append({
            "split": split, "unit": unit.name, "origin": task.asof.isoformat(),
            "family": task.family, "panel": "+".join(dict.fromkeys(task.asset_panels[a] for a in task.assets)),
            "assets": list(task.assets), "horizons": list(task.horizons),
            "stress_group": "stressed" if state.market_stress > .1 or state.shock_probability > .2 else "calm",
            "text_source": full.macro.source, "confidence": state.confidence,
            "scores": scores,
        })
    # Strength is selected using train cases only. Validation and held-out are later.
    candidates = ["F full x0", "F full x0.5", "F full"]
    def relative(case, name):
        return case["scores"][name]["composite"] / max(case["scores"]["A numeric"]["composite"], 1e-12)
    train = [r for r in results if r["split"] == "train"]
    chosen = min(candidates, key=lambda name: (np.mean([relative(r, name) for r in train]), candidates.index(name)))
    return {"method": "offline lexical proxy; fixed parameter grid; raw unranked local scores",
            "selection": "first and last eligible unit per family/era; origin 30 weekdays before unit cutoff; 10/21 weekday targets",
            "chosen_from_train": chosen, "cases": results}


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-12 else (1.0 if abs(numerator) < 1e-12 else float("nan"))


def _print_heldout(report: dict[str, object]) -> None:
    cases = [row for row in report["cases"] if row["split"] == "heldout"]
    metrics = ("marginal", "joint", "tail", "composite")

    def ratios(rows, mode, horizon=None):
        values = {}
        for metric in metrics:
            observed = []
            for row in rows:
                score = row["scores"][mode]
                base = row["scores"]["A numeric"]
                if horizon is not None:
                    score, base = score["by_horizon"][horizon], base["by_horizon"][horizon]
                observed.append(_ratio(score[metric], base[metric]))
            values[metric] = float(np.nanmean(observed))
        return values

    print("Held-out mean score ratios to numeric (1.000 = same; lower is better)")
    print(f"{'group':<27} {'mode':<12} {'CRPS':>8} {'variogram':>10} {'tail':>8} {'composite':>10}")
    for mode in ABLATIONS:
        v = ratios(cases, mode)
        print(f"{'overall':<27} {mode:<12} {v['marginal']:>8.4f} {v['joint']:>10.4f} {v['tail']:>8.4f} {v['composite']:>10.4f}")
    for field, label in (("panel", "panel"), ("stress_group", "text stress")):
        for group in sorted({row[field] for row in cases}):
            rows = [row for row in cases if row[field] == group]
            v = ratios(rows, "F full")
            print(f"{(label + ': ' + group)[:27]:<27} {'F full':<12} {v['marginal']:>8.4f} {v['joint']:>10.4f} {v['tail']:>8.4f} {v['composite']:>10.4f}")
    for horizon in ("10", "21"):
        v = ratios(cases, "F full", horizon)
        print(f"{'horizon ' + horizon:<27} {'F full':<12} {v['marginal']:>8.4f} {v['joint']:>10.4f} {v['tail']:>8.4f} {v['composite']:>10.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track2-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/macro_ablation.json"))
    args = parser.parse_args()
    report = run(args.track2_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Selected on train:", report["chosen_from_train"])
    print(f"{'split':<11} {'family':<6} {'panel':<24} {'mode':<12} {'CRPS':>9} {'variogram':>10} {'tail':>9} {'composite':>10}")
    for split in SPLITS:
        subset = [r for r in report["cases"] if r["split"] == split]
        for family in ("T2-F1", "T2-F2", "T2-F3", "T2-F4"):
            rows = [r for r in subset if r["family"] == family]
            if not rows:
                continue
            for mode in dict.fromkeys(("A numeric", "F full", report["chosen_from_train"])):
                vals = {k: np.mean([r["scores"][mode][k] for r in rows]) for k in ("marginal", "joint", "tail", "composite")}
                panel = rows[0]["panel"] if len({r["panel"] for r in rows}) == 1 else "mixed panels"
                print(f"{split:<11} {family:<6} {panel:<24} {mode:<12} {vals['marginal']:>9.4f} {vals['joint']:>10.4f} {vals['tail']:>9.4f} {vals['composite']:>10.4f}")
    _print_heldout(report)
    print(f"Detailed panel, horizon, calm/stressed, and ablation scores: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
