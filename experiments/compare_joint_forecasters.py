"""Compare joint-row bootstrap and PCA/AR(1) block innovation forecasts.

Uses only pre-as-of panel rows for each forecast. All metrics are raw, unranked
local diagnostics and should only be compared within the same panel and grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from off_the_tape.bootstrap import JointRowBootstrap
from off_the_tape.historical import evaluate, historical_cases
from off_the_tape.pca_joint import PCAJointForecaster

from .bootstrap_backtest import DEFAULT_ORIGINS, GROUPS

METRICS = ("marginal", "joint", "tail", "composite")


def compare(
    rates_panel: Path, fx_panel: Path, origins: tuple[str, ...] = DEFAULT_ORIGINS
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scores: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for (label, panel_id, assets, mode), path in zip(GROUPS, (rates_panel, fx_panel)):
        cases = historical_cases(
            pd.read_parquet(path), family="T2-F3", panel_id=panel_id,
            assets=assets, horizons=(21, 63), target_type="level", origins=origins,
        )
        bootstrap = JointRowBootstrap(mode=mode, lookback=504)
        for score in evaluate(cases, bootstrap):
            scores.append({"panel": label, "model": "bootstrap", **score})

        pca = PCAJointForecaster(mode=mode, lookback=504, block_size=5)

        def forecast(card):
            result = pca.forecast(card)
            diagnostics.append({"panel": label, **result.diagnostics})
            return result.draws

        for score in evaluate(cases, forecast):
            scores.append({"panel": label, "model": "PCA joint", **score})
    return scores, diagnostics


def summarize(scores: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for panel in ("UST rates", "G10 FX"):
        for model in ("bootstrap", "PCA joint"):
            group = [row for row in scores if row["panel"] == panel and row["model"] == model]
            summary.append({
                "panel": panel,
                "model": model,
                "n_origins": len(group),
                **{metric: sum(float(row[metric]) for row in group) / len(group) for metric in METRICS},
            })
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates-panel", type=Path, required=True)
    parser.add_argument("--fx-panel", type=Path, required=True)
    parser.add_argument("--origins", nargs="+", default=DEFAULT_ORIGINS)
    parser.add_argument("--diagnostics-out", type=Path, default=Path("reports/pca_joint_diagnostics.json"))
    args = parser.parse_args(argv)

    scores, diagnostics = compare(args.rates_panel, args.fx_panel, tuple(args.origins))
    summary = summarize(scores)
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.write_text(
        json.dumps({
            "rankable": False,
            "note": "Raw, unnormalized historical scores; compare within a panel/grid only.",
            "summary": summary,
            "scores_by_origin": scores,
            "pca_diagnostics": diagnostics,
        }, indent=2) + "\n", encoding="utf-8",
    )

    print("Mean raw historical scores over the same origins (lower is better)")
    print(f"{'panel':<11} {'model':<10} {'CRPS':>11} {'variogram':>11} {'tail':>11} {'composite':>11}")
    for row in summary:
        print(
            f"{row['panel']:<11} {row['model']:<10} "
            f"{row['marginal']:>11.6f} {row['joint']:>11.6f} "
            f"{row['tail']:>11.6f} {row['composite']:>11.6f}"
        )
    print("PCA failures by origin (higher composite than bootstrap):")
    for panel in ("UST rates", "G10 FX"):
        for origin in args.origins:
            matched = [row for row in scores if row["panel"] == panel and row["asof"] == origin]
            naive = next(row for row in matched if row["model"] == "bootstrap")
            pca = next(row for row in matched if row["model"] == "PCA joint")
            if pca["composite"] > naive["composite"]:
                print(f"  {panel} {origin}: {naive['composite']:.6f} -> {pca['composite']:.6f}")
    print(f"Detailed scores and PCA diagnostics: {args.diagnostics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
