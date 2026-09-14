"""Score saved historical forecasts without running a forecasting model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .historical import PseudoCard, evaluate, historical_cases


def samples_from_frame(frame: pd.DataFrame, card: PseudoCard) -> np.ndarray:
    """Read the exact Track 2 draw grid in card order, rejecting gaps and repeats."""
    if set(frame.columns) != {"draw", "asset", "horizon", "value"}:
        raise ValueError("forecast needs exactly draw, asset, horizon, value columns")
    if frame.empty or frame[["draw", "asset", "horizon", "value"]].isna().any().any():
        raise ValueError("forecast must be nonempty and contain no nulls")
    if not pd.api.types.is_integer_dtype(frame["draw"]) or not pd.api.types.is_integer_dtype(frame["horizon"]):
        raise ValueError("draw and horizon must be integers")
    draw_ids = sorted(frame["draw"].unique().tolist())
    if draw_ids != list(range(len(draw_ids))):
        raise ValueError("draw ids must be contiguous starting at zero")
    if frame.duplicated(["draw", "asset", "horizon"]).any():
        raise ValueError("forecast contains duplicate grid cells")
    expected = pd.MultiIndex.from_product(
        [draw_ids, card.assets, card.horizons], names=["draw", "asset", "horizon"]
    )
    indexed = frame.set_index(["draw", "asset", "horizon"])
    if len(indexed) != len(expected) or not indexed.index.isin(expected).all():
        raise ValueError("forecast grid differs from the pseudo-card")
    ordered = indexed.reindex(expected)
    if ordered["value"].isna().any():
        raise ValueError("forecast grid has missing cells")
    values = ordered["value"].to_numpy(dtype=float).reshape(len(draw_ids), len(card.cells))
    if not np.isfinite(values).all():
        raise ValueError("forecast values must be finite")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="full historical long-format Parquet panel")
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--family", choices=("T2-F1", "T2-F2", "T2-F3", "T2-F4"), required=True)
    parser.add_argument("--forecasts", type=Path, required=True, help="directory of YYYY-MM-DD.parquet forecast files")
    parser.add_argument("--assets", nargs="+", required=True)
    parser.add_argument("--horizons", nargs="+", required=True, type=int)
    parser.add_argument("--target-type", choices=("level", "log_return"), required=True)
    parser.add_argument("--origins", nargs="+", required=True, help="historical as-of dates")
    parser.add_argument("--output", type=Path, required=True, help="local JSON report path")
    args = parser.parse_args(argv)

    cases = historical_cases(
        pd.read_parquet(args.panel), family=args.family, panel_id=args.panel_id,
        assets=args.assets, horizons=args.horizons,
        target_type=args.target_type, origins=args.origins,
    )

    def load_forecast(card: PseudoCard) -> np.ndarray:
        path = args.forecasts / f"{card.asof.isoformat()}.parquet"
        return samples_from_frame(pd.read_parquet(path), card)

    results = evaluate(cases, load_forecast)
    report = {
        "rankable": False,
        "normalization_mode": "raw_unranked",
        "note": "Historical diagnostics only; no official reference-scale normalization or sealed outcomes.",
        "summary": {
            "n_cases": len(results),
            "mean_marginal": float(np.mean([row["marginal"] for row in results])),
            "mean_joint": float(np.mean([row["joint"] for row in results])),
            "mean_tail": float(np.mean([row["tail"] for row in results])),
            "mean_composite": float(np.mean([row["composite"] for row in results])),
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
