"""Run the joint-row bootstrap on historical UST and G10 FX pseudo-cards.

The source Parquet files must contain dates after every requested origin. Scores
are raw local diagnostics and cannot be compared to the official leaderboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from off_the_tape.bootstrap import JointRowBootstrap
from off_the_tape.historical import evaluate, historical_cases

DEFAULT_ORIGINS = ("2018-01-31", "2020-01-31", "2022-01-31", "2024-01-31")
GROUPS = (
    ("UST rates", "rates_daily", ("UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"), "difference"),
    ("G10 FX", "g10_fx_daily", ("EUR", "GBP", "AUD", "NZD"), "log_price"),
)


def run(rates_panel: Path, fx_panel: Path, origins: tuple[str, ...] = DEFAULT_ORIGINS) -> list[dict[str, object]]:
    rows = []
    for (label, panel_id, assets, mode), path in zip(GROUPS, (rates_panel, fx_panel)):
        cases = historical_cases(
            pd.read_parquet(path), family="T2-F3", panel_id=panel_id,
            assets=assets, horizons=(21, 63), target_type="level", origins=origins,
        )
        for result in evaluate(cases, JointRowBootstrap(mode=mode)):
            rows.append({"panel": label, **result})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates-panel", type=Path, required=True)
    parser.add_argument("--fx-panel", type=Path, required=True)
    parser.add_argument("--origins", nargs="+", default=DEFAULT_ORIGINS)
    args = parser.parse_args(argv)

    rows = run(args.rates_panel, args.fx_panel, tuple(args.origins))
    print("Raw, unranked historical scores (1,000 joint draws; lower is better)")
    print(f"{'panel':<11} {'as-of':<10} {'CRPS':>12} {'variogram':>12} {'tail':>12} {'composite':>12}")
    for row in rows:
        print(
            f"{row['panel']:<11} {row['asof']:<10} "
            f"{row['marginal']:>12.6f} {row['joint']:>12.6f} "
            f"{row['tail']:>12.6f} {row['composite']:>12.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
