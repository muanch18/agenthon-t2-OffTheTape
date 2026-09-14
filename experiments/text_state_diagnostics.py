"""Inspect macro states on F1-F4 practice text without touching forecasts.

Use --offline-heuristic for a labeled local proxy when MODEL_ENDPOINT is absent.
That mode tests the pipeline; it is not evidence of LLM extraction quality.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from off_the_tape.macro_extractor import extract_macro_state, extract_with_prior
from off_the_tape.macro_heuristic import OfflineHeuristicModel

PERIODS = (
    ("calm FX", "t2-F1-eur-range-2017", None, "T2-F1", "g10_fx_daily"),
    ("pre liftoff", "t2-F1-liftoff-telegraph-2015", "2015-09-17", "T2-F1", "rates_daily"),
    ("post liftoff", "t2-F1-liftoff-telegraph-2015", None, "T2-F1", "rates_daily"),
    ("inflation", "t2-F1-cpi-glidepath-2023", None, "T2-F1", "rates_daily"),
    ("credit stress", "t2-F2-credit-crunch-2007", None, "T2-F2", "rates_daily"),
    ("joint FX", "t2-F3-election-2024-joint", None, "T2-F3", "g10_fx_daily"),
    ("JPY crowding", "t2-F4-jpy-crowding-2024", None, "T2-F4", "g10_fx_daily"),
)


def run(
    track2_root: Path, *, window_days: int, client, model_name: str,
    cache_dir: Path, text_enabled: bool = True,
) -> tuple[list[dict[str, object]], list[str]]:
    rows = []
    for label, unit_name, override_asof, family, panel_id in PERIODS:
        unit = track2_root / "units" / unit_name
        index = json.loads((unit / "text" / "corpus_index.json").read_text(encoding="utf-8"))
        asof = date.fromisoformat(override_asof or index["asof"])
        common = dict(
            family=family, panel_id=panel_id, client=client, model_name=model_name,
            cache_dir=cache_dir, text_enabled=text_enabled,
        )
        comparison = extract_with_prior(unit, asof=asof, window_days=window_days, **common)
        repeated = extract_macro_state(unit, asof=asof, **common)
        state = comparison.current.state
        rows.append({
            "period": label,
            "unit": unit_name,
            "asof": asof.isoformat(),
            "panel": panel_id,
            "family": family,
            "source": comparison.current.source,
            "stable_repeat": repeated.state == state,
            "selected_docs": list(comparison.current.selected_doc_ids),
            "state": state.to_dict(),
            "delta": comparison.delta.to_dict(),
            "evidence": [item.to_dict() for item in comparison.current.evidence],
        })
    warnings = []
    if next(row for row in rows if row["period"] == "calm FX")["state"]["shock_probability"] > 0.7:
        warnings.append("calm-period shock probability exceeds 0.7")
    if sum(abs(row["state"]["policy_stance"]) >= 0.99 for row in rows) > len(rows) // 2:
        warnings.append("policy stance saturates for most periods")
    if all(row["state"]["confidence"] >= 0.99 for row in rows):
        warnings.append("confidence saturates at 1")
    if len({json.dumps(row["state"], sort_keys=True) for row in rows}) == 1:
        warnings.append("all extracted states are identical")
    if not all(row["stable_repeat"] for row in rows):
        warnings.append("repeated inputs produced unstable states")
    return rows, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track2-root", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--offline-heuristic", action="store_true")
    parser.add_argument("--text-ablated", action="store_true")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "offline-heuristic-v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/macro_state"))
    parser.add_argument("--output", type=Path, default=Path("reports/text_state_diagnostics.json"))
    args = parser.parse_args(argv)
    if not args.offline_heuristic and not args.text_ablated and not os.environ.get("MODEL_ENDPOINT"):
        parser.error("set MODEL_ENDPOINT or use --offline-heuristic")
    client = OfflineHeuristicModel() if args.offline_heuristic else None
    rows, warnings = run(
        args.track2_root, window_days=args.window_days, client=client,
        model_name=args.model, cache_dir=args.cache_dir, text_enabled=not args.text_ablated,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "mode": "offline heuristic proxy" if args.offline_heuristic else "model endpoint",
        "window_days": args.window_days,
        "rows": rows,
        "sanity_warnings": warnings,
    }, indent=2) + "\n", encoding="utf-8")
    print("Macro state diagnostic (offline lexical proxy, not LLM)" if args.offline_heuristic else "Macro state diagnostic (model endpoint)")
    print(f"{'period':<14} {'asof':<10} {'panel':<15} {'policy':>7} {'d_policy':>8} {'infl':>7} {'d_infl':>7} {'growth':>7} {'shock':>7} {'vol':>7} {'skew':>7} {'conf':>7}")
    for row in rows:
        state, delta = row["state"], row["delta"]
        print(
            f"{row['period']:<14} {row['asof']:<10} {row['panel']:<15} "
            f"{state['policy_stance']:>7.2f} {delta['policy_stance_delta']:>8.2f} "
            f"{state['inflation_pressure']:>7.2f} {delta['inflation_pressure_delta']:>7.2f} "
            f"{state['growth_outlook']:>7.2f} {state['shock_probability']:>7.2f} "
            f"{state['expected_volatility_change']:>7.2f} {state['expected_skew']:>7.2f} "
            f"{state['confidence']:>7.2f}"
        )
    print("Sanity checks:", "passed" if not warnings else "; ".join(warnings))
    print(f"Detailed evidence and states: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
