"""Build numeric Final-like historical cases without selecting on future moves.

The manifest contains case specifications, never outcomes. Hydration returns
pre-cutoff model inputs separately from realized values using historical_cases.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from off_the_tape.historical import historical_cases

GROUPS = {
    "rates_single": ("UST_2Y",),
    "fx_single": ("EUR",),
    "rates_curve": ("UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"),
    "fx_basket": ("EUR", "GBP", "JPY", "CHF"),
    "mixed": ("UST_2Y", "UST_10Y", "EUR", "JPY"),
}
RATES = ("UST_2Y", "UST_5Y", "UST_7Y", "UST_10Y", "UST_20Y", "UST_30Y")
FX = ("EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK", "DKK")


def load_panel(rates: Path, fx: Path) -> pd.DataFrame:
    frames = []
    for path, panel_id in ((rates, "rates_daily"), (fx, "g10_fx_daily")):
        frame = pd.read_parquet(path).rename(columns={"asset_id": "asset"})
        frame = frame.loc[:, ["date", "asset", "value"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["panel_id"] = panel_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def partition(asof, last_target):
    """Purge cases whose targets cross a chronological split boundary."""
    for name, lo, hi in (("train", "2005-01-01", "2017-12-31"),
                         ("validation", "2018-01-01", "2021-12-31"),
                         ("holdout", "2022-01-01", "2024-12-31")):
        if pd.Timestamp(lo) <= asof <= last_target <= pd.Timestamp(hi):
            return name
    return None


def build_manifest(panel, *, start="2005-01-03", end="2024-10-31", stride=20,
                   seed=2026, min_history=505):
    if stride < 1 or min_history < 20 or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("invalid schedule or history requirement")
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("duplicate date/asset rows")
    wide = panel.pivot(index="date", columns="asset", values="value").sort_index()
    cases, excluded = [], []
    # Calendar schedule is constructed before any outcome availability checks.
    for asof in pd.bdate_range(start, end)[::stride]:
        for shape, assets in GROUPS.items():
            key = f"{seed}:{asof.date()}:{shape}"
            digest = sha256(key.encode()).digest()
            if shape == "rates_single":
                assets = (RATES[digest[2] % len(RATES)],)
            elif shape == "fx_single":
                assets = (FX[digest[2] % len(FX)],)
            h1 = 2 + digest[0] % 11
            h2 = 2 + digest[1] % 10
            if h2 >= h1:
                h2 += 1
            horizons = sorted((h1, h2)) if shape not in {"rates_single", "fx_single"} else [h1]
            targets = [asof + pd.offsets.BDay(h) for h in horizons]
            split = partition(asof, max(targets))
            reason = None
            if split is None:
                reason = "split_boundary"
            elif not set(assets).issubset(wide.columns):
                reason = "missing_asset"
            else:
                history = wide.loc[wide.index <= asof, list(assets)].dropna()
                if len(history) < min_history:
                    reason = "insufficient_history"
                elif asof not in wide.index or wide.loc[asof, list(assets)].isna().any():
                    reason = "missing_origin"
                elif any(t not in wide.index or wide.loc[t, list(assets)].isna().any() for t in targets):
                    reason = "missing_target"
            if reason:
                excluded.append({"asof": str(asof.date()), "shape": shape, "reason": reason})
                continue
            cases.append({"case_id": sha256(key.encode()).hexdigest()[:20],
                          "asof": str(asof.date()), "shape": shape, "split": split,
                          "assets": list(assets), "horizons": horizons,
                          "target_dates": [str(t.date()) for t in targets]})
    return {"version": 1, "seed": seed, "start": start, "end": end,
            "stride_business_days": stride, "min_history": min_history,
            "target_policy": "exact pandas BDay; missing targets excluded, never filled",
            "text_available": False, "family_semantics_validated": False,
            "cases": cases, "excluded": excluded}


def hydrate(panel, spec):
    """Build one case on demand; F3 is a numeric API label, not inferred text semantics."""
    frame = panel.loc[panel.asset.isin(spec["assets"]), ["date", "asset", "value"]]
    # Align using pre-asof complete rows only; never impute historical gaps.
    wide = frame.pivot(index="date", columns="asset", values="value").dropna()
    aligned = wide.rename_axis(columns="asset").stack().rename("value").reset_index()
    return historical_cases(aligned, family="T2-F3", panel_id="final_numeric",
                            assets=spec["assets"], horizons=spec["horizons"],
                            target_type="level", origins=[spec["asof"]])[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates-panel", type=Path, required=True)
    parser.add_argument("--fx-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/final_cases.json"))
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    panel = load_panel(args.rates_panel, args.fx_panel)
    result = build_manifest(panel, stride=args.stride, seed=args.seed)
    result["sources"] = [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()}
                         for p in (args.rates_panel, args.fx_panel)]
    # Verify every admitted case can actually be hydrated before writing output.
    for spec in result["cases"]:
        case = hydrate(panel, spec)
        assert case.card.panel.date.max().date() <= case.card.asof
    result["counts"] = dict(Counter(f"{c['split']}/{c['shape']}" for c in result["cases"]))
    result["exclusion_counts"] = dict(Counter(c["reason"] for c in result["excluded"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"admitted": len(result["cases"]), "counts": result["counts"],
                      "exclusions": result["exclusion_counts"]}, indent=2))


if __name__ == "__main__":
    main()
