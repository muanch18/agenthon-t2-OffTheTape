"""Demonstrate shared features on earlier historical cases, without scoring outcomes."""

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from off_the_tape.features import FeatureLayer
from off_the_tape.historical import PseudoCard
from .final_cases import load_panel
from .volatility_ablation import scale_ratios as historical_scale_ratios


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/final_cases.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/shared_features.json"))
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    for source in manifest["sources"]:
        if sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("historical source changed")
    panel = load_panel(*[Path(s["path"]) for s in manifest["sources"]])
    selected = {}
    for spec in manifest["cases"]:
        if spec["split"] == "train":
            selected.setdefault(spec["shape"], spec)
    rows = []
    max_difference = 0.0
    for spec in selected.values():
        # Earlier observed values only; this demo neither constructs nor scores outcomes.
        frame = panel.loc[(panel.date <= spec["asof"]) & panel.asset.isin(spec["assets"])]
        # Match the historical suite's explicit complete-case alignment policy.
        wide = frame.pivot(index="date", columns="asset", values="value").dropna()
        frame = wide.rename_axis(columns="asset").stack().rename("value").reset_index()
        for asset in spec["assets"]:
            mode = "difference" if asset.startswith("UST_") else "log_price"
            layer = FeatureLayer(frame, assets=(asset,), asof=spec["asof"])
            vol = layer.volatility(mode)
            rows.append({"asof": spec["asof"], "shape": spec["shape"], "asset": asset,
                         "mode": mode, "historical_sd": vol.historical_sd.tolist(),
                         "recent_sd": vol.recent_sd.tolist(), "ewma_sd": vol.ewma_sd.tolist(),
                         "observations": vol.observations, "version": vol.version})
        # Parity only against the old volatility feature formula; do not score its predictions.
        card = PseudoCard("T2-F3", "features_demo", date.fromisoformat(spec["asof"]), frame,
                          tuple(spec["assets"]), tuple(spec["horizons"]), "level", ())
        previous = historical_scale_ratios(card, "pca")
        for j, asset in enumerate(spec["assets"]):
            mode = "difference" if asset.startswith("UST_") else "log_price"
            vol = FeatureLayer(card.panel, assets=(asset,), asof=card.asof).volatility(mode)
            variants = {"current": vol.historical_sd, "recent60": vol.recent_sd,
                        "blend60": vol.variance_blend(source="recent"),
                        "ewma20_blend": vol.variance_blend(source="ewma")}
            for rule, sd in variants.items():
                ratio = np.clip(sd[0]/vol.historical_sd[0], 0.5, 2)
                expected = previous[rule][j*len(spec["horizons"])]
                difference = abs(ratio-expected)
                max_difference = max(max_difference, difference)
                if difference > 1e-12:
                    raise ValueError("shared volatility definition differs from historical experiment")
    result = {"scores_computed": False, "holdout_scored": False,
              "historical_cases_checked": len(selected), "max_sd_ratio_difference": max_difference,
              "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(), "features": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"Shared feature parity on {len(selected)} training examples; max ratio difference {max_difference:.3g}")


if __name__ == "__main__":
    main()
