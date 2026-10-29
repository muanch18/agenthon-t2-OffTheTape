"""Test predeclared zero/half/full historical drift with identical random draws.

Only the historical mean component changes; PCA's conditional AR dynamics,
innovations, widths, horizons and dependence stay fixed. Never scores holdout.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from qfbench2_track_forecasting.scoring import _composite

from off_the_tape.gaussian_joint import GaussianJointForecaster
from .final_cases import hydrate, load_panel
from .gaussian_comparison import expected_scales, pca_draws

FRACTIONS = (0.0, 0.5, 1.0)


def variants(card, seed):
    """Return paired scenarios; drift=1 exactly preserves the previous experiment."""
    gaussian = GaussianJointForecaster(seed=seed)
    mean, cov = gaussian.moments(card)
    gd = gaussian(card)
    pd = pca_draws(card, seed)
    anchors = np.array([card.panel.loc[card.panel.asset == a].sort_values("date").value.iloc[-1]
                        for a, _ in card.cells])
    gshift = mean - anchors
    pshift = []
    for asset, horizon in card.cells:
        raw = card.panel.loc[card.panel.asset == asset].sort_values("date").value.to_numpy()
        steps = np.diff(raw) if asset.startswith("UST_") else np.diff(np.log(raw))
        pshift.append(horizon * steps[-504:].mean())
    pshift = np.asarray(pshift)
    fx = np.array([not a.startswith("UST_") for a, _ in card.cells])
    result = {}
    for fraction in FRACTIONS:
        if fraction == 1:
            result[("gaussian", fraction)] = gd
            result[("pca", fraction)] = pd
            continue
        result[("gaussian", fraction)] = gd - (1-fraction) * gshift
        adjusted = pd.copy()
        adjusted[:, ~fx] -= (1-fraction) * pshift[~fx]
        adjusted[:, fx] *= np.exp(-(1-fraction) * pshift[fx])
        result[("pca", fraction)] = adjusted
    return result, expected_scales(mean, cov)


def summarize(rows):
    """Cluster by origin across seeds and shapes; preserve case-weighted means."""
    result = []
    for split in ("train", "validation"):
        for model in ("pca", "gaussian"):
            for shape in ("all", "rates_single", "rates_curve", "fx_single", "fx_basket", "mixed"):
                for fraction in FRACTIONS:
                    group = [r for r in rows if r["split"] == split and r["model"] == model
                             and r["fraction"] == fraction and (shape == "all" or r["shape"] == shape)]
                    if not group:
                        continue
                    # Each origin carries all of its correlated cases and seed replicates.
                    origins = sorted({r["asof"] for r in group})
                    sums = np.array([sum(r["score"]-r["full_score"] for r in group if r["asof"] == d)
                                     for d in origins])
                    counts = np.array([sum(r["asof"] == d for r in group) for d in origins])
                    rng = np.random.default_rng(901)
                    idx = rng.integers(0, len(origins), size=(2000, len(origins)))
                    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
                    result.append({"split": split, "model": model, "shape": shape,
                                   "fraction": fraction, "origins": len(origins),
                                   "score": float(np.mean([r["score"] for r in group])),
                                   "delta_vs_full": float(sums.sum()/counts.sum()),
                                   "origin_ci95": np.quantile(boot, [0.025, 0.975]).tolist()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/final_cases.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/drift_ablation.json"))
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    paths = [Path(s["path"]) for s in manifest["sources"]]
    for path, src in zip(paths, manifest["sources"]):
        if sha256(path.read_bytes()).hexdigest() != src["sha256"]:
            raise ValueError(f"changed source: {path}")
    panel = load_panel(*paths)
    specs = [s for s in manifest["cases"] if s["split"] in {"train", "validation"}]
    seeds = (2026, 17, 83)
    rows = []
    for i, spec in enumerate(specs):
        case = hydrate(panel, spec)
        weights = (0.5, 0.3, 0.2) if len(case.card.cells) > 1 else (5/7, 0, 2/7)
        for seed in seeds:
            draws, scales = variants(case.card, seed)
            scores = {}
            components = {}
            for key, samples in draws.items():
                c = _composite(samples, case.realized, weights=weights,
                               tail_levels=(0.01, 0.05, 0.95, 0.99), joint="variogram",
                               tail_metric="pinball", ref_scale=scales)
                if not np.isfinite(list(c.values())).all():
                    raise ValueError("non-finite score")
                scores[key] = float(np.clip(c["composite"], 0, 8))
                components[key] = c
            for (model, fraction), score in scores.items():
                rows.append({"case_id": spec["case_id"], "asof": spec["asof"],
                             "split": spec["split"], "shape": spec["shape"],
                             "model": model, "fraction": fraction, "seed": seed,
                             "score": score, "full_score": scores[(model, 1.0)],
                             "components": components[(model, fraction)]})
        if (i+1) % 100 == 0:
            print(f"Completed {i+1}/{len(specs)} cases", flush=True)
    summaries = summarize(rows)
    # One global fraction per model, chosen exclusively by training mean.
    selected = {model: min((s for s in summaries if s["split"] == "train"
                           and s["shape"] == "all" and s["model"] == model),
                          key=lambda s: s["score"])["fraction"] for model in ("pca", "gaussian")}
    payload = {"official_run": False, "holdout_evaluated": False,
               "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
               "seeds": seeds, "fractions": FRACTIONS, "selected_on_training": selected,
               "summary": summaries, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"selected_on_training": selected,
                      "overall": [s for s in summaries if s["shape"] == "all"]}, indent=2))


if __name__ == "__main__":
    main()
