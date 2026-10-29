"""Isolate cross-panel dependence on the fixed mixed historical cases only.

Gaussian varies only off-panel covariance; PCA varies only whether historical
innovation blocks share dates across panels. Holdout is never evaluated.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from qfbench2_track_forecasting.scoring import _composite

from off_the_tape.gaussian_joint import GaussianJointForecaster
from off_the_tape.pca_joint import PCAJointForecaster
from off_the_tape.seeds import SeedManager
from .final_cases import hydrate, load_panel
from .gaussian_comparison import expected_scales

WEIGHTS = (0.0, 0.5, 1.0)
SEEDS = (2026, 17, 83, 9, 123)


def paired_draws(card, seed):
    if not any(a.startswith("UST_") for a in card.assets) or not any(not a.startswith("UST_") for a in card.assets):
        raise ValueError("mixed rates/FX assets required")
    g = GaussianJointForecaster(seed=seed)
    mean, cov = g.moments(card)
    rate_cells = np.array([a.startswith("UST_") for a, _ in card.cells])
    cross = rate_cells[:, None] != rate_cells[None, :]
    independent_cov = cov.copy()
    independent_cov[cross] = 0
    z = np.random.default_rng(seed + card.asof.toordinal()).standard_normal((1000, len(mean)))
    result = {}
    for weight in WEIGHTS:
        target_cov = independent_cov + weight * (cov-independent_cov)
        result[("gaussian", weight)] = mean + z @ np.linalg.cholesky(target_cov).T

    manager = SeedManager(seed)
    panel_draws = {}
    schedules = {}
    locals_ = {}
    for panel_id, mode, is_rate in (("rates_daily", "difference", True), ("g10_fx_daily", "log_price", False)):
        assets = tuple(a for a in card.assets if a.startswith("UST_") == is_rate)
        local = replace(card, panel=card.panel.loc[card.panel.asset.isin(assets)], panel_id=panel_id, assets=assets)
        model = PCAJointForecaster(mode=mode, seed=manager.derive(f"pca:{panel_id}"))
        # Matches PCA's training/innovation counts and default five-day block rule.
        n_dates = local.panel.date.nunique()
        innovations = min(504, n_dates-1)-1
        block = min(5, innovations)
        n_blocks = (max(card.horizons)+block-1)//block
        rng = np.random.default_rng(model.seed + card.asof.toordinal())
        schedules[panel_id] = rng.integers(0, innovations-block+1, size=(1000, n_blocks))
        locals_[panel_id] = (local, model)
        panel_draws[panel_id] = model.forecast(local).draws
    if locals_["rates_daily"][0].panel.date.nunique() != locals_["g10_fx_daily"][0].panel.date.nunique():
        raise ValueError("shared PCA schedules require identical historical dates")
    rate_dates = sorted(locals_["rates_daily"][0].panel.date.unique())
    fx_dates = sorted(locals_["g10_fx_daily"][0].panel.date.unique())
    if rate_dates != fx_dates:
        raise ValueError("panels have different historical dates")
    # Nested path-wise mixing, independent of outcomes, keeps all horizons together.
    mix = np.random.default_rng(manager.derive("cross_panel_mix")).random(1000)
    for weight in WEIGHTS:
        if weight == 0:
            fx = panel_draws["g10_fx_daily"]
        else:
            starts = schedules["g10_fx_daily"].copy()
            starts[mix < weight] = schedules["rates_daily"][mix < weight]
            local, model = locals_["g10_fx_daily"]
            fx = model.forecast(local, block_starts=starts).draws
        combined = np.empty((1000, len(card.cells)))
        for asset_index, asset in enumerate(card.assets):
            panel_id = "rates_daily" if asset.startswith("UST_") else "g10_fx_daily"
            local, _ = locals_[panel_id]
            j = local.assets.index(asset)
            source = panel_draws[panel_id] if panel_id == "rates_daily" else fx
            h = len(card.horizons)
            combined[:, asset_index*h:(asset_index+1)*h] = source[:, j*h:(j+1)*h]
        result[("pca", weight)] = combined
    return result, expected_scales(mean, cov)


def summarize(rows):
    summary = []
    for split in ("train", "validation"):
        for model in ("pca", "gaussian"):
            for weight in WEIGHTS:
                group = [r for r in rows if r["split"] == split and r["model"] == model and r["weight"] == weight]
                origins = sorted({r["asof"] for r in group})
                if not origins:
                    continue
                deltas = np.array([np.mean([r["score"]-r["independent_score"] for r in group if r["asof"] == d]) for d in origins])
                boot = np.random.default_rng(901).choice(deltas, (2000, len(deltas))).mean(axis=1)
                summary.append({"split": split, "model": model, "weight": weight,
                                "origins": len(origins), "score": float(np.mean([r["score"] for r in group])),
                                "delta_vs_independent": float(deltas.mean()),
                                "origin_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                                "delta_by_seed": {str(seed): float(np.mean([r["score"]-r["independent_score"]
                                                   for r in group if r["seed"] == seed])) for seed in SEEDS}})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/final_cases.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/cross_panel_ablation.json"))
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    paths = [Path(s["path"]) for s in manifest["sources"]]
    for path, src in zip(paths, manifest["sources"]):
        if sha256(path.read_bytes()).hexdigest() != src["sha256"]:
            raise ValueError(f"changed source: {path}")
    panel = load_panel(*paths)
    specs = [s for s in manifest["cases"] if s["shape"] == "mixed" and s["split"] in {"train", "validation"}]
    rows = []
    for index, spec in enumerate(specs):
        case = hydrate(panel, spec)
        for seed in SEEDS:
            draws, scale = paired_draws(case.card, seed)
            scores, components = {}, {}
            for key, samples in draws.items():
                c = _composite(samples, case.realized, weights=(0.5, 0.3, 0.2),
                               tail_levels=(0.01, 0.05, 0.95, 0.99), joint="variogram",
                               tail_metric="pinball", ref_scale=scale)
                if not np.isfinite(list(c.values())).all():
                    raise ValueError("non-finite score")
                scores[key] = float(np.clip(c["composite"], 0, 8))
                components[key] = c
            for (model, weight), score in scores.items():
                rows.append({"case_id": spec["case_id"], "asof": spec["asof"], "split": spec["split"],
                             "model": model, "weight": weight, "seed": seed, "score": score,
                             "independent_score": scores[(model, 0.0)], "components": components[(model, weight)]})
        if (index+1) % 25 == 0:
            print(f"Completed {index+1}/{len(specs)} mixed cases", flush=True)
    stats = summarize(rows)
    selected = {model: min((s for s in stats if s["split"] == "train" and s["model"] == model),
                          key=lambda s: s["score"])["weight"] for model in ("pca", "gaussian")}
    payload = {"official_run": False, "holdout_evaluated": False, "seeds": SEEDS,
               "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
               "selected_on_training": selected, "summary": stats, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"selected_on_training": selected, "summary": stats}, indent=2))


if __name__ == "__main__":
    main()
