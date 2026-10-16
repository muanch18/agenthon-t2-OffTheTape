"""Compare fixed volatility rules with drift, dependence and shocks unchanged.

Only pre-cutoff history estimates volatility. Training selects one global rule
per model. Validation assesses it; holdout is never scored.
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

RULES = ("current", "recent60", "blend60", "ewma20_blend")
SEEDS = (2026, 17, 83)


def scale_ratios(card, model):
    """Assetwise sd ratios; changing a rule never refits drift or correlations."""
    ratios = {rule: [] for rule in RULES}
    for asset in card.assets:
        raw = card.panel.loc[card.panel.asset == asset].sort_values("date").value.to_numpy()
        if model == "gaussian":
            pool = np.diff(raw[-300:])
        else:
            pool = np.diff(raw)[-504:] if asset.startswith("UST_") else np.diff(np.log(raw))[-504:]
        if len(pool) < 60 or not np.isfinite(pool).all():
            raise ValueError("need at least 60 finite historical changes")
        base = pool.var(ddof=1)
        if base <= 0:
            raise ValueError("positive historical variance required")
        recent = pool[-60:].var(ddof=1)
        w = 0.5 ** (np.arange(len(pool))[::-1]/20)
        w /= w.sum()
        ewma = np.sum(w*(pool-np.sum(w*pool))**2)/(1-np.sum(w*w))
        variances = (base, recent, 0.5*base+0.5*recent, 0.5*base+0.5*ewma)
        for rule, variance in zip(RULES, variances):
            ratios[rule].append(float(np.clip(np.sqrt(variance/base), 0.5, 2.0)))
    return {rule: np.repeat(r, len(card.horizons)) for rule, r in ratios.items()}


def pca_control_and_location(card, seed):
    """Production scenarios and deterministic conditional location in modeling space."""
    output = np.empty((1000, len(card.cells)))
    location = np.empty(len(card.cells))
    manager = SeedManager(seed)
    for panel_id, mode, is_rate in (("rates_daily", "difference", True), ("g10_fx_daily", "log_price", False)):
        assets = tuple(a for a in card.assets if a.startswith("UST_") == is_rate)
        if not assets:
            continue
        local = replace(card, assets=assets, panel_id=panel_id, panel=card.panel.loc[card.panel.asset.isin(assets)])
        fit = PCAJointForecaster(mode=mode, seed=manager.derive(f"pca:{panel_id}")).forecast(local)
        values = local.panel.pivot(index="date", columns="asset", values="value").sort_index().loc[:, list(assets)].to_numpy()
        transformed = values if is_rate else np.log(values)
        training = np.diff(transformed, axis=0)[-504:]
        center, sd = training.mean(axis=0), training.std(axis=0, ddof=1)
        loadings = np.asarray(fit.diagnostics["loading_matrix_standardized"])
        slope = np.asarray(fit.diagnostics["ar1_slope"])
        factors = ((training-center)/sd) @ loadings
        intercept = factors[1:].mean(axis=0)-slope*factors[:-1].mean(axis=0)
        state = factors[-1].copy()
        expected_daily = []
        for _ in range(max(card.horizons)):
            state = intercept+slope*state
            expected_daily.append(center+sd*(state @ loadings.T))
        expected_path = transformed[-1]+np.cumsum(expected_daily, axis=0)
        for j, asset in enumerate(assets):
            start = card.assets.index(asset)*len(card.horizons)
            stop = start+len(card.horizons)
            output[:, start:stop] = fit.draws[:, j*len(card.horizons):(j+1)*len(card.horizons)]
            location[start:stop] = expected_path[np.array(card.horizons)-1, j]
    return output, location


def rescale(draws, location, ratio, log_cells):
    space = draws.copy()
    space[:, log_cells] = np.log(space[:, log_cells])
    space = location+(space-location)*ratio
    space[:, log_cells] = np.exp(space[:, log_cells])
    return space


def variants(card, seed):
    g = GaussianJointForecaster(seed=seed)
    mean, cov = g.moments(card)
    pc, ploc = pca_control_and_location(card, seed)
    controls = {"pca": pc, "gaussian": g(card)}
    locations = {"pca": ploc, "gaussian": mean}
    output, ratios = {}, {}
    for model in ("pca", "gaussian"):
        ratios[model] = scale_ratios(card, model)
        log_cells = np.array([model == "pca" and not a.startswith("UST_") for a, _ in card.cells])
        for rule in RULES:
            # Avoid round-trip logarithms for exact control parity.
            output[(model, rule)] = controls[model] if rule == "current" else rescale(
                controls[model], locations[model], ratios[model][rule], log_cells)
    return output, expected_scales(mean, cov), ratios


def summarize(rows):
    stats = []
    for split in ("train", "validation"):
        for model in ("pca", "gaussian"):
            for shape in ("all", "rates_single", "rates_curve", "fx_single", "fx_basket", "mixed"):
                for rule in RULES:
                    group = [r for r in rows if r["split"] == split and r["model"] == model
                             and r["rule"] == rule and (shape == "all" or r["shape"] == shape)]
                    if not group:
                        continue
                    origins = sorted({r["asof"] for r in group})
                    sums = np.array([sum(r["score"]-r["current_score"] for r in group if r["asof"] == d) for d in origins])
                    counts = np.array([sum(r["asof"] == d for r in group) for d in origins])
                    idx = np.random.default_rng(901).integers(0, len(origins), (2000, len(origins)))
                    boot = sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
                    stats.append({"split": split, "model": model, "shape": shape, "rule": rule,
                                  "origins": len(origins), "score": float(np.mean([r["score"] for r in group])),
                                  "delta_vs_current": float(sums.sum()/counts.sum()),
                                  "origin_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                                  "delta_by_seed": {str(seed): float(np.mean([r["score"]-r["current_score"]
                                                     for r in group if r["seed"] == seed])) for seed in SEEDS}})
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/final_cases.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/volatility_ablation.json"))
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    paths = [Path(s["path"]) for s in manifest["sources"]]
    for path, src in zip(paths, manifest["sources"]):
        if sha256(path.read_bytes()).hexdigest() != src["sha256"]:
            raise ValueError(f"changed source: {path}")
    panel = load_panel(*paths)
    specs = [s for s in manifest["cases"] if s["split"] in {"train", "validation"}]
    rows = []
    for index, spec in enumerate(specs):
        case = hydrate(panel, spec)
        weights = (0.5, 0.3, 0.2) if len(case.card.cells) > 1 else (5/7, 0, 2/7)
        for seed in SEEDS:
            draws, scales, ratios = variants(case.card, seed)
            scores, components = {}, {}
            for key, samples in draws.items():
                if not np.isfinite(samples).all():
                    raise ValueError("invalid scenarios")
                c = _composite(samples, case.realized, weights=weights, joint="variogram", tail_metric="pinball",
                               tail_levels=(0.01, 0.05, 0.95, 0.99), ref_scale=scales)
                if not np.isfinite(list(c.values())).all():
                    raise ValueError("invalid score")
                components[key] = c
                scores[key] = float(np.clip(c["composite"], 0, 8))
            for (model, rule), score in scores.items():
                rows.append({"case_id": spec["case_id"], "asof": spec["asof"], "shape": spec["shape"],
                             "split": spec["split"], "model": model, "rule": rule, "seed": seed,
                             "score": score, "current_score": scores[(model, "current")],
                             "components": components[(model, rule)], "scale_ratios": ratios[model][rule].tolist()})
        if (index+1) % 100 == 0:
            print(f"Completed {index+1}/{len(specs)} cases", flush=True)
    stats = summarize(rows)
    selected = {model: min((s for s in stats if s["model"] == model and s["split"] == "train" and s["shape"] == "all"),
                          key=lambda s: s["score"])["rule"] for model in ("pca", "gaussian")}
    payload = {"official_run": False, "holdout_evaluated": False, "rules": RULES, "seeds": SEEDS,
               "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
               "selected_on_training": selected, "summary": stats, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"selected_on_training": selected, "overall": [s for s in stats if s["shape"] == "all"]}, indent=2))


if __name__ == "__main__":
    main()
