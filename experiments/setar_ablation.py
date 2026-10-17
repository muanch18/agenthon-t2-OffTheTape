"""Exploratory band-SETAR mean dynamics with unchanged PCA/AR innovation draws.

One fixed specification, train/validation only. Does not modify production,
reuse the consumed holdout, or apply the recently tested volatility blend.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from qfbench2_track_forecasting.scoring import _composite

from off_the_tape.gaussian_joint import GaussianJointForecaster
from off_the_tape.pca_joint import PCAJointForecaster
from off_the_tape.seeds import SeedManager
from .final_cases import hydrate, load_panel
from .gaussian_comparison import expected_scales

SPEC = {"threshold_quantile": 0.8, "regime_weight": 0.5, "min_regime": 50,
        "slope_bound": 0.95, "innovation_pool": "unchanged pooled AR1",
        "volatility": "unchanged original PCA", "seeds": [2026, 17, 83]}


def fit_band(factors, pooled_intercept, pooled_slope, *, quantile=0.8, weight=0.5, min_regime=50):
    if not 0 < quantile < 1 or not 0 <= weight <= 1 or min_regime < 3:
        raise ValueError("invalid band-SETAR settings")
    lagged, current = factors[:-1], factors[1:]
    threshold = np.quantile(np.abs(lagged), quantile, axis=0)
    intercept = np.tile(pooled_intercept, (2, 1))
    slope = np.tile(pooled_slope, (2, 1))
    counts = np.zeros((2, factors.shape[1]), dtype=int)
    fallback = []
    for j in range(factors.shape[1]):
        regime = np.abs(lagged[:, j]) > threshold[j]
        counts[:, j] = [np.sum(~regime), np.sum(regime)]
        if min(counts[:, j]) < min_regime:
            fallback.append(j)
            continue
        masks = (~regime, regime)
        denominators = [np.sum((lagged[m, j]-lagged[m, j].mean())**2) for m in masks]
        if min(denominators) <= 1e-12:
            fallback.append(j)
            continue
        for r, mask in enumerate(masks):
            x, y = lagged[mask, j], current[mask, j]
            b = np.clip(np.sum((x-x.mean())*(y-y.mean()))/denominators[r], -0.95, 0.95)
            a = y.mean()-b*x.mean()
            slope[r, j] = (1-weight)*pooled_slope[j]+weight*b
            intercept[r, j] = (1-weight)*pooled_intercept[j]+weight*a
    return {"threshold": threshold, "intercept": intercept, "slope": slope,
            "counts": counts, "fallback_factors": fallback}


def advance(state, shock, band):
    regime = np.abs(state) > band["threshold"]
    factors = np.arange(state.shape[1])[None, :]
    a = band["intercept"][regime.astype(int), factors]
    b = band["slope"][regime.astype(int), factors]
    return a+b*state+shock, regime


def panel_pair(card, mode, seed):
    model = PCAJointForecaster(mode=mode, seed=seed)
    control = model.forecast(card)
    wide = card.panel.pivot(index="date", columns="asset", values="value").sort_index().loc[:, list(card.assets)]
    values = wide.to_numpy()
    training = np.diff(values if mode == "difference" else np.log(values), axis=0)[-504:]
    center, scale = training.mean(axis=0), training.std(axis=0, ddof=1)
    standardized = (training-center)/scale
    loadings = np.asarray(control.diagnostics["loading_matrix_standardized"])
    factors = standardized @ loadings
    residuals = standardized-factors @ loadings.T
    slope = np.asarray(control.diagnostics["ar1_slope"])
    intercept = factors[1:].mean(axis=0)-slope*factors[:-1].mean(axis=0)
    band = fit_band(factors, intercept, slope, quantile=SPEC["threshold_quantile"],
                    weight=SPEC["regime_weight"], min_regime=SPEC["min_regime"])
    shocks = factors[1:]-intercept-slope*factors[:-1]
    innovations = np.concatenate((shocks, residuals[1:]), axis=1)
    innovations -= innovations.mean(axis=0)
    h = max(card.horizons)
    block = min(5, len(innovations))
    rng = np.random.default_rng(seed+card.asof.toordinal())
    starts = rng.integers(0, len(innovations)-block+1, size=(1000, (h+block-1)//block))
    indices = (starts[:, :, None]+np.arange(block)).reshape(1000, -1)[:, :h]
    sampled = innovations[indices]
    k = factors.shape[1]
    state = np.broadcast_to(factors[-1], (1000, k)).copy()
    # Audit reconstruction of the unchanged AR1 arm against production paths.
    linear_state = state.copy()
    daily, linear_daily = [], []
    stress_counts = np.zeros(k, dtype=int)
    max_state = np.zeros(k)
    for day in range(h):
        state, regime = advance(state, sampled[:, day, :k], band)
        linear_state = intercept+slope*linear_state+sampled[:, day, :k]
        stress_counts += regime.sum(axis=0)
        max_state = np.maximum(max_state, np.max(np.abs(state), axis=0))
        daily.append(center+scale*(state @ loadings.T+sampled[:, day, k:]))
        linear_daily.append(center+scale*(linear_state @ loadings.T+sampled[:, day, k:]))
    daily = np.stack(daily, axis=1)
    linear_daily = np.stack(linear_daily, axis=1)
    if not np.array_equal(linear_daily, control.daily_transforms):
        raise ValueError("innovation reconstruction differs from production AR1")
    cumulative = np.cumsum(daily, axis=1)
    paths = values[-1]+cumulative if mode == "difference" else values[-1]*np.exp(cumulative)
    if not np.isfinite(paths).all() or (mode == "log_price" and np.any(paths <= 0)):
        raise ValueError("invalid threshold simulation")
    draws = np.column_stack([paths[:, t-1, j] for j in range(len(card.assets)) for t in card.horizons])
    diagnostics = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in band.items()}
    diagnostics.update(stress_fraction=(stress_counts/(1000*h)).tolist(), max_abs_factor_state=max_state.tolist(),
                       control_innovations_exact=True)
    return control.draws, draws, diagnostics


def paired_draws(card, seed):
    output = {name: np.empty((1000, len(card.cells))) for name in ("ar1", "setar")}
    manager, diagnostics = SeedManager(seed), {}
    for panel_id, mode, is_rate in (("rates_daily", "difference", True), ("g10_fx_daily", "log_price", False)):
        assets = tuple(a for a in card.assets if a.startswith("UST_") == is_rate)
        if not assets:
            continue
        local = replace(card, assets=assets, panel_id=panel_id, panel=card.panel.loc[card.panel.asset.isin(assets)])
        ar1, setar, diagnostic = panel_pair(local, mode, manager.derive(f"pca:{panel_id}"))
        diagnostics[panel_id] = diagnostic
        width = len(card.horizons)
        for j, asset in enumerate(assets):
            dest = card.assets.index(asset)*width
            for name, source in (("ar1", ar1), ("setar", setar)):
                output[name][:, dest:dest+width] = source[:, j*width:(j+1)*width]
    return output, diagnostics


def summarize(rows):
    result = []
    for split in ("train", "validation"):
        for shape in ("all", "rates_single", "rates_curve", "fx_single", "fx_basket", "mixed"):
            group = [r for r in rows if r["split"] == split and (shape == "all" or r["shape"] == shape)]
            if not group:
                continue
            origins = sorted({r["asof"] for r in group})
            sums = np.array([sum(r["setar"]-r["ar1"] for r in group if r["asof"] == d) for d in origins])
            counts = np.array([sum(r["asof"] == d for r in group) for d in origins])
            block = min(4, len(origins))
            starts = np.random.default_rng(901).integers(0, len(origins)-block+1, (10000, (len(origins)+block-1)//block))
            indices = (starts[:, :, None]+np.arange(block)).reshape(10000, -1)[:, :len(origins)]
            boot = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
            ar1, setar = [float(np.mean([r[name] for r in group])) for name in ("ar1", "setar")]
            result.append({"split": split, "shape": shape, "origins": len(origins), "ar1": ar1, "setar": setar,
                           "delta": setar-ar1, "relative_improvement_pct": 100*(ar1-setar)/ar1,
                           "exploratory_block_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                           "delta_by_seed": {str(seed): float(np.mean([r["setar"]-r["ar1"] for r in group
                                                            if r["seed"] == seed])) for seed in SPEC["seeds"]}})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/final_cases.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/setar_ablation.json"))
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("result already exists; refusing overwrite")
    manifest = json.loads(args.manifest.read_text())
    for source in manifest["sources"]:
        if sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("historical panel changed")
    panel = load_panel(*[Path(s["path"]) for s in manifest["sources"]])
    specs = [s for s in manifest["cases"] if s["split"] in {"train", "validation"}]
    rows = []
    for index, spec in enumerate(specs):
        case = hydrate(panel, spec)
        scales = expected_scales(*GaussianJointForecaster().moments(case.card))
        weights = (0.5, 0.3, 0.2) if len(case.card.cells) > 1 else (5/7, 0, 2/7)
        for seed in SPEC["seeds"]:
            draws, diagnostic = paired_draws(case.card, seed)
            row = {**spec, "seed": seed, "diagnostics": diagnostic, "components": {}}
            for name, samples in draws.items():
                c = _composite(samples, case.realized, weights=weights, joint="variogram", tail_metric="pinball",
                               tail_levels=(0.01, 0.05, 0.95, 0.99), ref_scale=scales)
                if not np.isfinite(list(c.values())).all():
                    raise ValueError("invalid score")
                row[name] = float(np.clip(c["composite"], 0, 8))
                row["components"][name] = c
            rows.append(row)
        if (index+1) % 100 == 0:
            print(f"Completed {index+1}/{len(specs)} exploratory cases", flush=True)
    stats = summarize(rows)
    result = {"exploratory": True, "holdout_scored": False, "specification": SPEC,
              "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
              "code_sha256": sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "qfbench2-common", "qfbench2-track-forecasting")},
              "summary": stats, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps([s for s in stats if s["shape"] == "all"], indent=2))


if __name__ == "__main__":
    main()
