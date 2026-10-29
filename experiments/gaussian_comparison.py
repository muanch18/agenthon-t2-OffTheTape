"""Compare frozen PCA numeric logic with a Gaussian benchmark on non-holdout cases.

Calls official component scorers. Reconstructs the published daily M0 expected
divisors locally; these are research scores, not an official leaderboard run.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import gamma, hyp1f1, ndtr
from scipy.stats import norm
from qfbench2_track_forecasting.scoring import _composite

from off_the_tape.gaussian_joint import GaussianJointForecaster
from off_the_tape.pca_joint import PCAJointForecaster
from off_the_tape.seeds import SeedManager
from .final_cases import hydrate, load_panel


def expected_scales(mean, cov):
    """Published M0 §5 formulas; all inputs are known before outcomes."""
    sd = np.sqrt(np.diag(cov))
    joint = 0.0
    for i in range(len(mean)):
        for j in range(i):
            delta = abs(mean[i] - mean[j])
            s = np.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 0))
            if s == 0:
                continue
            z = delta / s
            if z > 100:
                # Asymptotic expansion avoids catastrophic cancellation.
                variance = s * s / (4 * delta) * (1 + 7 / (8 * z * z))
            else:
                e_abs = s * np.sqrt(2 / np.pi) * np.exp(-z*z/2) + delta * (2*ndtr(z)-1)
                e_root = np.sqrt(s) * 2**0.25 * gamma(0.75) / np.sqrt(np.pi) * hyp1f1(-0.25, 0.5, -z*z/2)
                variance = max(e_abs - e_root**2, 0)
            joint += 2 * variance
    return {"marginal": float(sd.mean()/np.sqrt(np.pi)),
            "tail": float(norm.pdf(norm.ppf([0.01, 0.05, 0.95, 0.99])).mean()*sd.mean()),
            "joint": float(joint) if joint > 0 else 1.0}


def pca_draws(card, seed):
    """Match production panel separation, transforms, defaults, and seed derivation."""
    output = np.empty((1000, len(card.cells)))
    seeds = SeedManager(seed)
    for panel_id, mode, assets in (
        ("rates_daily", "difference", tuple(a for a in card.assets if a.startswith("UST_"))),
        ("g10_fx_daily", "log_price", tuple(a for a in card.assets if not a.startswith("UST_"))),
    ):
        if not assets:
            continue
        local = replace(card, panel=card.panel.loc[card.panel.asset.isin(assets)],
                        panel_id=panel_id, assets=assets)
        draws = PCAJointForecaster(mode=mode, seed=seeds.derive(f"pca:{panel_id}")).forecast(local).draws
        for j, asset in enumerate(assets):
            dest = card.assets.index(asset) * len(card.horizons)
            output[:, dest:dest+len(card.horizons)] = draws[:, j*len(card.horizons):(j+1)*len(card.horizons)]
    return output


def summary(rows):
    result = []
    for split, shape in sorted({(r["split"], r["shape"]) for r in rows}):
        group = [r for r in rows if (r["split"], r["shape"]) == (split, shape)]
        # First average seeds within case. Paired bootstrap resamples origins.
        origins = sorted({r["asof"] for r in group})
        deltas = np.array([np.mean([r["pca"]-r["gaussian"] for r in group if r["asof"] == d]) for d in origins])
        rng = np.random.default_rng(901)
        boot = rng.choice(deltas, (2000, len(deltas)), replace=True).mean(axis=1)
        result.append({"split": split, "shape": shape, "origins": len(origins),
                       "pca": float(np.mean([r["pca"] for r in group])),
                       "gaussian": float(np.mean([r["gaussian"] for r in group])),
                       "pca_minus_gaussian": float(deltas.mean()),
                       "paired_origin_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                       "pca_win_fraction": float(np.mean(deltas < 0))})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("reports/final_cases.json"))
    p.add_argument("--out", type=Path, default=Path("reports/gaussian_comparison.json"))
    p.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train", "validation"])
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 17, 83])
    args = p.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    paths = [Path(r["path"]) for r in manifest["sources"]]
    for path, src in zip(paths, manifest["sources"]):
        if sha256(path.read_bytes()).hexdigest() != src["sha256"]:
            raise ValueError(f"panel changed since manifest creation: {path}")
    panel = load_panel(*paths)
    rows = []
    specs = [s for s in manifest["cases"] if s["split"] in args.splits]
    for index, spec in enumerate(specs):
        case = hydrate(panel, spec)
        gaussian = GaussianJointForecaster()
        scale = expected_scales(*gaussian.moments(case.card))
        weights = (0.5, 0.3, 0.2) if len(case.card.cells) > 1 else (5/7, 0, 2/7)
        for seed in args.seeds:
            row = {**spec, "seed": seed}
            for name, draws in (("gaussian", replace(gaussian, seed=seed)(case.card)),
                                ("pca", pca_draws(case.card, seed))):
                components = _composite(draws, case.realized, weights=weights,
                                        tail_levels=(0.01, 0.05, 0.95, 0.99), joint="variogram",
                                        tail_metric="pinball", ref_scale=scale)
                if not np.isfinite(list(components.values())).all():
                    raise ValueError("non-finite score")
                row[name] = float(np.clip(components["composite"], 0, 8))
                row[name+"_components"] = components
            rows.append(row)
        if (index+1) % 100 == 0:
            print(f"Completed {index+1}/{len(specs)} cases", flush=True)
    payload = {"official_run": False, "holdout_evaluated": False,
               "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
               "seeds": args.seeds, "summary": summary(rows), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
