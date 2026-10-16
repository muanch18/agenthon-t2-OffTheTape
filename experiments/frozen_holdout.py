"""Evaluate only hash-frozen candidates on the previously unscored holdout.

Protocol creation/commit precedes outcome scoring. No candidate tuning or
production changes occur here. A protocol-bound result is written once.
"""

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from qfbench2_track_forecasting.scoring import _composite

from off_the_tape.gaussian_joint import GaussianJointForecaster
from .final_cases import hydrate, load_panel
from .gaussian_comparison import expected_scales
from .volatility_ablation import pca_control_and_location, rescale, scale_ratios

ARM_RULES = {"pca": "ewma20_blend", "gaussian": "recent60"}


def frozen_draws(card, seed):
    g = GaussianJointForecaster(seed=seed)
    mean, cov = g.moments(card)
    pc, location = pca_control_and_location(card, seed)
    output = {"pca_current": pc, "gaussian_current": g(card)}
    for model, loc in (("pca", location), ("gaussian", mean)):
        log_cells = np.array([model == "pca" and not a.startswith("UST_") for a, _ in card.cells])
        ratio = scale_ratios(card, model)[ARM_RULES[model]]
        output[model+"_candidate"] = rescale(output[model+"_current"], loc, ratio, log_cells)
    return output, expected_scales(mean, cov)


def verify_protocol(protocol, manifest_path):
    if protocol["candidate_rules"] != ARM_RULES:
        raise ValueError("candidate definition differs from frozen protocol")
    if sha256(manifest_path.read_bytes()).hexdigest() != protocol["manifest_sha256"]:
        raise ValueError("historical manifest changed after freeze")
    for name, digest in protocol["code_sha256"].items():
        # Universal-newline text hashing survives Git's Windows CRLF checkout.
        if sha256(Path(name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"code changed after freeze: {name}")
    for name, version in protocol["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"evaluation package changed: {name}")
    if sys.version.split()[0] != protocol["python_version"]:
        raise ValueError("Python version changed")
    if protocol["seeds"] != [2026, 17, 83] or protocol["primary_comparisons"] != ["pca", "gaussian"]:
        raise ValueError("unexpected frozen seeds/comparisons")


def paired_summary(rows, model, *, block_length, reps, quantiles, seed):
    origins = sorted({r["asof"] for r in rows})
    if not origins:
        raise ValueError("empty paired comparison")
    sums, counts = [], []
    for origin in origins:
        group = [r for r in rows if r["asof"] == origin]
        sums.append(sum(r[model+"_candidate"]-r[model+"_current"] for r in group))
        counts.append(len(group))
    sums, counts = np.asarray(sums), np.asarray(counts)
    n = len(origins)
    block_length = min(block_length, n)
    # Moving blocks: sample only contiguous non-wrapping historical blocks.
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-block_length+1, size=(reps, (n+block_length-1)//block_length))
    indices = (starts[:, :, None]+np.arange(block_length)).reshape(reps, -1)[:, :n]
    boot = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    current = float(np.mean([r[model+"_current"] for r in rows]))
    candidate = float(np.mean([r[model+"_candidate"] for r in rows]))
    return {"model": model, "origins": n, "cases": len({r["case_id"] for r in rows}),
            "current": current, "candidate": candidate,
            "delta": float(sums.sum()/counts.sum()),
            "relative_improvement_pct": 100*(current-candidate)/current,
            "block_interval": np.quantile(boot, quantiles).tolist(),
            "delta_by_seed": {str(s): float(np.mean([r[model+"_candidate"]-r[model+"_current"]
                                    for r in rows if r["seed"] == s])) for s in sorted({r["seed"] for r in rows})}}


def summarize(rows, protocol):
    settings = {"block_length": protocol["bootstrap"]["block_length"],
                "reps": protocol["bootstrap"]["replicates"], "seed": protocol["bootstrap"]["seed"]}
    primary = []
    for model in protocol["primary_comparisons"]:
        result = paired_summary(rows, model, quantiles=[0.0125, 0.9875], **settings)
        result["conclusion"] = "supported_on_this_holdout" if result["block_interval"][1] < 0 else "not_confirmed"
        primary.append(result)
    diagnostics = []
    for field in ("shape", "year"):
        for value in sorted({r[field] for r in rows}):
            group = [r for r in rows if r[field] == value]
            for model in protocol["primary_comparisons"]:
                diagnostics.append({"group": field, "value": value,
                                    **paired_summary(group, model, quantiles=[0.025, 0.975], **settings)})
    return {"primary": primary, "diagnostics": diagnostics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/frozen/holdout_20261016.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/frozen_holdout_20261016.json"))
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("holdout output already exists; refusing to overwrite")
    protocol = json.loads(args.protocol.read_text())
    manifest_path = Path(protocol["manifest_path"])
    verify_protocol(protocol, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for source in manifest["sources"]:
        if sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("source panel changed after historical case construction")
    specs = [s for s in manifest["cases"] if s["split"] == "holdout"]
    if len(specs) != protocol["holdout_cases"] or len({s["asof"] for s in specs}) != protocol["holdout_origins"]:
        raise ValueError("holdout roster changed")
    panel = load_panel(*[Path(s["path"]) for s in manifest["sources"]])
    rows = []
    for index, spec in enumerate(specs):
        case = hydrate(panel, spec)
        if case.card.panel.date.max().date() > case.card.asof:
            raise ValueError("post-cutoff history")
        weights = (0.5, 0.3, 0.2) if len(case.card.cells) > 1 else (5/7, 0, 2/7)
        for seed in protocol["seeds"]:
            draws, scales = frozen_draws(case.card, seed)
            row = {**spec, "year": spec["asof"][:4], "seed": seed, "components": {}}
            for arm, samples in draws.items():
                if samples.shape != (1000, len(case.card.cells)) or not np.isfinite(samples).all():
                    raise ValueError(f"invalid scenarios: {arm}")
                c = _composite(samples, case.realized, weights=weights, joint="variogram", tail_metric="pinball",
                               tail_levels=(0.01, 0.05, 0.95, 0.99), ref_scale=scales)
                if not np.isfinite(list(c.values())).all():
                    raise ValueError("invalid score")
                row[arm] = float(np.clip(c["composite"], 0, 8))
                row["components"][arm] = c
            rows.append(row)
        if (index+1) % 25 == 0:
            print(f"Completed {index+1}/{len(specs)} frozen holdout cases", flush=True)
    result = {"official_run": False, "holdout_now_consumed": True,
              "protocol_sha256": sha256(args.protocol.read_bytes()).hexdigest(),
              "summary": summarize(rows, protocol), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards against an overlapping evaluation process.
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["summary"]["primary"], indent=2))


if __name__ == "__main__":
    main()
