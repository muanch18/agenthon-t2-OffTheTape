"""Frozen exploratory comparison with shared inputs; never evaluates holdout."""

import argparse
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from qfbench2_track_forecasting.scoring import _composite

from off_the_tape.features import FeatureLayer
from off_the_tape.gaussian_joint import GaussianJointForecaster
from off_the_tape.pca_joint import PCAJointForecaster
from off_the_tape.seeds import SeedManager
from .final_cases import hydrate, load_panel
from .gaussian_comparison import expected_scales

PROTOCOL = Path('experiments/frozen/controlled_architecture_v1.json')
ARMS = ('gaussian', 'pca')
SETTINGS = dict(split='validation', lookback_changes=504, draws=1000,
                seeds=[2026, 17, 83], drift='full historical sample mean',
                volatility='historical sample sd; no adaptive rescaling',
                panels='independent rates and FX', fx='log_price', rates='difference',
                pca_factors=2, pca_block=5, bootstrap_reps=10000,
                bootstrap_block=4, bootstrap_seed=901, timing_repeats=3)
SOURCES = ['experiments/controlled_architecture.py', 'experiments/final_cases.py',
           'experiments/gaussian_comparison.py', 'src/off_the_tape/features.py',
           'src/off_the_tape/pca_joint.py', 'src/off_the_tape/gaussian_joint.py',
           'src/off_the_tape/historical.py', 'src/off_the_tape/seeds.py',
           'tests/test_controlled_architecture.py', 'docs/controlled_architecture_protocol.md']
PACKAGES = ('numpy', 'pandas', 'pyarrow', 'scipy', 'qfbench2-common', 'qfbench2-track-forecasting')


def text_hash(path):
    return sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()


def prepare(card):
    """Shared aligned history and transformations, prepared outside timing."""
    groups = []
    for panel, mode, rate in [('rates_daily', 'difference', True), ('g10_fx_daily', 'log_price', False)]:
        assets = tuple(a for a in card.assets if a.startswith('UST_') == rate)
        if not assets:
            continue
        features = FeatureLayer(card.panel, assets=assets, asof=card.asof)
        local = replace(card, panel=features.frame, assets=assets, panel_id=panel)
        history = features.history(mode)
        if len(history.changes) < 504:
            raise ValueError('Insufficient shared history')
        train = history.changes[-504:]
        if np.any(train.std(axis=0, ddof=1) <= 0):
            raise ValueError('Degenerate history')
        groups.append((local, features, mode, train, history.values[-1]))
    return groups


def gaussian_daily(train, horizon, draws, seed):
    """Daily multivariate Gaussian shocks from the same transformed sample."""
    covariance = np.atleast_2d(np.cov(train, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-10 * max(1., float(eigenvalues.max())):
        raise ValueError('Invalid covariance')
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    rng = np.random.default_rng(seed)
    return train.mean(axis=0) + rng.standard_normal((draws, horizon, train.shape[1])) @ root.T


def forecast(card, groups, arm, seed, n_draws=1000):
    output = np.empty((n_draws, len(card.cells)))
    manager = SeedManager(seed)
    for local, features, mode, train, anchor in groups:
        local_seed = manager.derive(f'pca:{local.panel_id}')
        if arm == 'pca':
            samples = PCAJointForecaster(mode=mode, n_draws=n_draws,
                                        seed=local_seed).forecast(local, features=features).draws
        elif arm == 'gaussian':
            daily = gaussian_daily(train, max(card.horizons), n_draws,
                                   local_seed + card.asof.toordinal())
            cumulative = daily.cumsum(axis=1)
            paths = anchor + cumulative if mode == 'difference' else anchor * np.exp(cumulative)
            samples = np.column_stack([paths[:, h-1, j] for j in range(len(local.assets))
                                       for h in card.horizons])
        else:
            raise ValueError('Unknown arm')
        for j, asset in enumerate(local.assets):
            dest = card.assets.index(asset)*len(card.horizons)
            output[:, dest:dest+len(card.horizons)] = samples[:, j*len(card.horizons):(j+1)*len(card.horizons)]
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite forecast')
    fx = [i for i, (a, _) in enumerate(card.cells) if not a.startswith('UST_')]
    if fx and np.any(output[:, fx] <= 0):
        raise ValueError('Nonpositive FX')
    return output


def summary(rows):
    origins = sorted({r['asof'] for r in rows})
    sums = np.array([sum(r['pca']-r['gaussian'] for r in rows if r['asof'] == o) for o in origins])
    counts = np.array([sum(r['asof'] == o for r in rows) for o in origins])
    n = len(origins)
    b = min(4, n)
    starts = np.random.default_rng(901).integers(0, n-b+1, (10000, (n+b-1)//b))
    ix = (starts[:, :, None]+np.arange(b)).reshape(10000, -1)[:, :n]
    boot = sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    means = {a: float(np.mean([r[a] for r in rows])) for a in ARMS}
    return dict(cases=len({r['case_id'] for r in rows}), origins=n, **means,
                delta_pca_minus_gaussian=means['pca']-means['gaussian'],
                pca_relative_reduction_pct=100*(means['gaussian']-means['pca'])/means['gaussian'],
                exploratory_ci95=np.quantile(boot, [.025, .975]).tolist(),
                delta_by_seed={str(s): float(np.mean([r['pca']-r['gaussian'] for r in rows if r['seed'] == s]))
                               for s in SETTINGS['seeds']})


def environment():
    threads = {p: os.environ.get(p) for p in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}
    if set(threads.values()) != {'1'}:
        raise ValueError('Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1')
    return dict(python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in PACKAGES},
                threads=threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Write protocol without accessing outcomes')
    args = parser.parse_args()
    manifest_path = Path('reports/final_cases.json')
    manifest = json.loads(manifest_path.read_text())
    state = dict(settings=SETTINGS, manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                 code_sha256={p: text_hash(p) for p in SOURCES}, environment=environment(),
                 sources=manifest['sources'], exploratory=True, holdout_scored=False)
    for source in state['sources']:
        if sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Source data hash mismatch')
    if args.freeze:
        with PROTOCOL.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(state, indent=2)+'\n')
        print('Protocol written without scoring; commit before running.')
        return
    if json.loads(PROTOCOL.read_text()) != state:
        raise ValueError('Protocol mismatch')
    output = Path('reports/controlled_architecture_v1.json')
    if output.exists():
        raise ValueError('Refusing to overwrite result')
    specs = [s for s in manifest['cases'] if s['split'] == 'validation']
    if len(specs) != 226:
        raise ValueError('Unexpected roster')
    panel = load_panel(*[Path(s['path']) for s in state['sources']])
    rows, timings = [], []
    for index, spec in enumerate(specs):
        case = hydrate(panel, spec)
        groups = prepare(case.card)
        scale = expected_scales(*GaussianJointForecaster().moments(case.card))
        weights = (.5, .3, .2) if len(case.card.cells) > 1 else (5/7, 0, 2/7)
        for seed in SETTINGS['seeds']:
            row = {**spec, 'seed': seed, 'components': {}}
            for arm in ARMS:
                draws = forecast(case.card, groups, arm, seed)
                scores = _composite(draws, case.realized, weights=weights, joint='variogram',
                                    tail_metric='pinball', tail_levels=(.01, .05, .95, .99), ref_scale=scale)
                if not np.isfinite(list(scores.values())).all():
                    raise ValueError('Invalid score')
                row[arm] = float(np.clip(scores['composite'], 0, 8))
                row['components'][arm] = scores
            rows.append(row)
        # Above calls warm both arms. Alternate order to reduce systematic timing bias.
        measured = {a: [] for a in ARMS}
        for repeat in range(3):
            for arm in ARMS if (index+repeat) % 2 == 0 else ARMS[::-1]:
                start = perf_counter()
                forecast(case.card, groups, arm, 2026)
                measured[arm].append(perf_counter()-start)
        timings.append(dict(case_id=spec['case_id'], shape=spec['shape'],
                            **{a: float(np.median(measured[a])) for a in ARMS}))
        if (index+1) % 25 == 0:
            print(f'Completed {index+1}/226 validation cases', flush=True)
    result = dict(protocol_sha256=sha256(PROTOCOL.read_bytes()).hexdigest(),
                  exploratory=True, holdout_scored=False, official_run=False,
                  summary=summary(rows),
                  by_shape={s: summary([r for r in rows if r['shape'] == s]) for s in sorted({r['shape'] for r in rows})},
                  runtime_seconds={a: dict(median=float(np.median([r[a] for r in timings])),
                                          p90=float(np.quantile([r[a] for r in timings], .9))) for a in ARMS},
                  runtime_environment=dict(platform=platform.platform(), processor=platform.processor(),
                                           note='Warm model API calls; PCA includes diagnostic construction; excludes shared preparation and scoring'),
                  rows=rows, timings=timings)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    audit = {k: v for k, v in result.items() if k not in ('rows', 'timings')}
    audit['raw_result_sha256'] = sha256(output.read_bytes()).hexdigest()
    audit['raw_result_path'] = str(output)
    with Path('experiments/results/controlled_architecture_v1.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
