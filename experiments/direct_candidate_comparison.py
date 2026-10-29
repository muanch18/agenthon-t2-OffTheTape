"""Exploratory paired comparison of saved, already inspected holdout scores.

No forecasts are regenerated and no candidate or frozen protocol is modified.
"""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def summarize(rows, block_length=4, reps=10000, seed=901):
    cases = defaultdict(list)
    for row in rows:
        cases[row['case_id']].append(row)
    averaged = []
    for group in cases.values():
        averaged.append({
            'asof': group[0]['asof'],
            'pca': float(np.mean([r['pca_candidate'] for r in group])),
            'gaussian': float(np.mean([r['gaussian_candidate'] for r in group])),
        })
    origins = sorted({r['asof'] for r in averaged})
    sums = np.array([sum(r['gaussian']-r['pca'] for r in averaged if r['asof'] == o)
                     for o in origins])
    counts = np.array([sum(r['asof'] == o for r in averaged) for o in origins])
    n = len(origins)
    length = min(block_length, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-length+1, size=(reps, (n+length-1)//length))
    indices = (starts[:, :, None]+np.arange(length)).reshape(reps, -1)[:, :n]
    boot = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    pca = float(np.mean([r['pca'] for r in averaged]))
    gaussian = float(np.mean([r['gaussian'] for r in averaged]))
    return dict(cases=len(averaged), origins=n, pca=pca, gaussian=gaussian,
                delta_gaussian_minus_pca=gaussian-pca,
                gaussian_relative_improvement_pct=100*(pca-gaussian)/pca,
                exploratory_95_interval=np.quantile(boot, [.025, .975]).tolist(),
                exploratory_97_5_interval=np.quantile(boot, [.0125, .9875]).tolist(),
                gaussian_case_win_fraction=float(np.mean([r['gaussian'] < r['pca'] for r in averaged])),
                gaussian_origin_win_fraction=float(np.mean(sums < 0)),
                delta_by_seed={str(s): float(np.mean([r['gaussian_candidate']-r['pca_candidate']
                              for r in rows if r['seed'] == s])) for s in sorted({r['seed'] for r in rows})})


def analyze(source, audit):
    digest = sha256(source.read_bytes()).hexdigest()
    if digest != audit['raw_result_sha256']:
        raise ValueError('Saved results do not match the tracked audit hash')
    data = json.loads(source.read_text())
    if data['protocol_sha256'] != audit['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    rows = data['rows']
    seen = set()
    for r in rows:
        key = (r['case_id'], r['seed'])
        if key in seen or r['split'] != 'holdout':
            raise ValueError('Duplicate record or non-holdout record')
        seen.add(key)
        if not np.isfinite([r['pca_candidate'], r['gaussian_candidate']]).all():
            raise ValueError('Nonfinite score')
    groups = defaultdict(set)
    for r in rows:
        groups[r['case_id']].add(r['seed'])
    if len(rows) != 468 or len(groups) != 156 or len({r['asof'] for r in rows}) != 33:
        raise ValueError('Unexpected holdout roster')
    if any(seeds != {2026, 17, 83} for seeds in groups.values()):
        raise ValueError('Incomplete seed grid')
    return dict(exploratory=True, official_run=False, source_sha256=digest,
                delta_definition='Gaussian candidate minus PCA candidate; negative favors Gaussian',
                bootstrap=dict(replicates=10000, block_length=4, seed=901,
                               method='non-wrapping moving blocks of ordered origins; equal case weights'),
                overall=summarize(rows),
                diagnostics={field: {value: summarize([r for r in rows if r[field] == value])
                             for value in sorted({r[field] for r in rows})} for field in ('year', 'shape')},
                block_length_sensitivity={str(b): summarize(rows, block_length=b)['exploratory_95_interval']
                                          for b in (2, 6, 8)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('reports/frozen_holdout_20261016.json'))
    parser.add_argument('--out', type=Path, default=Path('reports/direct_candidate_comparison.json'))
    args = parser.parse_args()
    audit = json.loads(Path('experiments/frozen/holdout_20261016_results.json').read_text())
    result = analyze(args.input, audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
