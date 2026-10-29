"""Demonstrate base numerical models with synthetic yields and no scorer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from off_the_tape.features import FeatureLayer
from off_the_tape.gaussian_joint import GaussianJointForecaster
from off_the_tape.historical import PseudoCard
from off_the_tape.pca_joint import PCAJointForecaster


def run_demo():
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range('2020-01-02', periods=620)
    assets = ('UST_2Y', 'UST_5Y', 'UST_10Y')
    common = rng.normal(0, .015, (len(dates), 1))
    changes = common * np.array([1., .9, .8]) + rng.normal(0, .008, (len(dates), 3))
    values = np.array([3., 3.2, 3.5]) + np.cumsum(changes, axis=0)
    panel = pd.DataFrame({'date': np.repeat(dates, len(assets)),
                          'asset': np.tile(assets, len(dates)), 'value': values.reshape(-1)})
    asof = dates[-1].date()
    horizons = (2, 5, 10)
    features = FeatureLayer(panel, assets=assets, asof=asof)
    card = PseudoCard('T2-F3', 'rates_daily', asof, features.frame, assets, horizons, 'level',
                     tuple((dates[-1] + pd.offsets.BDay(h)).date() for h in horizons))
    draws = {'pca_base': PCAJointForecaster(mode='difference', seed=2026).forecast(card, features=features).draws,
             'gaussian_base': GaussianJointForecaster(seed=2026)(card)}
    output = {'synthetic': True, 'accuracy_evaluated': False, 'seed': 2026,
              'asof': asof.isoformat(), 'history_rows_per_asset': len(dates),
              'cells': [{'asset': a, 'horizon': h} for a, h in card.cells], 'models': {}}
    for name, samples in draws.items():
        if samples.shape != (1000, len(card.cells)) or not np.isfinite(samples).all():
            raise ValueError(f'Invalid demo scenarios: {name}')
        output['models'][name] = {'shape': list(samples.shape),
                                  'quantiles_05_50_95': np.quantile(samples, [.05, .5, .95], axis=0).tolist()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('reports/synthetic_demo.json'))
    args = parser.parse_args()
    result = run_demo()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('Synthetic demonstration only; no forecasting accuracy evaluated.')
    for name, model in result['models'].items():
        print(f"{name}: {model['shape'][0]} draws x {model['shape'][1]} asset/horizon cells")
        q = model['quantiles_05_50_95']
        print(f"  UST_2Y at 2 weekdays: 5%={q[0][0]:.4f}, median={q[1][0]:.4f}, 95%={q[2][0]:.4f}")
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
