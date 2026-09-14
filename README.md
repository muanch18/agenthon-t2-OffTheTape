# Off The Tape: historical Track 2 evaluation

This repository currently provides a local historical evaluation harness. It creates
pseudo-cards from a full historical panel, passes only observations through each
as-of date to a forecasting callback, constructs later outcomes, and scores saved
joint draws. A deliberately simple numeric baseline samples whole historical
cross-asset change rows and produces 1,000 joint scenarios. The LLM/text
component in [the plan](docs/plan.md) is not implemented yet.

The input panel must be long-format Parquet with `date`, `asset`, and `value`
columns, as in the [Track 2 public repository](https://github.com/Agenthon-2026/track2-forecasting-public).
An optional `panel_id` column is checked against the requested panel ID. The public practice units contain
cutoff panels but **no answer keys**, so this harness needs a separate full
historical panel for retrospective runs. It does not reconstruct or publish
official sealed outcomes. Keep local reports and full panels out of Git.

Use Python 3.13 or newer. Install the shared toolkit at the public repository's
documented tag, then the Track 2 scorer and this package:

```powershell
python -m pip install "qfbench2-common[data] @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.0#subdirectory=common"
python -m pip install "qfbench2-track-forecasting @ git+https://github.com/Agenthon-2026/track2-forecasting-public.git@83c6dc036bec03862b78e093bf8804d98964dad5"
python -m pip install -e ".[test]"
```

To score forecasts already saved for a historical panel:

```powershell
off-the-tape-backtest --panel C:\data\rates_daily.parquet `
  --panel-id rates_daily --family T2-F3 `
  --forecasts C:\data\historical_draws --assets UST_2Y UST_10Y `
  --horizons 21 63 --target-type level `
  --origins 2018-01-31 2018-04-30 --output reports\rates.json
```

For each origin, provide `<forecasts>/<YYYY-MM-DD>.parquet` with exactly
`draw`, `asset`, `horizon`, and `value` columns. Draw IDs start at zero and
identify complete joint scenarios across the asset/horizon grid. At least 200
draws are required. The same source panel can be used programmatically:

```python
from off_the_tape.historical import evaluate, historical_cases

cases = historical_cases(panel, family="T2-F3", panel_id="rates_daily",
                         assets=["UST_2Y", "UST_10Y"],
                         horizons=[21], target_type="level",
                         origins=["2018-01-31"])
from off_the_tape.bootstrap import JointRowBootstrap
results = evaluate(cases, JointRowBootstrap(mode="difference"))
```

`make_joint_draws` receives a `PseudoCard` containing only history through
`card.asof`, and returns a NumPy matrix with shape `(n_draws,
len(card.assets) * len(card.horizons))`. Columns follow asset-major,
horizon-minor order. A horizon uses the exact weekday date `asof + BDay(h)`;
if that date is absent from the panel, the case fails rather than shifting its
target. For `level`, the outcome is the value on that target date. For
`log_return`, panel values are daily simple returns; the outcome is the sum of
`log(1 + r)` over available observations after as-of through the target date.
All requested assets must be present on every panel date. Missing or incomplete
origins fail instead of being silently excluded.

The bootstrap has three modes: `difference` adds resampled daily level changes
(suited to UST yields), `log_price` compounds resampled log-price changes
(suited to positive FX rates), and `simple_return` sums `log(1+r)` from daily
simple-return panels. Each simulated day samples one **joint** historical row,
preserving contemporaneous cross-asset dependence. The default lookback is 504
observations. Its random seed is fixed per as-of date for reproducibility; the
method does not model serial dependence, regimes, or text.

The stronger numeric candidate is `PCAJointForecaster` in
[`src/off_the_tape/pca_joint.py`](src/off_the_tape/pca_joint.py). It transforms
rates to daily changes, FX levels to daily log returns, and factor daily simple
returns to `log(1+r)` for cumulative-log-return targets. On the last 504
pre-as-of observations it standardizes each series, decomposes the joint panel
as `X_t = B f_t + e_t`, fits a clipped AR(1) to each factor, and resamples
paired factor innovations and residuals in five-day blocks. It then simulates
the full path once per draw and reads every requested horizon from that same
path. The empirical shocks retain observed extremes; they do not extrapolate
beyond the historical shock pool.

```python
from off_the_tape.pca_joint import PCAJointForecaster

model = PCAJointForecaster(mode="difference")  # "log_price" for FX
forecast = model.forecast(cases[0].card)
draws = forecast.draws                  # [1000, assets * horizons]
paths = forecast.paths                  # [1000, max_horizon, assets]
diagnostics = forecast.diagnostics
```

The diagnostics include PCA explained-variance ratios, the standardized
loading matrix, residual covariance in native daily-change/return units,
historical and simulated marginal volatility and cross-asset correlation, and
the 1%, 5%, 95%, and 99% quantiles of each asset/horizon forecast cell.

Run the UST/G10 example after cloning the public Track 2 repository and
installing this package:

```powershell
$Track2 = 'C:\path\to\track2-forecasting-public'
python experiments\bootstrap_backtest.py `
  --rates-panel "$Track2\units\t2-F1-hawkish-cut-2024\rates_daily.parquet" `
  --fx-panel "$Track2\units\t2-F3-election-2024-joint\g10_fx_daily.parquet"
```

It evaluates 21- and 63-business-day forecasts at four historical as-of dates
for UST 2Y/5Y/10Y/30Y and EUR/GBP/AUD/NZD. The supplied panels contain later
observations for constructing retrospective outcomes; each forecast still
receives only its own pre-as-of history. The script prints CRPS, variogram,
tail pinball loss, and composite score per case.

Compare the bootstrap and PCA candidate on exactly those same pseudo-cards:

```powershell
python -m experiments.compare_joint_forecasters `
  --rates-panel "$Track2\units\t2-F1-hawkish-cut-2024\rates_daily.parquet" `
  --fx-panel "$Track2\units\t2-F3-election-2024-joint\g10_fx_daily.parquet"
```

It prints mean component scores by panel and model, plus origins where PCA has
a worse composite. It saves per-origin scores and all PCA diagnostics to
`reports/pca_joint_diagnostics.json` by default. Both methods use a 504-row
lookback, 1,000 draws, the same four cutoffs, and the same 21/63-business-day
grid. Raw means are meaningful within one panel/grid, not between UST and FX.

Reports use the [shared toolkit's CRPS and variogram functions](https://github.com/Agenthon-2026/Agenthon2026-public)
and the [Track 2 scorer's pinball tail loss](https://github.com/Agenthon-2026/track2-forecasting-public).
The single-cell score uses the published weight redistribution. Scores are
**raw and unranked**: official baseline reference scales and sealed outcomes
are unavailable locally, and raw scores from different panels or units should
not be compared as leaderboard scores. The summary averages only the cases
in one invocation.

Run tests with `python -m pytest` after installing dependencies. The in-memory
unit tests can also run with `python -m unittest discover -s tests`.
