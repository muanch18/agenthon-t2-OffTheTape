# Off The Tape: historical Track 2 evaluation

This repository currently provides a local historical evaluation harness. It creates
pseudo-cards from a full historical panel, passes only observations through each
as-of date to a forecasting callback, constructs later outcomes, and scores saved
joint draws. The LLM and numeric forecasting components in [the plan](docs/plan.md)
are not implemented yet.

The input panel must be long-format Parquet with `date`, `asset`, and `value`
columns, as in the [Track 2 public repository](https://github.com/Agenthon-2026/track2-forecasting-public).
An optional `panel_id` column is ignored. The public practice units contain
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

cases = historical_cases(panel, assets=["UST_2Y", "UST_10Y"],
                         horizons=[21], target_type="level",
                         origins=["2018-01-31"])
results = evaluate(cases, lambda card: make_joint_draws(card))
```

`make_joint_draws` receives a `PseudoCard` containing only history through
`card.asof`, and returns a NumPy matrix with shape `(n_draws,
len(card.assets) * len(card.horizons))`. Columns follow asset-major,
horizon-minor order. For `level`, the outcome is the value after `h` observed
panel business days. For `log_return`, panel values are daily simple returns;
the outcome is the sum of `log(1 + r)` over the following `h` observations.
All requested assets must be present on every panel date. Missing or incomplete
origins fail instead of being silently excluded.

Reports use the [shared toolkit's CRPS and variogram functions](https://github.com/Agenthon-2026/Agenthon2026-public)
and the [Track 2 scorer's pinball tail loss](https://github.com/Agenthon-2026/track2-forecasting-public).
The single-cell score uses the published weight redistribution. Scores are
**raw and unranked**: official baseline reference scales and sealed outcomes
are unavailable locally, and raw scores from different panels or units should
not be compared as leaderboard scores. The summary averages only the cases
in one invocation.

Run tests with `python -m pytest` after installing dependencies. The in-memory
unit tests can also run with `python -m unittest discover -s tests`.
