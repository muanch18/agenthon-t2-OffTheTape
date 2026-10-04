# Off The Tape: historical Track 2 evaluation

This repository currently provides a local historical evaluation harness. It creates
pseudo-cards from a full historical panel, passes only observations through each
as-of date to a forecasting callback, constructs later outcomes, and scores saved
joint draws. A deliberately simple numeric baseline samples whole historical
cross-asset change rows and produces 1,000 joint scenarios. A separate text
research pipeline extracts macro state; it does not alter either numeric model.

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

The text pipeline reads each practice unit's `text/corpus_index.json` and only
documents whose indexed publication timestamp is on or before the requested
as-of date. [Corpus inspection notes](docs/text_corpus_notes.md) describe the
actual file types and sizes. `DocumentSelector` ranks eligible documents by
recency, source/type, and family/panel keywords, then sends bounded excerpts
to an OpenAI-style `MODEL_ENDPOINT/chat/completions`. Set `MODEL_NAME` for the
organizer model. The prompt asks for macro and regime signals only; the strict
`MacroState` schema and cited document IDs are validated before accepting a
response. Invalid output or an unavailable endpoint yields a neutral state.

```python
from datetime import date
from pathlib import Path
from off_the_tape.macro_extractor import extract_macro_state, extract_with_prior

unit = Path(r"C:\path\to\track2-forecasting-public\units\t2-F1-cpi-glidepath-2023")
result = extract_macro_state(unit, asof=date(2023, 7, 12),
                             family="T2-F1", panel_id="rates_daily")
comparison = extract_with_prior(unit, asof=date(2023, 7, 12),
                                window_days=60, family="T2-F1",
                                panel_id="rates_daily")
neutral = extract_macro_state(unit, asof=date(2023, 7, 12),
                              text_enabled=False)
```

Validated outputs are cached under `.cache/macro_state/` using the full eligible
corpus content and metadata hashes, as-of date, selected excerpts, model name,
seed, prompt/schema/selector versions, and selection settings. Thus a changed
document, even one not selected for the prompt, invalidates the cache. The
`text_enabled=False` path bypasses document loading and returns neutral state.

To inspect F1–F4 examples locally, run:

```powershell
python -m experiments.text_state_diagnostics --track2-root "$Track2" --offline-heuristic
```

The explicit offline heuristic is a lexical **pipeline proxy**, not an LLM
quality estimate or automatic fallback. With a configured `MODEL_ENDPOINT`,
omit `--offline-heuristic` to test the model. The script prints period states,
configurable prior-window deltas, and sanity warnings and writes evidence to
`reports/text_state_diagnostics.json`. It does not call the forecaster or
modify forecast distributions.

The first runnable Track 2 agent composes PCA joint numeric paths with a
bounded, family-aware macro conditioner. The `forecast` entry point parses the
unit's `card.toml`, loads the exact target assets and horizons, selects dated
text, and writes `forecast.parquet`, `forecast_meta.json`, and
`forecast_rationale.md`. Without `MODEL_ENDPOINT`, extraction falls back to a
neutral state and the numeric forecast remains valid. `--numeric-only` bypasses
text inference and yields exactly the same base draws at a fixed seed.

```powershell
forecast --panels "$Track2\units\t2-F3-election-2024-joint\panels" `
  --text "$Track2\units\t2-F3-election-2024-joint\text" `
  --asof 2024-10-31 --out reports\rehearsal_f3\forecast.parquet
```

`MacroConditioner` exposes separately ablatable mean, volatility, skew,
shock, and common-factor adjustments. Its defaults are deliberately small and
confidence-gated. Run the chronological local comparison with:

```powershell
python -m experiments.macro_ablation --track2-root "$Track2"
```

The method, held-out ablations, limitations, and public-unit gate checks are
recorded in [conditioning backtest notes](docs/conditioning_backtest_notes.md).
The current offline-proxy experiment does not support enabling text
conditioning as the competition default.

## Frozen submission candidate and rehearsal

The current candidate is frozen in
[`candidate.py`](src/off_the_tape/candidate.py): PCA joint paths, 1,000 draws,
root seed 2026, and text conditioning disabled. The ordinary `forecast` command
therefore makes no model call. Use `--text-conditioned` only for explicit
research runs; `--numeric-only` remains accepted for clarity. The contract
review and corrected discrepancies are in
[`docs/contract_audit.md`](docs/contract_audit.md).

Before serialization, the agent validates the complete asset/horizon grid,
draw floor, contiguous IDs, dtypes, finite values, non-degenerate marginals,
broad magnitude bounds, and positive FX levels. PCA failures use a deterministic
joint empirical fallback with a volatility floor. Every stochastic production
component derives from the CLI's root `--seed`.

Run every public unit and generate `reports/submission_readiness.md` with:

```powershell
python scripts\smoke_all.py --track2-root "$Track2"
```

This records per-unit gates, runtime stages, fallbacks, calls, and token counts.
On Windows it calls the Track 2 scorer's official g0–g3 functions directly,
because the shared smoke wrapper requires POSIX no-follow directory flags.

The submission [`Dockerfile`](Dockerfile) uses Python 3.13, exact runtime
dependency pins, a non-root user, no runtime installation, and the required
`qfbench2.interface_version="2.0"` label. The official scorer is installed
only in [`Dockerfile.verifier`](Dockerfile.verifier), keeping it out of the
submitted image. The public Git repository holds Parquet panels at each unit's
root, while the evaluator mounts them under `panels/`. Stage the public units
from committed Git bytes, which also preserves their manifest checksums on
Windows, then run the network-isolated rehearsal:

```powershell
python scripts\stage_public_units.py --public-repo "$Track2" --out .cache\staged-public
python scripts\rehearse_submission.py --track2-root .cache\staged-public --all
```

The rehearsal runs the agent and official `qfbench2-smoke` wrapper in separate
Linux containers with `--network=none`. Use `--skip-build` only when both
images already exist. Use a new staging directory for each run. The complete
new-image result is in [the rehearsal record](docs/new_image_rehearsal.md).

## Submission packaging

The current [submission contract](https://github.com/Agenthon-2026/track2-forecasting-public/blob/main/SUBMISSION_CLI.md)
requires an anonymously pullable `linux/amd64` image by immutable registry
digest. Build, rehearse, and push the **runtime** image; the verifier image is
local only. Confirm the registry package is public and test an anonymous pull.
Build for the required architecture and push to your own public registry:

```powershell
docker buildx build --platform linux/amd64 --push -t ghcr.io/ORG/IMAGE:TAG .
```

Record the immutable registry digest from the push result. Then install the
[submission toolkit](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.4.4/starter-packs/track2/SUBMISSION-DESCRIPTOR.md)
at version 2.4.4 on the host and run:

```powershell
python scripts\package_submission.py --image ghcr.io/ORG/IMAGE@sha256:DIGEST --team-number N --license YOUR_SPDX_LICENSE
```

On Windows when Python 3.13 is unavailable on `PATH`, build the local packager
container instead and run the same script through it. The bind mount lets the
toolkit write `submission.zip` into this repository:

```powershell
docker build --file Dockerfile.packager --build-arg RUNTIME_IMAGE=off-the-tape:candidate --tag off-the-tape:packager .
docker run --rm -it --mount "type=bind,src=$((Get-Location).Path),dst=/workspace" off-the-tape:packager python scripts/package_submission.py --image ghcr.io/ORG/IMAGE@sha256:DIGEST --team-number N --license YOUR_SPDX_LICENSE
```

Use your actual image digest, team number, and code license. The script copies
the official Track 2 descriptor fixture, declares the numeric-only candidate
with `models: []`, and invokes the toolkit's `submission pack`. The toolkit
prompts privately for the Team Key and writes `submission.zip` with only
`submission.json` and `team-claim.json`. Neither file is tracked in Git.
Packaging does not upload the zip or consume a submission attempt.
The MIT-licensed runtime image was pushed to Docker Hub as
`docker.io/anushm22/off-the-tape@sha256:99a9fe62fdf906b6627a1f8fcec0d5b42ddc627abec3edc26c63e4f4df7abc2e`
on 2026-09-30. It is `linux/amd64`, 495,782,725 bytes (495.8 MB), and an
unauthenticated manifest lookup succeeded. Use this exact digest for Team 404's
Development submission. The local packager command is:

```powershell
docker run --rm -it --mount "type=bind,src=$((Get-Location).Path),dst=/workspace" off-the-tape:packager python scripts/package_submission.py --image docker.io/anushm22/off-the-tape@sha256:99a9fe62fdf906b6627a1f8fcec0d5b42ddc627abec3edc26c63e4f4df7abc2e --team-number 404 --license MIT
```

The prior combined image measured 755.5 MB.
All 104 public units produced the required three files under `--network=none`
and passed `qfbench2-smoke` in the verifier image. The earlier
`reports/submission_readiness.md` describes the prior combined image.

Run tests with `python -m pytest` after installing dependencies. The in-memory
unit tests can also run with `python -m unittest discover -s tests`.
