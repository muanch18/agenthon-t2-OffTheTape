# Historical cases for Final preparation

This numeric research suite chooses dates without looking at subsequent market
moves. It does not recreate the hidden Final cards or their text corpora.
Scope follows the organizer clarification at
https://github.com/Agenthon-2026/track2-forecasting-public/issues/26
and the Final notice linked there: UST/G10 FX, short business-day horizons,
and no guarantee that an F4 risk becomes a shock.

Run from the repository root with the project Python environment:

```powershell
python -m experiments.final_cases --rates-panel .cache/track2-public/units/t2-F1-hawkish-cut-2024/rates_daily.parquet --fx-panel .cache/track2-public/units/t2-F3-election-2024-joint/g10_fx_daily.parquet
```

The default manifest is `reports/final_cases.json` (ignored research output).
It records source hashes, admitted cases, exclusions, splits, and methodology.
It contains no outcome values. `hydrate(panel, spec)` materializes a case on
demand, separating the historical `card` from its `realized` vector. Never pass
the HistoricalCase wrapper to a model; pass only its card or pre-cutoff panels.
Full panel files are retrospective evaluation inputs, not production assets.

Dates are every 20 weekdays from January 2005 through October 2024. Five shapes
cover single rates, single FX, a rates curve, FX basket, and mixed rates/FX.
Single assets rotate deterministically through six UST tenors and ten FX series.
Horizons are deterministically selected from 2–12 weekdays. Joint cases have
two horizons. The seed controls specifications, not market outcomes.

Train is 2005–2017, validation 2018–2021, holdout 2022–2024. Cases whose targets
cross a split boundary are excluded. Require 505 complete pre-origin rows.
Targets must exist on the exact pandas BDay date; missing targets are logged
and excluded, never forward-filled. Historical rows align on complete dates.
Holiday exclusions can affect representativeness; inspect the exclusion log.

The 20-weekday origin spacing exceeds the maximum 12-day horizon, so target
windows at distinct origins do not overlap. Cases at the same origin share
outcomes and must be grouped by origin in uncertainty estimates. Historical
training windows overlap. Do not treat the case count as independent samples.

Keep holdout scores uninspected until candidate settings are fixed. Do not
optimize the seed, calendar start, assets, or shapes against holdout results.
This step builds and verifies cases only; it does not select a better model,
implement revised official scoring, or change the frozen submission.

Without dated historical text, these cases have no validated family semantics.
The hydration API uses T2-F3 solely to satisfy the existing numeric interface;
do not use it to evaluate family-specific or text-conditioned strategies.
Mixed levels also have different units: avoid interpreting raw aggregate
scores as leaderboard estimates. Official normalized scoring is a separate task.

## Initial generation (2026-10-16)

The cached public panels produced 1,073 admitted cases: 691 training,
226 validation, and 156 holdout cases. All admitted cases hydrated successfully
with the forecast-history cutoff check. Of 1,295 scheduled cases, 151 lacked
an exact origin observation, 69 lacked an exact target, and two crossed a split
boundary. No cases were excluded based on move size or direction. The manifest
retains every exclusion and source file hash. No holdout model scores have been
computed. Two new regression tests and 14 existing historical tests passed.

## Holdout state after 2026-10-16 evaluation

The fixed 156 holdout cases were subsequently evaluated under the separately
committed protocol `9113289`. This holdout is now consumed and must not be
described as untouched in future research. See `frozen_holdout_results.md` for
the frozen comparisons, findings and limitations. The original roster was not
changed and the results were not used to retune candidate definitions.
