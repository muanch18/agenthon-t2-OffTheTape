# Shared numeric feature layer — 2026-10-23

The initial implementation centralizes cutoff handling, panel alignment,
step transformations and volatility definitions. It is a small Python batch
layer, not a general event-driven planner. It does not change the production
candidate or enable the volatility blend. SETAR work stays on its own branch.

## API

```python
from datetime import date
from off_the_tape.features import FeatureLayer

features = FeatureLayer(
    panel, assets=("EUR", "GBP"), asof=date(2024, 10, 31),
    future_policy="filter", duplicate_policy="last",
)
history = features.history("log_price")
vol = features.volatility("log_price", lookback_changes=504,
                          recent_changes=60, half_life=20)
blended_sd = vol.variance_blend(source="ewma", recent_weight=0.5)
```

Default policy rejects future rows and duplicate date/asset pairs. The agent
explicitly selects its existing filter/keep-last policies. Both require a
non-null date-only cutoff. Snapshots select requested assets, coerce values as
the prior agent did, and align dates in caller-specified asset order. Missing
cells remain missing: transforms reject incomplete/nonfinite matrices so
existing production fallbacks remain visible. No filling or silent complete-
case row removal occurs in this layer.

Transform modes are level differences, log-price differences and log1p of
daily simple returns. Difference transforms have one fewer row, indexed by
the ending observation date. The simple-return transform includes every input
row. Lookback/recent windows count observed changes, not calendar days. The
layer does not reinterpret monthly horizon keys or impose a new calendar.

Volatility outputs identify assets, cutoff, mode, requested lookback, actual
observation count, recent-window length, exponential half-life and feature
version. Historical and recent sd use ddof=1. Exponential variance uses a
weighted mean and denominator `1-sum(weights**2)`. Blends combine variances,
not standard deviations. The layer returns raw features; clipping sd ratios
and applying adjustments to forecasts remain model decisions.

Snapshots own their data. Frame/wide accessors return copies and NumPy outputs
are read-only. Cached transformations and volatility calculations belong to
one snapshot only; there is no persistent or cross-unit cache. A cutoff does
not prove point-in-time vintage availability: callers remain responsible for
input provenance and publication lags. Text/document features are not included
in this first version.

## Production and research use

The agent constructs one snapshot per numeric panel and passes it into PCA,
which verifies that the prepared input matches its card. PCA uses the same
transformed arrays and all existing model settings and RNG calls. Existing
direct PCA callers automatically use the same layer. Normalization, Gaussian
gap handling, fallback repair rules and historical target construction retain
their existing distinct semantics.

`python -m experiments.feature_diagnostics` demonstrates reuse on five earlier
training examples and compares volatility ratios with the historical formulas.
It never constructs realized targets, predicts outcomes or scores holdout.
Frozen experiment scripts are kept as historical records; do not silently
update their old results or source-hash commitments. To reproduce a pinned
historical run, use its original freeze commit. The frozen holdout runner should
reject this changed source tree; its protocol hashes have not been rewritten.

## Verification

Before modifying the inference code, forecast-value hashes and fallback flags
were captured on all 104 cached public units. After the refactor every unit
matched exactly, at root seed 2026. This checks values, shapes and fallback
decisions, not byte-identical Parquet encoding or an unchanged Docker image.

```powershell
python scripts/check_feature_refactor.py --units .cache/track2-public/units --baseline .cache/features-before.json --capture
# Run the comparison after editing; capture refuses to overwrite the baseline.
python scripts/check_feature_refactor.py --units .cache/track2-public/units --baseline .cache/features-before.json
python -m unittest discover -s tests
python -m experiments.feature_diagnostics
```

Feature tests cover cutoff/duplicate policies, malformed cutoff rejection,
all transformations, independent volatility formulas, snapshot isolation,
read-only results, prepared-input reuse and stale-input rejection. The feature
demo matched historical sd ratios within 2.22e-16 on five training cases.
The full offline verifier test suite passed 65 tests, including six feature
tests. Final public-unit forecast and fallback parity passed 104/104.

This is a correctness and reuse refactor. No runtime speedup is claimed until
measured. No Docker image has been rebuilt or submitted, and no accuracy
experiment or consumed holdout has been rerun as part of this change.
