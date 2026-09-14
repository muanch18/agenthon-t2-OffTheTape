# First macro-conditioning experiment

The numeric-only control calls the existing `PCAJointForecaster` unchanged.
`MacroConditioner` transforms its sampled daily paths, then reads every horizon
from each adjusted path. Mean, volatility, skew, shock, and F3 common-factor
components can be ablated separately. Directional exposure rules cover UST
curve tenors, quoted FX pairs, and selected equity factors. Every adjustment is
confidence-gated and capped; a neutral state or failed extraction leaves the
numeric draws byte-for-byte unchanged. F4 shocks are one shared Student-t
mixture event per draw, with probability capped at 20% and size capped at two
historical horizon-volatility units.

`python -m experiments.macro_ablation --track2-root <public-checkout>` uses
public unit panels only for **retrospective pseudo-card outcomes before each
unit's own cutoff**. It never reads sealed answers or later sibling panels for
calibration. For every family and era (pre-2018 training, 2018–2021 validation,
2022–2024 held-out), it chooses the first and last eligible unit in date order.
The pseudo-origin is 30 weekdays before the unit cutoff; target horizons are
10 and 21 weekdays. Units without exact target weekdays are excluded before
scoring. Family labels are inherited from the curated practice units and are
proxies, not proof that a prospective card with the same label would behave
identically. The earlier text view is incomplete because each unit ships a
small curated corpus.

The script uses the explicitly labeled offline lexical proxy for MacroState
because no organizer model endpoint was available. This measures pipeline and
mapping behavior, **not LLM text uplift**. Of 24 selected pseudo-cards, 18 had
proxy states and six fell back to neutral. The only strength search was
`0, 0.5, 1` on pre-2018 training; it selected `1`. Validation and held-out
outcomes played no role in that selection. All numbers below are mean ratios
to the same case's numeric-only **raw, unranked** component score. Lower is
better; ratios are averaged per case so unlike panel units are not added.

| Held-out mode | CRPS | Variogram | Tail | Composite |
|---|---:|---:|---:|---:|
| Numeric only / neutral text | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Mean | 1.0107 | 1.0117 | 1.0064 | 1.0084 |
| Mean + volatility | 1.0130 | 1.0153 | 1.0218 | 1.0008 |
| + skew | 1.0124 | 1.0145 | 1.0203 | 1.0001 |
| + shock | 1.0124 | 1.0145 | 1.0203 | 1.0000 |
| Full family-aware | 1.0139 | 1.0161 | 1.0295 | 1.0017 |

The held-out full conditioner improved the G10 FX composite ratio to 0.9638
but worsened rates to 1.0155 and mixed rates/FX to 1.0211. It worsened the
21-day horizon (1.0229) more than the 10-day horizon (1.0062). Text-defined
stressed cases improved to 0.9744 while text-defined calm cases worsened to
1.0108; these small groups are descriptive, not a calibration target. By
practice family, full conditioning worsened F1, F3, and F4 held-out raw means;
F2 improved slightly. The training F3 gain did not generalize.

There is not enough evidence to replace the numeric-only candidate. The
lexical proxy can miss tabular COT positioning and can confuse FOMC changes;
many pseudo-origins have no eligible curated documents. On mixed-panel F3
cards, the first agent composes separate unchanged per-panel PCA path models,
then applies one shared macro factor. This preserves draw coherence after
conditioning but does not learn numeric rates/FX cross-panel covariance.

The agent generated F1, mixed F3, and F4 public-unit outputs with the required
three files. The Track 2 scorer's actual `g0`–`g3` gate functions passed for
all three. The `qfbench2-smoke` wrapper itself cannot complete on this Windows
host: its manifest verifier refuses to run without POSIX `O_NOFOLLOW` and
`O_DIRECTORY`. Re-run the wrapper in the eventual Linux container.
