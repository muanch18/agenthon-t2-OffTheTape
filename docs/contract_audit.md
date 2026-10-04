# Track 2 contract and hardening audit

Audited on 2026-09-15 against public Track 2 commit
`83c6dc036bec03862b78e093bf8804d98964dad5` and shared toolkit checkout
`09873cad2f3ea1171acd4e19cd8c10b3cb6a126f` / published `v2.4.0`.

## Frozen candidate

`CURRENT_CANDIDATE` is numeric-only `PCAJointForecaster`, 1,000 draws, root
seed 2026. Text mean, volatility, skew, shock, and joint conditioning are all
disabled by default. The previous chronological held-out ablation had a full
conditioner composite ratio of 1.0017 versus numeric-only 1.0000. Text remains
available only through explicit `--text-conditioned` research mode.

## Authoritative contract

| Item | Current requirement | Implementation |
|---|---|---|
| CLI | `forecast --panels … --text … --asof YYYY-MM-DD --out …` | `forecast` console script; accepts optional leading verb |
| Inputs | read-only `/input/panels`, `/input/text`; as-of must match card | panel rows are truncated at as-of; corpus uses indexed release timestamps; card cutoff is exact |
| Outputs | exactly three files | Parquet, metadata JSON, and non-empty rationale |
| Parquet | `draw:int32`, `asset:string`, `horizon:int32`, `value:float64` | validated before serialization and by g0–g3 |
| Metadata | `unit_id`, `asof`, ordered `asset_ids`, ordered `horizons`, `samples`, `n_draws` | populated from `card.toml` |
| Draws | minimum 200; complete joint grid | frozen at 1,000; contiguous draw IDs and exact grid validated |
| Targets | `level` and `log_return` | read per card; rates/macro use changes, FX uses log prices, factors return cumulative log return |
| Families | F1–F4 | parsed and preserved; production candidate does not condition |
| Network | restricted; only `MODEL_ENDPOINT`/`MODEL_NAME`; local smoke is none | numeric default makes no calls; `QFBENCH_NETWORK=none` disables endpoint; urllib honors proxy environment |
| Model budget | 1M input / 100k output tokens per unit | zero in candidate; optional text uses one bounded request and 18k-character selection budget |
| Image | Python 3.13; label `qfbench2.interface_version="2.0"` | Runtime Dockerfile uses Python 3.13 and declares interface, track, and verb labels |
| Pins | exact production libraries; scorer for rehearsal | Runtime pins NumPy, pandas, and PyArrow; separate verifier pins toolkit `v2.4.0` and Track 2 3.1.0 at audited commit |

## Discrepancies corrected

- The earlier CLI conditioned on text by default even though held-out evidence
  favored numeric-only. The frozen default now follows the supported result.
- Dependency ranges allowed drift. Runtime libraries are pinned in the
  Dockerfile; the historical-evaluation dependencies are pinned in
  `pyproject.toml`, and the official scorer is isolated in Dockerfile.verifier.
- The agent previously propagated PCA/numerical failures. It now uses a joint
  empirical fallback with a volatility floor and deterministic component seeds.
- Output checks previously relied mostly on the scorer. The agent now checks
  finite values, exact grid and dtypes, contiguous draws, duplicates, variance,
  broad magnitude limits, and positive FX levels before writing.
- Endpoint timeout is now 20 seconds, offline mode suppresses endpoint use, and
  extraction records calls, approximate tokens, selection, parsing, and latency.
- Public-set robustness and runtime were previously sampled. The batch runner
  now records every unit and invokes the official Track 2 g0–g3 functions.

The official `qfbench2-smoke` wrapper cannot securely walk manifests on
Windows because POSIX `O_NOFOLLOW` and `O_DIRECTORY` are unavailable. Direct
official g0–g3 calls validate outputs here; the wrapper remains a required step
in the Linux Docker rehearsal.
