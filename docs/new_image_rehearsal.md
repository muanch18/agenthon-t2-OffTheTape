# New runtime image: public-unit rehearsal

Run on 2026-09-29 using Docker Desktop's Linux engine. The public Track 2
checkout was at `28a6cae9674f69e63a07a19165a9217e85eacfff`. The staging
script moved root Parquet inputs to `panels/` and kept the Git blob bytes so
the manifests verified on Windows.

| Check | Result |
|---|---:|
| Public units, including worked exemplar | 104/104 admissible |
| Units with exactly the three required outputs | 104/104 |
| Official gate | `qfbench2-smoke` in Linux verifier image |
| Network | `none` for agent and verifier |
| Runtime image | `off-the-tape:candidate`, `linux/amd64`, user `runner` |
| Local image ID | `sha256:d9cd70ffd1de85fe1359990c9b95ccfb1e69d7545479b9fd4cc2d1a5996f3824` |
| Runtime image size | 495,768,397 bytes (495.8 MB) |
| Median / P90 / worst two-container rehearsal | 10.44 / 11.25 / 53.61 seconds |

The verifier image extends the runtime image with the pinned official scorer;
only the runtime image is intended for submission. Two units took about 53
seconds; the cause of these wall-clock outliers was not isolated. The times
include both agent and verifier container launches and do not
estimate the evaluator's per-unit forecast runtime. Public smoke gates test
admissibility, not forecast accuracy on sealed outcomes.

The detailed local records and outputs are under
`.cache/new-image-rehearsal/` and are ignored by Git. Reproduce with:

```powershell
python scripts\stage_public_units.py --public-repo "$Track2" --out .cache\staged-public-new
python scripts\rehearse_submission.py --track2-root .cache\staged-public-new --all --skip-build
```

The table records the pre-license rebuild. On 2026-09-30 the runtime image was
rebuilt with the MIT `LICENSE` file but no forecast-code changes, then tagged
and pushed as
`docker.io/anushm22/off-the-tape@sha256:99a9fe62fdf906b6627a1f8fcec0d5b42ddc627abec3edc26c63e4f4df7abc2e`.
The rebuilt image is `linux/amd64`, 495,782,725 bytes, and passed the worked
exemplar through the official `qfbench2-smoke` gate with `--network=none`.
An unauthenticated manifest lookup confirmed public access to the digest.
