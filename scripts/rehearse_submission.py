"""Build/run the actual image and invoke qfbench2-smoke; exits on any failure."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

REPRESENTATIVES = (
    "t2-F1-greater-confidence-2024",
    "t2-F2-cut-sizing-2024",
    "t2-F3-election-2024-joint",
    "t2-F4-jpy-crowding-2024",
    "t2-F4-vaccine-timeline-2020",  # log_return
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track2-root", type=Path, required=True)
    parser.add_argument("--image", default="off-the-tape:candidate")
    parser.add_argument("--output-root", type=Path, default=Path("reports/docker_rehearsal"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if shutil.which("docker") is None:
        parser.error("docker is not installed or not on PATH")
    output_root = args.output_root if args.output_root.is_absolute() else Path.cwd() / args.output_root
    if not args.skip_build:
        run(["docker", "build", "--tag", args.image, "."])
    units = sorted(path for path in (args.track2_root / "units").iterdir()
                   if (path / "card.toml").is_file()) if args.all else [
        args.track2_root / "units" / name for name in REPRESENTATIVES
    ]
    records = []
    for unit in units:
        with (unit / "card.toml").open("rb") as stream:
            import tomllib
            card = tomllib.load(stream)
        asof = card["provenance"]["data_cutoff"]
        output = output_root / unit.name
        output.mkdir(parents=True, exist_ok=True)
        for name in ("forecast.parquet", "forecast_meta.json", "forecast_rationale.md"):
            (output / name).unlink(missing_ok=True)
        started = time.perf_counter()
        run(["docker", "run", "--rm", "--network=none", "--env", "QFBENCH_NETWORK=none",
             "--mount", f"type=bind,src={unit.resolve()},dst=/input,readonly",
             "--mount", f"type=bind,src={output},dst=/output",
             args.image, "forecast", "--panels", "/input/panels", "--text", "/input/text",
             "--asof", asof, "--out", "/output/forecast.parquet"])
        # Run the official wrapper in Linux. Its no-follow manifest walk uses
        # POSIX directory-descriptor semantics that are unavailable on Windows.
        run(["docker", "run", "--rm", "--network=none",
             "--mount", f"type=bind,src={unit.resolve()},dst=/input,readonly",
             "--mount", f"type=bind,src={output},dst=/output,readonly",
             args.image, "qfbench2-smoke", "/input", "/output", "--track", "forecasting"])
        records.append({"unit_id": unit.name, "runtime_seconds": time.perf_counter() - started,
                        "outputs": sorted(path.name for path in output.iterdir())})
        print(f"PASS {unit.name} {records[-1]['runtime_seconds']:.2f}s")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rehearsal.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
