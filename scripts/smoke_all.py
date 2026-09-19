"""Run the frozen candidate over every public unit and write readiness reports."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from off_the_tape.agent import TaskSpec, Track2Agent, classify_failure
from off_the_tape.candidate import CURRENT_CANDIDATE


def _gate_results(unit: Path, output: Path) -> dict[str, object]:
    try:
        from qfbench2_track_forecasting.scoring import (
            _g0_integrity, _g1_schema, _g2_cutoff_resource, _g3_domain_semantics, hydrate_ctx,
        )
        context = {"unit_dir": unit, "output_dir": output}
        hydrate_ctx(context)
        return {name: function(context).passed for name, function in (
            ("g0", _g0_integrity), ("g1", _g1_schema),
            ("g2", _g2_cutoff_resource), ("g3", _g3_domain_semantics),
        )}
    except Exception as exc:
        return {"g0": False, "g1": False, "g2": False, "g3": False,
                "gate_error": f"{type(exc).__name__}: {exc}"[:500]}


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def run(root: Path, output_root: Path, *, limit: int | None = None) -> dict[str, object]:
    startup = time.perf_counter()
    agent = Track2Agent(n_draws=CURRENT_CANDIDATE.n_draws, candidate=CURRENT_CANDIDATE)
    startup_seconds = time.perf_counter() - startup
    units = [path for path in sorted((root / "units").iterdir()) if (path / "card.toml").is_file()]
    if limit:
        units = units[:limit]
    rows = []
    for unit in units:
        started = time.perf_counter()
        row = {"unit_id": unit.name, "family": None, "target_type": None,
               "assets": [], "horizons": [], "runtime_seconds": 0.0,
               "exit_code": 1, "g0": False, "g1": False, "g2": False, "g3": False,
               "fallback_used": False, "model_calls": 0, "input_tokens": 0,
               "output_tokens": 0, "n_draws": CURRENT_CANDIDATE.n_draws,
               "error_message": None, "failure_category": None, "timings": {}}
        out_dir = output_root / unit.name
        try:
            task = TaskSpec.from_card(unit / "card.toml")
            row.update(family=task.family, target_type=task.target_type,
                       assets=list(task.assets), horizons=list(task.horizons))
            forecast = agent.forecast(task, unit / "panels", unit / "text", task.asof,
                                      CURRENT_CANDIDATE.root_seed, numeric_only=None)
            serialization = time.perf_counter()
            agent.write(forecast, out_dir / "forecast.parquet")
            serialization_seconds = time.perf_counter() - serialization
            gates = _gate_results(unit, out_dir)
            row.update(gates)
            row.update(exit_code=0 if all(gates.get(f"g{i}") for i in range(4)) else 1,
                       fallback_used=forecast.fallback_used,
                       failure_category=forecast.failure_category,
                       model_calls=int(forecast.diagnostics.get("model_calls", 0)),
                       input_tokens=int(forecast.diagnostics.get("input_tokens", 0)),
                       output_tokens=int(forecast.diagnostics.get("output_tokens", 0)),
                       timings={**forecast.diagnostics, "output_serialization_seconds": serialization_seconds,
                                "startup_seconds": startup_seconds})
            if row["exit_code"]:
                row["error_message"] = gates.get("gate_error", "one or more admissibility gates failed")
                row["failure_category"] = "schema"
        except Exception as exc:
            row["error_message"] = f"{type(exc).__name__}: {exc}"[:500]
            row["failure_category"] = classify_failure(exc)
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(f"{row['unit_id']}: {'PASS' if row['exit_code'] == 0 else 'FAIL'} "
              f"{row['runtime_seconds']:.2f}s fallback={row['fallback_used']}")
    runtimes = [r["runtime_seconds"] for r in rows]
    summary = {
        "candidate": CURRENT_CANDIDATE.__dict__,
        "contract": {"interface_version": "2.0", "track_scorer": "3.1.0",
                     "common": "v2.4.0 tag (package metadata 2.3.1)",
                     "minimum_draws": 200, "configured_draws": CURRENT_CANDIDATE.n_draws},
        "host": {"platform": platform.platform(), "docker_available": shutil.which("docker") is not None},
        "public_units_tested": len(rows),
        "public_units_passed": sum(r["exit_code"] == 0 for r in rows),
        "admissibility_pass_rate": sum(r["exit_code"] == 0 for r in rows) / max(len(rows), 1),
        "runtime_seconds": {"median": statistics.median(runtimes) if runtimes else 0,
                            "p90": _percentile(runtimes, .9), "max": max(runtimes, default=0)},
        "model_calls": {"median": statistics.median([r["model_calls"] for r in rows]) if rows else 0,
                        "max": max((r["model_calls"] for r in rows), default=0)},
        "input_tokens": {"median": statistics.median([r["input_tokens"] for r in rows]) if rows else 0,
                         "max": max((r["input_tokens"] for r in rows), default=0)},
        "fallback_activations": sum(bool(r["fallback_used"]) for r in rows),
        "failures": dict(Counter(r["failure_category"] or "none" for r in rows if r["exit_code"])),
        "rows": rows,
    }
    return summary


def readiness_markdown(report: dict[str, object]) -> str:
    rows = report["rows"]
    docker = report.get("docker")
    image_size = (f"{docker['image_size_bytes'] / 1_000_000:.1f} MB"
                  if docker else "not measured (Docker unavailable)")
    lines = ["# Submission readiness", "", "## Candidate", "",
             "The frozen candidate is numeric-only PCA with 1,000 joint draws. Text conditioning is retained as an opt-in research mode and disabled by default because it did not improve held-out aggregate score.", "",
             "## Public-unit result", "",
             f"- Public units tested: {report['public_units_tested']}",
             f"- Public units passed: {report['public_units_passed']}",
             f"- Admissibility pass rate: {report['admissibility_pass_rate']:.1%}",
             f"- Median runtime: {report['runtime_seconds']['median']:.3f} s",
             f"- P90 runtime: {report['runtime_seconds']['p90']:.3f} s",
             f"- Worst runtime: {report['runtime_seconds']['max']:.3f} s",
             f"- Median LLM calls: {report['model_calls']['median']}",
             f"- Maximum LLM calls: {report['model_calls']['max']}",
             f"- Median input tokens: {report['input_tokens']['median']}",
             f"- Maximum input tokens: {report['input_tokens']['max']}",
             f"- Numeric fallback activations: {report['fallback_activations']}",
             f"- Docker image size: {image_size}",
             f"- Cold Python import/process time: {report['host'].get('cold_import_seconds', 0):.3f} s", "",
             "## Runtime components", "", "| Component | Median s | P90 s | Maximum s |", "|---|---:|---:|---:|"]
    timing_keys = (
        ("input_loading_and_preprocessing_seconds", "Input loading + preprocessing"),
        ("numeric_fit_seconds", "Numeric fit"), ("numeric_simulation_seconds", "Numeric simulation"),
        ("text_selection_seconds", "Text selection"), ("llm_seconds", "LLM request"),
        ("llm_parsing_seconds", "LLM parsing"), ("conditioning_seconds", "Conditioning"),
        ("output_serialization_seconds", "Output serialization"),
    )
    for key, label in timing_keys:
        values = [float(row["timings"].get(key, 0) or 0) for row in rows]
        lines.append(f"| {label} | {statistics.median(values):.4f} | {_percentile(values, .9):.4f} | {max(values, default=0):.4f} |")
    lines.extend(["", "## Coverage", "", "| Group | Passed | Total | Rate |", "|---|---:|---:|---:|"])
    for field, groups in (("family", ("T2-F1", "T2-F2", "T2-F3", "T2-F4")),
                          ("target_type", ("level", "log_return"))):
        for group in groups:
            selected = [row for row in rows if row[field] == group]
            passed = sum(row["exit_code"] == 0 for row in selected)
            lines.append(f"| {group} | {passed} | {len(selected)} | {passed / max(len(selected), 1):.1%} |")
    if docker:
        runtime = docker["runtime_seconds"]
        lines.extend(["", "## Docker rehearsal", "",
                      f"- Image: `{docker['image']}`",
                      f"- Docker Desktop / engine: {docker['desktop_version']} / {docker['engine_version']}",
                      f"- Public units passed: {docker['public_units_passed']}/{docker['public_units_tested']}",
                      f"- Median / P90 / worst container rehearsal: {runtime['median']:.3f} / {runtime['p90']:.3f} / {runtime['max']:.3f} s",
                      f"- Network mode: `{docker['network_mode']}`",
                      f"- Container identity: `{docker['container_user']}` (UID {docker['container_uid']})",
                      f"- Official scorer: {docker['official_smoke']}"])
    lines.extend(["", "## Known issues", ""])
    failures = [row for row in rows if row["exit_code"]]
    if failures:
        lines.extend(f"- `{row['unit_id']}` ({row['failure_category']}): {row['error_message']}" for row in failures)
    else:
        lines.append("- No public-unit agent or g0-g3 failures.")
    fallbacks = [row for row in rows if row["fallback_used"]]
    if fallbacks:
        lines.append("- Safe numeric fallback activations: " + ", ".join(
            f"`{row['unit_id']}` ({row['failure_category']})" for row in fallbacks
        ) + ".")
    if report["public_units_tested"] == 104:
        lines.append("- Coverage is all 103 practice units plus the worked exemplar directory.")
    if not docker:
        lines.append("- Docker is not installed on this host, so image build, size, cold start, and network-none container execution remain unverified.")
    lines.extend(["- The host runner invokes the official Track 2 g0-g3 functions directly; the Docker rehearsal runs the full POSIX smoke wrapper inside Linux.",
                  "- Model-endpoint latency and token counts are zero for the frozen numeric-only candidate. Text-conditioned endpoint behavior is covered by unit tests, not a live organizer endpoint.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("reports/public_smoke_outputs"))
    parser.add_argument("--json", type=Path, default=Path("reports/public_smoke.json"))
    parser.add_argument("--readiness", type=Path, default=Path("reports/submission_readiness.md"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run(args.track2_root, args.output_root, limit=args.limit)
    cold_started = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import off_the_tape.agent"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report["host"]["cold_import_seconds"] = time.perf_counter() - cold_started
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    args.readiness.parent.mkdir(parents=True, exist_ok=True)
    args.readiness.write_text(readiness_markdown(report), encoding="utf-8")
    print(f"Passed {report['public_units_passed']}/{report['public_units_tested']}; "
          f"median={report['runtime_seconds']['median']:.2f}s p90={report['runtime_seconds']['p90']:.2f}s "
          f"max={report['runtime_seconds']['max']:.2f}s")
    return 0 if report["public_units_passed"] == report["public_units_tested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
