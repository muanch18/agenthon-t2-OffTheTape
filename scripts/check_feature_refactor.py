"""Capture or compare forecast-value fingerprints without scoring outcomes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from off_the_tape.agent import TaskSpec, Track2Agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    rows = {}
    for card_path in sorted(args.units.glob("*/card.toml")):
        task = TaskSpec.from_card(card_path)
        forecast = Track2Agent().forecast(task, card_path.parent, card_path.parent/"text",
                                          task.asof, seed=2026, numeric_only=True)
        rows[task.unit_id] = {"shape": list(forecast.draws.shape),
                              "values_sha256": sha256(forecast.draws.tobytes()).hexdigest(),
                              "fallback_used": forecast.fallback_used}
    if not rows:
        raise ValueError("no public units found")
    if args.capture:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with args.baseline.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(rows, indent=2)+"\n")
        print(f"Captured {len(rows)} public-unit forecast fingerprints")
    else:
        baseline = json.loads(args.baseline.read_text())
        if rows != baseline:
            mismatches = [key for key in set(rows)|set(baseline) if rows.get(key) != baseline.get(key)]
            raise ValueError(f"forecast fingerprints differ: {mismatches}")
        print(f"Exact forecast-value and fallback parity: {len(rows)}/{len(rows)} public units")


if __name__ == "__main__":
    main()
