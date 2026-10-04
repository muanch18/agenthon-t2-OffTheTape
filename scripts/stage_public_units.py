"""Stage public Track 2 Git units for the Docker harness without CRLF conversion.

Git archive supplies the committed bytes, which must match each unit's manifest.
The public repository keeps Parquet panels at each unit's root; the evaluator
mounts a staged tree with those files under ``panels/`` instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def stage(repo: Path, output: Path) -> int:
    repo = repo.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    process = subprocess.Popen(["git", "archive", "HEAD", "units"], cwd=repo,
                               stdout=subprocess.PIPE)
    assert process.stdout is not None
    count = 0
    with process.stdout, tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        output.mkdir(parents=True)
        for member in archive:
            relative = Path(member.name)
            if relative.parts[0] != "units" or ".." in relative.parts:
                raise ValueError(f"unexpected archive path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unexpected archive member: {member.name}")
            if len(relative.parts) == 3 and relative.suffix == ".parquet":
                relative = relative.parent / "panels" / relative.name
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    for manifest_path in (output / "units").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            if "/" not in row["path"] and row["path"].endswith(".parquet"):
                row["path"] = f"panels/{row['path']}"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = stage(args.public_repo, args.out)
    print(f"Staged {count} public units in {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
