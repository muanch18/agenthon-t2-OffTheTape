"""Prepare the official Track 2 descriptor and seal a CodaBench upload zip.

Requires qfbench2-common 2.4.4+ installed on the host. The toolkit prompts for
the Team Key; this script never accepts it as a command-line argument.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import sysconfig
from importlib import resources
from importlib.metadata import version
from pathlib import Path


def image_fields(reference: str) -> dict[str, str]:
    match = re.fullmatch(r"([a-zA-Z0-9.-]+)/(\S+)@(sha256:[0-9a-f]{64})", reference)
    if not match:
        raise ValueError("image must be registry/repository@sha256:<64 lowercase hex>")
    registry, repository, digest = match.groups()
    if ":" in repository.split("/")[-1]:
        raise ValueError("use an immutable digest, not an image tag")
    return {"registry": registry, "repository": repository, "digest": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True,
                        help="anonymously pullable registry/repository@sha256:<digest>")
    parser.add_argument("--team-number", required=True, type=int)
    parser.add_argument("--license", required=True, help="OSI license identifier for your code")
    parser.add_argument("--phase", choices=("dev", "final"), default="dev")
    parser.add_argument("--descriptor", type=Path, default=Path("submission.json"))
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    args = parser.parse_args(argv)
    if args.team_number <= 0:
        parser.error("team number must be positive")
    try:
        image = image_fields(args.image)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        installed = tuple(map(int, version("qfbench2-common").split(".")[:3]))
        if installed < (2, 4, 4):
            parser.error("install qfbench2-common 2.4.4 or newer on the host")
        from qfbench2_common.contracts.descriptor import seal_descriptor_digest
        fixture = resources.files("qfbench2_common").joinpath(
            f"contracts/fixtures/c5/forecasting_{args.phase}.json")
        descriptor = json.loads(fixture.read_text(encoding="utf-8"))
    except (ImportError, FileNotFoundError) as exc:
        parser.error(f"the current qfbench2-common toolkit is required: {exc}")
    # The published fixture carries a sample team_id. The official packer derives
    # the real value from the Team Number and hidden Team Key; retaining the
    # fixture value makes it refuse an otherwise valid submission.
    descriptor.pop("team_id", None)
    descriptor.update({"image": image, "image_access": "public", "license": args.license,
                       "category": "api", "models": [], "track": "forecasting", "phase": args.phase,
                       "competition_id": f"agenthon2026-forecasting-{args.phase}"})
    args.descriptor.parent.mkdir(parents=True, exist_ok=True)
    args.descriptor.write_text(json.dumps(seal_descriptor_digest(descriptor), indent=2) + "\n",
                               encoding="utf-8")
    scripts_dir = Path(sysconfig.get_path("scripts"))
    command = scripts_dir / ("qfbench2.exe" if sys.platform == "win32" else "qfbench2")
    if not command.is_file():
        parser.error(f"qfbench2 command is missing from {scripts_dir}")
    subprocess.run([str(command), "submission", "pack",
                    "--descriptor", str(args.descriptor), "--team-number", str(args.team_number),
                    "--out", str(args.out)], check=True)
    print(f"Packed {args.out}. Verify anonymous image pull before uploading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
