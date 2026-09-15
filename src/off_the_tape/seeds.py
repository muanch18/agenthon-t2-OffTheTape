"""Stable component seeds derived without Python's randomized hash()."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedManager:
    root_seed: int = 2026

    def derive(self, component: str) -> int:
        digest = hashlib.sha256(f"{self.root_seed}:{component}".encode()).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
