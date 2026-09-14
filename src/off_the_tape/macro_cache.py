"""Content-addressed cache for validated macro extractions."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def cache_key(parts: dict[str, Any]) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MacroCache:
    directory: Path

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return entry.get("payload") if isinstance(entry, dict) and entry.get("key") == key else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = self.directory / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps({"key": key, "payload": payload}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
