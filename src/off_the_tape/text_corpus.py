"""Read a Track 2 corpus using release timestamps from its frozen index."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class TextDocument:
    doc_id: str
    timestamp: date
    source: str
    doc_type: str
    file: str
    text: str
    sha256: str


@dataclass(frozen=True)
class TextCorpus:
    card_id: str
    asof: date
    documents: tuple[TextDocument, ...]

    @property
    def information_hash(self) -> str:
        """Hash the entire eligible information set, including metadata and bytes."""
        records = [
            (d.doc_id, d.timestamp.isoformat(), d.source, d.doc_type, d.file, d.sha256)
            for d in sorted(self.documents, key=lambda item: item.doc_id)
        ]
        encoded = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_corpus(unit_dir: Path, asof: date) -> TextCorpus:
    """Never read a document whose indexed public timestamp is after ``asof``."""
    root = Path(unit_dir)
    text_dir = (root if (root / "corpus_index.json").is_file() else root / "text").resolve()
    index = json.loads((text_dir / "corpus_index.json").read_text(encoding="utf-8"))
    frozen_asof = date.fromisoformat(index["asof"])
    if asof > frozen_asof:
        raise ValueError("requested as-of exceeds the frozen corpus index as-of")
    if not isinstance(index.get("documents"), list):
        raise ValueError("corpus index needs a documents array")
    documents: list[TextDocument] = []
    seen: set[str] = set()
    for entry in index["documents"]:
        doc_id = str(entry["doc_id"])
        if not doc_id or doc_id in seen:
            raise ValueError("corpus index has an empty or duplicate doc_id")
        seen.add(doc_id)
        published = date.fromisoformat(entry["timestamp"])
        if published > asof:
            continue
        relative = Path(entry["file"])
        path = (text_dir / relative).resolve()
        if relative.is_absolute() or text_dir not in path.parents or path.suffix.lower() != ".txt":
            raise ValueError("corpus index file must be a .txt below text/")
        body = path.read_bytes()
        documents.append(TextDocument(
            doc_id=doc_id,
            timestamp=published,
            source=str(entry["source"]),
            doc_type=str(entry["doc_type"]),
            file=relative.as_posix(),
            text=body.decode("utf-8", errors="replace"),
            sha256=hashlib.sha256(body).hexdigest(),
        ))
    return TextCorpus(str(index["card_id"]), asof, tuple(documents))
