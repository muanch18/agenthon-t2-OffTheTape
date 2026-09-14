"""Deterministic timestamp, source, and lexical ranking for Track 2 text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .text_corpus import TextCorpus, TextDocument

SELECTOR_VERSION = "1"
CORE_TERMS = (
    "policy", "rate", "inflation", "price", "growth", "employment",
    "labor", "labour", "stress", "uncertainty", "shock", "risk",
)
PANEL_TERMS = {
    "rates_daily": ("fomc", "fed", "treasury", "yield", "tightening", "easing"),
    "g10_fx_daily": ("currency", "dollar", "exchange", "fx", "carry", "yen", "ecb", "boj"),
    "factors_daily": ("equity", "factor", "market", "risk", "momentum", "credit"),
    "macro_quarterly": ("cpi", "pce", "nfp", "gdp", "release", "survey"),
}
FAMILY_TERMS = {
    "T2-F1": ("statement", "surprise", "shift"),
    "T2-F2": ("regime", "crisis", "transmission"),
    "T2-F3": ("cross-asset", "spillover", "correlation"),
    "T2-F4": ("tail", "shock", "stress", "crowded", "positioning"),
}
TYPE_WEIGHT = {
    "fomc_statement": 5.0,
    "fomc_minutes": 4.0,
    "macro_release": 4.0,
    "positioning_report": 4.0,
    "cb_speech": 3.0,
    "landmark": 3.0,
    "beige_book": 2.0,
    "corporate_8k": 2.0,
}


@dataclass(frozen=True)
class SelectedDocument:
    doc_id: str
    timestamp: date
    source: str
    doc_type: str
    sha256: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class DocumentSelector:
    max_documents: int = 6
    max_total_chars: int = 18000
    max_doc_chars: int = 4000

    def __post_init__(self) -> None:
        if min(self.max_documents, self.max_total_chars, self.max_doc_chars) < 1:
            raise ValueError("selection budgets must be positive")

    def select(
        self, corpus: TextCorpus, *, family: str | None = None, panel_id: str | None = None
    ) -> tuple[SelectedDocument, ...]:
        terms = CORE_TERMS + PANEL_TERMS.get(panel_id or "", ()) + FAMILY_TERMS.get(family or "", ())
        ranked = []
        for document in corpus.documents:
            if document.timestamp > corpus.asof:
                continue
            age = (corpus.asof - document.timestamp).days
            lexical = min(10, sum(min(3, document.text.casefold().count(term)) for term in terms))
            source_bonus = 1.0 if any(
                term in document.source.casefold() for term in ("federal reserve", "ecb", "boj", "bis")
            ) else 0.0
            score = TYPE_WEIGHT.get(document.doc_type, 1.0) + max(0.0, 2.0 - age / 45) + lexical / 5 + source_bonus
            ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], -item[1].timestamp.toordinal(), item[1].doc_id))

        chosen: list[SelectedDocument] = []
        remaining = self.max_total_chars
        for score, document in ranked[: self.max_documents]:
            budget = min(self.max_doc_chars, remaining)
            if budget <= 0:
                break
            excerpt = _excerpt(document, terms, budget)
            chosen.append(SelectedDocument(
                document.doc_id, document.timestamp, document.source, document.doc_type,
                document.sha256, excerpt, score,
            ))
            remaining -= len(excerpt)
        return tuple(chosen)


def _excerpt(document: TextDocument, terms: tuple[str, ...], budget: int) -> str:
    """Keep high-relevance paragraphs in document order, not just website headers."""
    chunks = []
    for paragraph in re.split(r"\n\s*\n", document.text):
        paragraph = " ".join(paragraph.split())
        for start in range(0, len(paragraph), 1000):
            chunk = paragraph[start : start + 1000]
            if chunk:
                chunks.append(chunk)
    if not chunks:
        return ""
    ranked = sorted(
        range(len(chunks)),
        key=lambda i: (-sum(min(3, chunks[i].casefold().count(term)) for term in terms), i),
    )
    selected = []
    used = 0
    for index in ranked:
        chunk = chunks[index]
        if used + len(chunk) + 2 > budget:
            continue
        selected.append(index)
        used += len(chunk) + 2
    if not selected:
        return chunks[0][:budget]
    return "\n\n".join(chunks[i] for i in sorted(selected))
