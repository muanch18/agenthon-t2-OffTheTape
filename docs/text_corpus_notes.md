# Track 2 frozen text corpus: observed format

Inspected the public `Agenthon-2026/track2-forecasting-public` checkout on
2026-09-14, including F1 `liftoff-telegraph-2015`, F2 `credit-crunch-2007`,
F3 `election-2024-joint`, F4 `jpy-crowding-2024`, plus the F1 2024 labor and
2023 CPI units. This describes shipped practice material, not the sealed set.

- Every one of the 104 unit directories has `text/corpus_index.json`. It holds
  `card_id`, `asof`, a note, and a `documents` array. Each entry has `doc_id`,
  `timestamp` (YYYY-MM-DD), `source`, `doc_type`, and relative `file`; some have
  a `note`. `doc_id` is often an event ID rather than a publication date.
  FOMC minutes and COT reports use **public release** timestamps in the index,
  later than their meeting or report dates. Selection must use `timestamp`.
- The 627 indexed documents are all `.txt` files. Per-unit count is 2–15
  (median 5); total indexed text is roughly 280 KB per unit at the median and
  reaches about 641 KB. All indexed filenames resolved in the inspected tree.
  Long speeches, Beige Books, minutes, and BLS releases therefore need excerpt
  selection and a character budget before model calls.
- Type counts: 227 `cb_speech`, 155 `fomc_statement`, 97 `beige_book`, 61
  `fomc_minutes`, 53 `macro_release`, 21 `landmark`, 7 `corporate_8k`, and 6
  `positioning_report`. The corpus index has no distinct news-headline type
  or headline files in this snapshot, despite headlines being mentioned in
  participant guidance. A later corpus may still contain headline-like text.
- FOMC statements are short plain text with a date/header followed by policy
  paragraphs and votes. Speeches can be long text or Markdown-like transcripts
  with headings. BLS CPI/employment releases include substantial website
  navigation boilerplate before tables and prose. COT reports can be tabular
  plain text with weekly positions and a separate release date. BOJ landmarks
  are OCR-like text with irregular whitespace. These are unstructured bodies;
  the index is the reliable source for timestamp and type.
- Representative sizes: F1 liftoff has 12 docs / 452 KB; F2 credit crunch
  7 / 151 KB; F3 election 8 / 520 KB; F4 JPY crowding 5 / 262 KB. The loader
  rejects path escapes and duplicate IDs, skips documents published after the
  requested as-of date, and refuses a requested as-of later than the unit's
  frozen index as-of. Historical comparisons within one unit have incomplete
  earlier information sets: that unit includes only its curated subset, not a
  comprehensive contemporaneous archive.
