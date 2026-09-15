Goal:
Macro-regime-conditioned joint probabilistic forecaster for Agenthon T2.

Architecture:
panel → numeric joint forecast → 1000 scenarios
text → LLM macro-state extraction
macro state → mean / vol / skew / correlation / shock adjustments
→ forecast.parquet

Priorities:
1. historical pseudo-card backtester
2. PCA/Student-t or bootstrap numeric baseline
3. macro-state extractor
4. distribution transformer
5. F1-F4 conditional logic
6. Docker + smoke gates