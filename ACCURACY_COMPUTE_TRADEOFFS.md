# Accuracy vs Compute Report (Local, Offline)

## Abstract
This report explains how DocSearch balances accuracy and computational cost for offline use on local machines. It documents the specific choices that trade accuracy for stability and speed.

## Constraints
- No cloud dependencies.
- CPU-first execution.
- Multilingual coverage for English, Hindi, and Telugu.

## OCR trade-off
Choice: Tesseract + PDFium
- Benefit: lightweight and fully offline; works on standard Windows setups.
- Cost: lower accuracy on degraded scans and handwritten documents.

## Embedding model trade-off
Choice: a small multilingual embedding model
- Benefit: fast CPU inference and low memory use.
- Cost: lower semantic accuracy than larger models.

## Hybrid retrieval trade-off
Hybrid search combines keyword and semantic results:
- Benefit: reduces misses when either method fails.
- Cost: slightly higher compute compared to single-mode search.

## Re-ranking trade-off
Cross-encoder re-ranking runs only on top candidates:
- Benefit: improved precision at top results.
- Cost: added latency proportional to candidate count.

## Storage trade-off
All indices are stored locally:
- Benefit: fast repeated queries without re-indexing.
- Cost: higher disk usage.

## Summary
The system prioritizes offline reliability and acceptable latency. Accuracy gains come from hybrid fusion and targeted enrichment rather than heavy models.
