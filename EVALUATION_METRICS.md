# Evaluation Report

## Abstract
This report explains the evaluation methodology used for DocSearch and why the reported metric values behave as they do. It focuses on information retrieval quality and coverage.

## Evaluation setup
- Query set: fixed list of 30 queries with ground-truth relevant documents.
- Retrieval modes: keyword, semantic, and hybrid with reranking.
- Matching method: normalized filenames to avoid formatting mismatches.

## Metrics and formulas
### Precision@K
$$
P@K = \frac{|\text{relevant} \cap \text{retrieved}_{1..K}|}{K}
$$

### Recall@K
$$
R@K = \frac{|\text{relevant} \cap \text{retrieved}_{1..K}|}{|\text{relevant}|}
$$

### F1@K
$$
F1@K = \frac{2 \cdot P@K \cdot R@K}{P@K + R@K}
$$

### MRR
$$
MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
$$

### MAP
$$
AP = \frac{1}{|\text{relevant}|} \sum_{k=1}^{K} P@k \cdot rel(k)
$$
$$
MAP = \frac{1}{N} \sum_{i=1}^{N} AP_i
$$

## Why these metrics
- Precision focuses on quality of top results.
- Recall measures coverage of all relevant documents.
- F1 balances precision and recall.
- MRR captures how quickly the first relevant result appears.
- MAP rewards ranking multiple relevant documents early.

## Why values increase or decrease
Evaluation scores are driven by text coverage and ranking quality:
- OCR and filename enrichment improve recall and MAP because more relevant terms exist in the index.
- Re-ranking improves precision and MRR by moving the most relevant results to the top.
- Missing terms in extracted text cause zero-hit queries regardless of ranking.

## How to run evaluation
```powershell
python -m src.main --evaluate
```

## Limitations
- If a relevant term is missing in extracted text, evaluation will show low recall.
- OCR quality directly affects all metrics for scanned documents.
