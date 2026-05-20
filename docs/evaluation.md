# Evaluation

DocSearch evaluates keyword, semantic, and hybrid retrieval using standard IR metrics. The evaluation set lives in [src/evaluation/test_queries.json](src/evaluation/test_queries.json) and maps each query to a list of relevant filenames.

## Metrics
- Precision@K: how many of the top K results are relevant
- Recall@K: how many relevant documents appear in the top K
- F1@K: balance between precision and recall
- MRR: rewards early appearance of the first relevant result
- MAP: rewards ranking multiple relevant docs early

### Definitions
$$
P@K = \frac{|relevant \cap retrieved_{1..K}|}{K}
$$

$$
R@K = \frac{|relevant \cap retrieved_{1..K}|}{|relevant|}
$$

$$
F1@K = \frac{2 \cdot P@K \cdot R@K}{P@K + R@K}
$$

$$
MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{rank_i}
$$

$$
AP = \frac{1}{|relevant|} \sum_{k=1}^{K} P@k \cdot rel(k)
$$

$$
MAP = \frac{1}{N} \sum_{i=1}^{N} AP_i
$$

## Experiment setup
- Retrieval modes: keyword (TF-IDF), semantic (FAISS), hybrid + rerank
- Evaluation uses normalized filenames to avoid formatting mismatch
- K values: 5 and 10

Implementation: [src/evaluation/metrics.py](src/evaluation/metrics.py)

## Results (sample run)
| Mode | MRR | MAP | P@5 | R@5 | F1@5 |
| --- | --- | --- | --- | --- | --- |
| Hybrid (no rerank) | 0.7204 | 0.6398 | 0.2400 | 0.7739 | 0.3472 |
| Hybrid + rerank | 0.7556 | 0.7052 | 0.2400 | 0.7961 | 0.3499 |

## Interpretation
- Hybrid + rerank improves ranking quality (MRR, MAP) without sacrificing top-5 precision.
- Recall improves slightly, indicating better coverage at top ranks.
- When scores are low, the usual cause is missing text in OCR output or insufficient term coverage in the index.

## Precision vs recall tradeoff
- Keyword search tends to have higher precision when exact terms are present.
- Semantic search tends to increase recall on paraphrases and multilingual queries.
- Hybrid fusion balances both, while reranking prioritizes the most likely correct results at the top.
