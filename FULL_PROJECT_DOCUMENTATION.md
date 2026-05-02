# Full Project Report

## Abstract
This report provides a complete project overview of DocSearch, including objectives, system design, implementation approach, evaluation, and usage. The system is built for offline multilingual search over local documents.

## Objectives
- Provide reliable offline document search.
- Support English, Hindi, and Telugu.
- Combine classic IR with semantic search for better recall.
- Offer a desktop UI alongside CLI tooling.

## System design
DocSearch is organized into ingestion, indexing, search, UI, and evaluation layers. The ingestion layer extracts text from files using parsers and OCR. The indexing layer builds sparse and dense indices. The search layer offers keyword, semantic, and hybrid retrieval with optional re-ranking.

## Implementation summary
- OCR: Tesseract for images and scanned PDFs.
- PDF rendering: PDFium wrapper.
- Keyword search: TF-IDF and inverted index.
- Semantic search: multilingual embeddings + FAISS.
- Hybrid ranking: fusion of keyword and semantic scores.
- Re-ranking: cross-encoder on top candidates.

## User workflow
1) Select a folder in the UI or run CLI indexing.
2) The system extracts text and builds indices.
3) Use keyword, semantic, or hybrid search.
4) Evaluate with the built-in query set.

## How to run
Indexing:
```powershell
python -m src.main --index IR_DOCUMNETS
```

Search:
```powershell
python -m src.main --search "query" --mode hybrid
```

Evaluation:
```powershell
python -m src.main --evaluate
```

UI:
```powershell
python -m src.main
```

## Evaluation summary
Evaluation uses Precision, Recall, F1, MRR, and MAP. Scores are primarily driven by OCR text coverage and filename enrichment. Hybrid search with reranking yields the best overall balance of precision and recall.

## Constraints and trade-offs
The system favors offline stability and lower compute. Larger OCR and embedding models could improve accuracy but would increase cost and complexity.

## Conclusion
DocSearch provides a practical offline retrieval system by combining OCR, classic IR indexing, semantic search, and hybrid ranking. It is designed to be deployable on a standard desktop without cloud dependencies.
