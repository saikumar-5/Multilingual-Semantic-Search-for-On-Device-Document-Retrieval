# Full Project Report

## Abstract
This report explains what DocSearch does, how it works, why its design is effective, and what the evaluation means for your final project.

The system is an offline, multilingual search engine for local documents. It combines OCR, classic keyword retrieval, semantic embedding search, and hybrid fusion to handle scanned documents and mixed-language collections.

## Project Purpose
- Deliver reliable local document retrieval without cloud services.
- Support English, Hindi, and Telugu content.
- Handle native text files plus scanned PDFs and images.
- Provide both CLI and desktop UI search workflows.

## What happens in this project
### 1) File ingestion
- The system scans a directory recursively and identifies supported files: PDF, DOCX, text, Excel, and images.
- Each file is routed to the appropriate parser.
- For scanned documents and images, the system uses Tesseract OCR.
- For PDFs, PDFium renders pages to images and OCR extracts the text.

### 2) Text enrichment
- After extraction, the system optionally injects normalized file-name terms into the document text.
- This filename injection is a practical improvement to recover queries like "id card" or "electricity bill" when the exact term is missing in extracted text.
- Tag mapping expands related terms, so "id" also brings in "identity" and "card".

### 3) Sparse indexing (classic IR)
- Every parsed document is added to an inverted index.
- TF-IDF weights are computed from this index for keyword ranking.
- This provides exact term matching and strong precision for queries that match extracted words.

### 4) Dense semantic indexing
- Documents are split into chunks and encoded with a multilingual sentence-transformer model.
- The model is a compact 384-dimensional embedding model optimized for English, Hindi, and Telugu.
- Chunk vectors are stored in a FAISS index using HNSW for fast approximate nearest neighbor search.

### 5) Search modes
- Keyword search uses the inverted index + TF-IDF and is best when the query term appears directly in the text.
- Semantic search uses dense vector similarity and is best for meaning-based matching, synonyms, and cross-language retrieval.
- Hybrid search combines both keyword and semantic scores with a weighted fusion.
- When enabled, the top hybrid candidates are re-ranked by a cross-encoder model for better top-result precision.

### 6) Evaluation
- The system evaluates three modes: keyword, semantic, and hybrid + cross-encoder.
- It uses a fixed query set (`src/evaluation/test_queries.json`).
- Evaluation metrics are Precision@K, Recall@K, F1@K, MRR, and MAP.
- The evaluation code loads the built indices, runs each query, and prints the report.

## Why this method is better for your final project
### Offline and stable on Windows
- The architecture avoids cloud APIs entirely.
- OCR uses Tesseract + PDFium, which is much more reliable than heavyweight stacks like PaddleOCR on Windows.
- All models can load from local directories when `OFFLINE_MODE` is enabled.

### Multimodal retrieval strength
- Combining OCR and parsing ensures both typed text and scanned text are searchable.
- Filename enrichment recovers documents when the extracted text alone is insufficient.
- Hybrid fusion reduces false negatives by letting one method compensate for the other.

### Efficient accuracy trade-offs
- The embedding model is intentionally small enough for CPU inference.
- FAISS HNSW settings favor moderate recall with good query speed.
- Cross-encoder reranking is applied only to the top 30 candidates, which improves final precision without re-scoring the full corpus.

### Practical robustness
- The system is designed for real desktop use, not just research.
- It supports common document formats and multilingual scanned data.
- It saves indices locally, so repeated queries are fast after the one-time indexing cost.

## Compute and latency behavior
### CPU usage
- The system caps CPU threads with `CPU_THREADS = max(1, min(4, os.cpu_count() or 1))`.
- Both the embedding model and reranker use the same CPU thread limit.
- This keeps the system responsive on a typical laptop or desktop.

### Indexing cost
- Indexing is the heaviest step because it includes OCR, text extraction, chunking, embedding, and FAISS index construction.
- OCR and embedding are the main compute drivers.
- This is a one-time cost per document collection, and the results are persisted to disk.

### Query latency
- Keyword-only search is fast and usually returns results in a fraction of a second.
- Semantic search is slower because it encodes the query on CPU, but the small multilingual model keeps it practical.
- Hybrid search adds fusion overhead and, if reranking is enabled, a second pass over the top 30 candidates.
- In practice, query latency is expected to remain in the interactive range on a desktop CPU: keyword < semantic < hybrid + rerank.

### FAISS tuning
- `FAISS_HNSW_M = 32` gives good retrieval quality without excessive graph cost.
- `FAISS_HNSW_EF_CONSTRUCTION = 200` balances build cost and recall.
- `FAISS_HNSW_EF_SEARCH = 64` keeps query time fast while preserving search quality.

## What the evaluation tells you
- The hybrid approach is the strongest overall because it uses both exact text matching and meaning-based retrieval.
- OCR and filename injection are the biggest sources of improved recall for scanned, noisy, or badly formatted files.
- Cross-encoder reranking improves top-ranked results, which is why it is used only after the initial hybrid candidate list.
- Low scores in any mode usually signal missing text coverage: if the relevant term is not extracted, no retrieval technique can fix it.

## Observed reranking improvement
- **Hybrid without rerank:** MRR 0.7204, MAP 0.6398, P@5 0.2400, R@5 0.7739, F1@5 0.3472.
- **Hybrid with rerank:** MRR 0.7556, MAP 0.7052, P@5 0.2400, R@5 0.7961, F1@5 0.3499.
- This means reranking improved overall rank quality noticeably: MRR increased by +0.0352 and MAP improved by +0.0654.
- The top-5 recall also rose by +0.0222, showing reranking pushed more relevant documents into the higher ranks.
- Precision at 5 stayed the same, which indicates the reranker preserved top precision while improving rank ordering.

## How to run the final evaluation
- Index your document folder once:
  ```powershell
  python -m src.main --index IR_DOCUMNETS
  ```
- Run evaluation:
  ```powershell
  python -m src.main --evaluate
  ```
- If you need interactive use, launch the UI:
  ```powershell
  python -m src.main
  ```

## Final takeaway
DocSearch is a practical local search system that is better for this project because it:
- works offline,
- supports multilingual scanned documents,
- combines keyword and semantic retrieval,
- and adds targeted improvements like filename-based term enrichment.

The final evaluation should compare keyword, semantic, and hybrid results and show that hybrid + rerank gives the best balance of precision and recall for your test queries.
