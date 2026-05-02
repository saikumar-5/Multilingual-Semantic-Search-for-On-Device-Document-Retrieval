# DocSearch Project Documentation (Overview)

## What this project does
DocSearch is a multilingual information retrieval system that indexes local documents and provides keyword, semantic, and hybrid search. It supports English, Hindi, and Telugu content, handles PDFs/DOCX/Excel/text/images, and exposes both a CLI and a CustomTkinter desktop UI. The system is designed for offline, local-first use, with careful trade-offs between accuracy and compute cost.

## High-level architecture
- Ingestion layer: parses files and extracts text (including OCR for images/scanned PDFs).
- Indexing layer: builds classic IR structures (inverted index, TF-IDF) and a semantic vector index (FAISS + sentence-transformers).
- Search layer: supports boolean, phrase, wildcard, keyword (TF-IDF), semantic (vector), and hybrid search.
- UI layer: desktop app with live search, settings, and indexing controls.
- Evaluation layer: computes IR metrics (Precision@K, Recall@K, MRR, MAP).

Key entry points:
- CLI/GUI launcher: [src/main.py](src/main.py)
- Configuration: [src/config.py](src/config.py)
- GUI application: [src/ui/app.py](src/ui/app.py)

## Data flow (end-to-end)
1) User selects folders in the UI or runs CLI indexing.
2) Ingestion scans supported files and extracts text.
3) Indexing builds:
   - Inverted index with positional postings
   - TF-IDF weights for ranking
   - Vector index with multilingual embeddings
4) Index artifacts are persisted to data/.
5) Search uses the chosen mode (keyword, semantic, hybrid) and formats results with snippets and language tags.

## Detailed technical flow (step by step)
### 1) File discovery and routing
The ingestion pipeline walks all subfolders under the selected directory and routes files by extension. The file type mapping is in [src/config.py](src/config.py). The routing logic is in [src/ingestion/file_router.py](src/ingestion/file_router.py).

### 2) Parsing and OCR extraction
The system extracts text from each supported file type using specialized parsers:
- PDFs: rendered with PDFium and OCR fallback on low-text pages.
- Images: OCR using Tesseract.
- DOCX/XLSX/text: parsed with format-aware libraries.

Files with no text still produce a document record, but are flagged with warnings in logs.

### 3) Filename and keyword enrichment
To close coverage gaps, the ingestion layer injects normalized filename tokens and mapped tags into the document text. This is a targeted recall boost for queries like "id card" or "house permission" that may appear in filenames but not in OCR output. This is implemented in [src/ingestion/file_router.py](src/ingestion/file_router.py).

### 4) Preprocessing
Tokenization, stopword handling, and script-aware normalization are applied for English, Hindi, and Telugu. The output is positional tokens used for phrase and wildcard search. This is implemented in [src/indexer/preprocessor.py](src/indexer/preprocessor.py).

### 5) Indexing
The system builds three indices:
- Inverted index (positional) for boolean and phrase queries.
- TF-IDF index for keyword ranking.
- FAISS vector index for semantic search.

All index artifacts are stored in data/ to allow reuse without re-indexing.

### 6) Search and ranking
Search mode controls how results are combined:
- Keyword mode: TF-IDF scoring.
- Semantic mode: embedding similarity.
- Hybrid mode: fusion of keyword and semantic scores.
An optional cross-encoder re-ranker refines the top candidates in hybrid mode for higher precision.

### 7) Evaluation
Evaluation uses a fixed query set with ground-truth file names to compute metrics. A filename normalization step makes matching resilient to formatting differences. The evaluation code is in [src/evaluation/metrics.py](src/evaluation/metrics.py), and the ground truth is in [src/evaluation/test_queries.json](src/evaluation/test_queries.json).

## Ingestion and document parsing
Routing and parsers:
- File routing and directory scan: [src/ingestion/file_router.py](src/ingestion/file_router.py)
- PDF parsing (text + OCR fallback): [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py)
- DOCX parsing: [src/ingestion/docx_parser.py](src/ingestion/docx_parser.py)
- Excel parsing: [src/ingestion/excel_parser.py](src/ingestion/excel_parser.py)
- Plain text parsing (encoding detection): [src/ingestion/text_parser.py](src/ingestion/text_parser.py)
- OCR for images and scanned PDFs: [src/ingestion/image_ocr.py](src/ingestion/image_ocr.py)

### OCR stack (current)
- Engine: Tesseract (local installation + pytesseract bindings).
- PDF rendering: PDFium via pypdfium2.
- Languages: English, Hindi, Telugu (mapped to eng, hin, tel).

This stack is chosen for stability and offline use on Windows, avoiding large model downloads and build-time native dependencies.

Supported file types and OCR settings are centralized in [src/config.py](src/config.py).

## Text preprocessing
- Tokenization supports English, Hindi (Devanagari), and Telugu scripts.
- Stopword filtering (for all three languages), length filtering, and simple language detection.
- Positional token output for phrase search.

Implementation: [src/indexer/preprocessor.py](src/indexer/preprocessor.py)

## Indexing methods
### 1) Inverted index (positional)
- Core term -> {doc_id: [positions]} structure.
- Enables boolean and phrase queries.
- Saved to data/inverted_index.pkl.

Implementation: [src/indexer/inverted_index.py](src/indexer/inverted_index.py)

### 2) TF-IDF weighting (vector space model)
- Computes IDF and per-document TF-IDF weights.
- Uses cosine similarity for ranking.
- Saved to data/tfidf_index.pkl.

Implementation: [src/indexer/tfidf.py](src/indexer/tfidf.py)

### 3) Vector index (semantic search)
- Uses sentence-transformers model: intfloat/multilingual-e5-small.
- Documents are chunked (approx 180 tokens) and encoded.
- FAISS IndexFlatIP used for cosine similarity search.
- Saves FAISS index and metadata.

Implementation: [src/indexer/vector_index.py](src/indexer/vector_index.py)

### 4) Incidence matrix (IR concept demo)
- Binary term-document matrix for illustrating Boolean model operations.

Implementation: [src/indexer/incidence_matrix.py](src/indexer/incidence_matrix.py)

### 5) Document clustering (optional)
- K-Means over document vectors for grouping similar documents.

Implementation: [src/indexer/clustering.py](src/indexer/clustering.py)

## Search methods
### Query processing
- Detects boolean, phrase, wildcard, and free-text queries.
- Provides cleaned query text for semantic embedding.

Implementation: [src/search/query_processor.py](src/search/query_processor.py)

### Keyword search (TF-IDF)
- Boolean/phrase queries use the inverted index, then TF-IDF for ranking.
- Free-text queries use TF-IDF cosine similarity.

Implementation: [src/search/keyword_search.py](src/search/keyword_search.py)

### Semantic search (vector)
- Encodes query in the same embedding space as documents.
- Uses FAISS for nearest-neighbor search.

Implementation: [src/search/semantic_search.py](src/search/semantic_search.py)

### Hybrid search (score fusion)
- Linear fusion of normalized keyword and semantic scores.
- Default alpha is 0.6 (more weight on semantic similarity).
- Optional cross-encoder reranking adds a precision-focused pass on top candidates.

Implementation: [src/search/hybrid_search.py](src/search/hybrid_search.py)

### Wildcard search
- Bigram index narrows candidate terms, then TF-IDF scoring.

Implementation: [src/search/wildcard.py](src/search/wildcard.py)

### Result formatting
- Enriches results with snippets, language tags, metadata.

Implementation: [src/search/ranker.py](src/search/ranker.py)

## UI (desktop app)
The UI is a CustomTkinter desktop app with a two-panel layout:
- Left panel: live search bar, status, and results list.
- Right panel: indexed folders, search mode selection, and stats.
- First-run dialog prompts folder selection.
- Indexing runs in background thread with progress updates.

UI components:
- Main app and indexing orchestration: [src/ui/app.py](src/ui/app.py)
- Search panel: [src/ui/search_frame.py](src/ui/search_frame.py)
- Settings panel: [src/ui/settings_frame.py](src/ui/settings_frame.py)
- Folder selection dialog: [src/ui/directory_dialog.py](src/ui/directory_dialog.py)

## Evaluation
Standard IR metrics are implemented for offline testing using a JSON file of queries and relevant docs:
- Precision@K, Recall@K, F1@K
- MRR, MAP

Implementation: [src/evaluation/metrics.py](src/evaluation/metrics.py)
Test queries: [src/evaluation/test_queries.json](src/evaluation/test_queries.json)

### Why these metrics and what they mean
- Precision@K: In the top K results, how many are relevant. It reflects result quality.
- Recall@K: Of all relevant docs, how many are found in the top K. It reflects coverage.
- F1@K: Balances precision and recall when you want both.
- MRR: Measures how quickly the first relevant result appears.
- MAP: Rewards methods that rank multiple relevant docs early, not just the first.

These metrics are standard for IR systems because they capture both user satisfaction (top results are correct) and coverage (relevant items are not missed).

### Why the values look the way they do
The evaluation scores are tightly correlated with text coverage in the index. If a term like "id card" or "house permission" never appears in extracted text, no retrieval model can surface it. Improvements in OCR and filename enrichment directly increase recall and MAP, while cross-encoder re-ranking primarily increases precision and MRR.

## Configuration and storage
Key settings include:
- Embedding model name and dimension
- OCR configuration (Tesseract path and languages)
- Supported file extensions
- Default top-k and hybrid alpha

Implementation: [src/config.py](src/config.py)

Artifacts stored in data/:
- inverted_index.pkl
- tfidf_index.pkl
- vectors.faiss
- vector_meta.pkl
- doc_store.pkl
- settings.json

## CLI usage (from main)
- Index a directory: python -m src.main --index <folder>
- Search the index: python -m src.main --search "query" --mode keyword|semantic|hybrid
- Run evaluation: python -m src.main --evaluate

Implementation: [src/main.py](src/main.py)

## Accuracy vs compute trade-offs (local, offline)
This system is intentionally designed for offline, local use. The trade-offs are:
- OCR: Tesseract + PDFium is lighter and more stable on Windows than heavy OCR models. It avoids GPU requirements and large model downloads. The trade-off is lower accuracy on complex scans.
- Semantic search: multilingual-e5-small is chosen for fast CPU inference and multilingual coverage. Larger models can be more accurate, but are slower and heavier.
- Hybrid ranking: combines keyword and semantic search to reduce miss rates without needing expensive semantic-only re-ranking on the full corpus.
- Cross-encoder reranking: applied only on top candidates to improve precision while keeping compute reasonable.

This balance prioritizes responsiveness, portability, and offline reliability, while still delivering solid accuracy via hybrid ranking and targeted enrichments.

## Full project documentation (component map)
- Config and constants: [src/config.py](src/config.py)
- Ingestion and routing: [src/ingestion/file_router.py](src/ingestion/file_router.py)
- PDF parsing: [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py)
- Image OCR: [src/ingestion/image_ocr.py](src/ingestion/image_ocr.py)
- Indexing:
   - Inverted index: [src/indexer/inverted_index.py](src/indexer/inverted_index.py)
   - TF-IDF: [src/indexer/tfidf.py](src/indexer/tfidf.py)
   - FAISS vector index: [src/indexer/vector_index.py](src/indexer/vector_index.py)
- Search:
   - Keyword search: [src/search/keyword_search.py](src/search/keyword_search.py)
   - Semantic search: [src/search/semantic_search.py](src/search/semantic_search.py)
   - Hybrid search: [src/search/hybrid_search.py](src/search/hybrid_search.py)
   - Reranker: [src/search/cross_encoder_reranker.py](src/search/cross_encoder_reranker.py)
- Evaluation: [src/evaluation/metrics.py](src/evaluation/metrics.py)
- UI:
   - Main app: [src/ui/app.py](src/ui/app.py)
   - Search panel: [src/ui/search_frame.py](src/ui/search_frame.py)
   - Settings panel: [src/ui/settings_frame.py](src/ui/settings_frame.py)

## Summary of IR methods used
- Boolean model: positional inverted index + boolean operators
- Vector space model (sparse): TF-IDF + cosine similarity
- Vector space model (dense): sentence-transformer embeddings + FAISS
- Hybrid ranking: linear fusion of normalized scores
- N-gram (bigram) indexing for wildcard queries
- OCR-based text extraction for images and scanned PDFs
- Language-aware preprocessing (English, Hindi, Telugu)
