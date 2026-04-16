# DocSearch Project Documentation

## What this project does
DocSearch is a multilingual information retrieval system that indexes local documents and provides keyword, semantic, and hybrid search. It supports English, Hindi, and Telugu content, handles PDFs/DOCX/Excel/text/images, and exposes both a CLI and a CustomTkinter desktop UI.

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

## Ingestion and document parsing
Routing and parsers:
- File routing and directory scan: [src/ingestion/file_router.py](src/ingestion/file_router.py)
- PDF parsing (text + OCR fallback): [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py)
- DOCX parsing: [src/ingestion/docx_parser.py](src/ingestion/docx_parser.py)
- Excel parsing: [src/ingestion/excel_parser.py](src/ingestion/excel_parser.py)
- Plain text parsing (encoding detection): [src/ingestion/text_parser.py](src/ingestion/text_parser.py)
- OCR for images and scanned PDFs: [src/ingestion/image_ocr.py](src/ingestion/image_ocr.py)

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
- Uses sentence-transformers model: paraphrase-multilingual-MiniLM-L12-v2.
- Documents are chunked (approx 256 tokens, with overlap) and encoded.
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

## Summary of IR methods used
- Boolean model: positional inverted index + boolean operators
- Vector space model (sparse): TF-IDF + cosine similarity
- Vector space model (dense): sentence-transformer embeddings + FAISS
- Hybrid ranking: linear fusion of normalized scores
- N-gram (bigram) indexing for wildcard queries
- OCR-based text extraction for images and scanned PDFs
- Language-aware preprocessing (English, Hindi, Telugu)
