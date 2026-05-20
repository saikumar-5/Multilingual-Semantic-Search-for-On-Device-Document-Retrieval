# Technical Report

## Abstract
This report documents the technical design and implementation of DocSearch, a local-first, multilingual information retrieval system. The system ingests local documents, extracts text using OCR where needed, builds multiple indices, and provides keyword, semantic, and hybrid search through a desktop UI and CLI.

## Objectives
- Support multilingual search for English, Hindi, and Telugu.
- Operate fully offline on local machines.
- Balance accuracy with reasonable CPU and memory usage.
- Provide both automated evaluation and user-facing UI search.

## System architecture
DocSearch is organized into five layers:
1) Ingestion: routing and parsing of input documents.
2) Indexing: inverted index, TF-IDF, and FAISS vector index.
3) Search: keyword, semantic, and hybrid ranking with optional re-ranking.
4) UI: desktop interface for indexing and search.
5) Evaluation: offline metrics using ground-truth queries.

## Core technical choices
### OCR and PDF handling
- OCR engine: Tesseract via Python bindings.
- PDF rendering: PDFium via a Python wrapper.
- Rationale: stable Windows support, offline operation, and low resource use.

### Retrieval models
- Keyword search uses TF-IDF for precise term matching.
- Semantic search uses a multilingual embedding model for meaning-based retrieval.
- Hybrid search fuses keyword and semantic scores to reduce misses.

### Re-ranking
- A cross-encoder re-ranks top candidates only.
- Rationale: improve precision without re-ranking the full corpus.

## Data storage
Index artifacts are persisted locally to avoid repeated computation:
- Inverted index, TF-IDF weights, vector index, and document store.

## Execution modes
DocSearch runs in two modes:
- CLI: for indexing, search, and evaluation.
- UI: for interactive search and indexing.

## Limitations
- OCR accuracy depends on scan quality and fonts.
- Semantics are limited by the size of the embedding model.
- Evaluation quality depends on the coverage of the ground-truth set.

## Conclusion
DocSearch delivers a practical offline search system by combining classic IR techniques with lightweight semantic retrieval and OCR. Its architecture is optimized for local use and balanced accuracy.
