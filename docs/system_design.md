# System design

DocSearch is designed as a local-first retrieval system that runs fully offline. The system favors predictable latency on CPU while maintaining high recall for noisy, multilingual document collections.

## Design goals
- Offline-only processing and private storage
- Multilingual retrieval across English, Hindi, and Telugu
- Fast interactive query latency on CPU
- Robustness to OCR noise and mixed file formats

## Key decisions

### RRF fusion over linear score fusion
RRF uses rank positions rather than raw scores. It stays stable when score scales drift between TF-IDF and vector similarity and prevents one channel from dominating due to calibration issues.

### HNSW over Flat index
HNSW provides strong recall while keeping latency reasonable on CPU. The configuration is tuned via `FAISS_HNSW_M`, `FAISS_HNSW_EF_CONSTRUCTION`, and `FAISS_HNSW_EF_SEARCH` in [src/config.py](src/config.py).

### multilingual-e5-small embeddings
E5 embeddings support explicit query/document prefixes and provide stronger multilingual alignment than older MiniLM variants at a similar footprint. The embedding model is configured in [src/config.py](src/config.py).

### Rerank only top candidates
Cross-encoders are accurate but expensive. DocSearch reranks only the top candidates, improving top-10 relevance without re-scoring the entire corpus.

### Offline-first model loading
When offline mode is enabled, models are loaded only from local directories. This makes the system portable to air-gapped environments.

## OCR stack rationale
- PDFium provides reliable rendering for scanned PDFs on Windows.
- Tesseract offers stable local OCR with manageable dependencies.
- OCR noise is mitigated by filename injection and query expansion.

## Future product preparation

### Packaging and installer readiness
- Build spec: [build.spec](build.spec)
- Packaging notes: [packaging/README.md](packaging/README.md)
- Installer notes: [installer/README.md](installer/README.md)

### Local deployment
- Indices are persisted to [data/](data/) for fast reuse.
- Settings are stored locally per machine.

### UI integration
The UI layer is decoupled from indexing and search. Future UI changes (desktop or web) can reuse the same search API.
