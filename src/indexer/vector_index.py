"""
FAISS Vector Index with sentence-transformers embeddings.

CO2/CO3 Alignment: Vector Space Model - representing documents and queries
as dense vectors and using cosine similarity for retrieval.

Uses multilingual-e5-small (~120MB), which supports multilingual
retrieval across English, Hindi, and Telugu in a shared vector space.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import re

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_LOCAL_DIR,
    EMBEDDING_DIMENSION,
    MAX_CHUNK_TOKENS,
    OFFLINE_MODE,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_DOCUMENT_PREFIX,
    CPU_THREADS,
    FAISS_INDEX_TYPE,
    FAISS_HNSW_M,
    FAISS_HNSW_EF_CONSTRUCTION,
    FAISS_HNSW_EF_SEARCH,
    CHUNKING_STRATEGY,
    SEMANTIC_CHUNK_SIMILARITY_THRESHOLD,
    SEMANTIC_CHUNK_MIN_TOKENS,
    SEMANTIC_CHUNK_OVERLAP_SENTENCES,
)

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Dense vector index using FAISS for efficient similarity search.

    Documents are:
    1. Split into chunks (max MAX_CHUNK_TOKENS words)
    2. Encoded into 384-dimensional vectors using multilingual-e5-small
    3. Stored in a FAISS index for fast nearest-neighbor search

    Queries are encoded the same way and matched via cosine similarity.
    """

    def __init__(self, device: str = "cpu"):
        self.model = None
        self.index = None
        # Maps FAISS internal index → (doc_id, chunk_id)
        self.index_to_doc: List[Tuple[int, int]] = []
        # All document vectors (for clustering etc.)
        self.vectors: Optional[np.ndarray] = None
        # Unique doc-level vectors (mean of all chunk vectors per doc)
        self.doc_vectors: dict = {}
        self._model_loaded = False
        self.device = device

    def load_model(self):
        """Load the sentence-transformer model into memory."""
        if self._model_loaded:
            return

        from sentence_transformers import SentenceTransformer
        import torch

        torch.set_num_threads(CPU_THREADS)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

        model_source = EMBEDDING_MODEL_NAME
        local_files_only = False

        if EMBEDDING_MODEL_LOCAL_DIR.exists():
            model_source = str(EMBEDDING_MODEL_LOCAL_DIR)
            local_files_only = True
        elif OFFLINE_MODE:
            raise FileNotFoundError(
                f"Offline mode enabled but embedding model not found at: {EMBEDDING_MODEL_LOCAL_DIR}"
            )

        logger.info(f"Loading embedding model from: {model_source}")
        self.model = SentenceTransformer(
            model_source,
            device=self.device,
            local_files_only=local_files_only,
        )
        self._model_loaded = True
        logger.info("Embedding model loaded successfully")

    def _embed_text(self, text: str, is_query: bool) -> str:
        """Format text for embedding model input.

        E5-family models are trained with explicit prefixes:
        - query: <text>
        - passage: <text>
        """
        prefix = EMBEDDING_QUERY_PREFIX if is_query else EMBEDDING_DOCUMENT_PREFIX
        clean = (text or "").strip()
        return f"{prefix}{clean}" if clean else clean

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using configured strategy."""
        strategy = (CHUNKING_STRATEGY or "fixed").strip().lower()
        if strategy == "semantic":
            return self._semantic_chunk_text(text)
        return self._fixed_chunk_text(text)

    def _fixed_chunk_text(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks suitable for embedding.

        Each chunk is approximately MAX_CHUNK_TOKENS words.
        We use word-based chunking with overlap for context continuity.
        """
        words = text.split()
        if len(words) <= MAX_CHUNK_TOKENS:
            return [text] if text.strip() else []

        chunks = []
        stride = MAX_CHUNK_TOKENS - 50  # 50-word overlap between chunks
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i : i + MAX_CHUNK_TOKENS])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentence-like units for semantic chunking."""
        if not text:
            return []

        blocks = [b.strip() for b in re.split(r"\n+", text) if b.strip()]
        sentences: List[str] = []

        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            for line in lines:
                if self._is_heading(line):
                    sentences.append(line)
                    continue

                parts = re.split(r"(?<=[.!?।॥])\s+", line)
                for part in parts:
                    clean = re.sub(r"\s+", " ", part).strip()
                    if clean:
                        sentences.append(clean)

        return sentences

    def _is_heading(self, line: str) -> bool:
        """Heuristic detection for headings to force chunk boundaries."""
        if not line:
            return False

        if len(line) <= 6:
            return False

        if line.endswith(":"):
            return True

        if line.isupper() and len(line.split()) <= 6:
            return True

        if re.match(r"^(section|chapter|unit)\s+\d+", line, re.IGNORECASE):
            return True

        return False

    def _split_long_text_by_words(self, text: str) -> List[str]:
        """Fallback splitter for very long sentence-like spans."""
        words = text.split()
        if len(words) <= MAX_CHUNK_TOKENS:
            return [text] if text.strip() else []

        stride = max(1, MAX_CHUNK_TOKENS - 50)
        chunks = []
        for i in range(0, len(words), stride):
            piece = " ".join(words[i : i + MAX_CHUNK_TOKENS]).strip()
            if piece:
                chunks.append(piece)
        return chunks

    def _semantic_chunk_text(self, text: str) -> List[str]:
        """
        Semantic chunking using sentence-transformer similarity between adjacent sentences.

        Boundary rule:
        - Split when adjacent sentence similarity drops below threshold, or
        - Split when chunk would exceed MAX_CHUNK_TOKENS.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1 or self.model is None:
            return self._fixed_chunk_text(text)

        sent_inputs = [self._embed_text(s, is_query=False) for s in sentences]
        sent_emb = self.model.encode(
            sent_inputs,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

        similarities = np.sum(sent_emb[:-1] * sent_emb[1:], axis=1)

        threshold = float(SEMANTIC_CHUNK_SIMILARITY_THRESHOLD)
        min_tokens = max(1, int(SEMANTIC_CHUNK_MIN_TOKENS))
        overlap_sentences = max(0, int(SEMANTIC_CHUNK_OVERLAP_SENTENCES))

        chunks: List[str] = []
        current_sentences: List[str] = [sentences[0]]
        current_tokens = len(sentences[0].split())

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = len(sentence.split())
            similarity = float(similarities[i - 1]) if (i - 1) < len(similarities) else 1.0

            if self._is_heading(sentence) and current_tokens >= min_tokens:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_tokens = 0

            if sentence_tokens > MAX_CHUNK_TOKENS:
                if current_sentences:
                    chunks.append(" ".join(current_sentences).strip())
                chunks.extend(self._split_long_text_by_words(sentence))
                current_sentences = []
                current_tokens = 0
                continue

            would_exceed_size = (current_tokens + sentence_tokens) > MAX_CHUNK_TOKENS
            topic_shift = similarity < threshold and current_tokens >= min_tokens

            if (would_exceed_size or topic_shift) and current_sentences:
                chunks.append(" ".join(current_sentences).strip())

                if overlap_sentences > 0:
                    current_sentences = current_sentences[-overlap_sentences:]
                else:
                    current_sentences = []

                current_tokens = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        return [c for c in chunks if c]

    def build(self, documents: List[dict], progress_callback=None):
        """
        Build the FAISS vector index from documents.

        Args:
            documents: List of dicts with 'doc_id' and 'text' keys.
            progress_callback: Optional callable(current, total, status).
        """
        import faiss

        self.load_model()

        all_chunks = []
        self.index_to_doc = []

        # Chunk all documents
        for doc in documents:
            chunks = self._chunk_text(doc["text"])
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                self.index_to_doc.append((doc["doc_id"], chunk_id))

        if not all_chunks:
            logger.warning("No text chunks to index")
            return

        logger.info(f"Encoding {len(all_chunks)} chunks from {len(documents)} documents")

        # Encode all chunks in batches
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(all_chunks), batch_size):
            batch = [
                self._embed_text(chunk, is_query=False)
                for chunk in all_chunks[i : i + batch_size]
            ]
            embeddings = self.model.encode(
                batch, show_progress_bar=False, normalize_embeddings=True
            )
            all_embeddings.append(embeddings)

            if progress_callback:
                progress_callback(
                    min(i + batch_size, len(all_chunks)),
                    len(all_chunks),
                    "Encoding documents...",
                )

        self.vectors = np.vstack(all_embeddings).astype("float32")

        # Build FAISS index (Inner Product = cosine similarity for normalized vectors)
        self.index = self._create_faiss_index()
        self.index.add(self.vectors)

        # Compute document-level vectors (mean of all chunk vectors per doc)
        self._compute_doc_vectors()

        logger.info(
            "Built FAISS %s index: %s vectors, %s dimensions",
            FAISS_INDEX_TYPE,
            self.index.ntotal,
            EMBEDDING_DIMENSION,
        )

    def _create_faiss_index(self):
        """Create FAISS index based on config."""
        import faiss

        index_type = (FAISS_INDEX_TYPE or "flat").strip().lower()
        if index_type == "hnsw":
            # HNSW with Inner Product for normalized vectors (cosine similarity).
            index = faiss.index_factory(
                EMBEDDING_DIMENSION,
                f"HNSW{FAISS_HNSW_M},Flat",
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = max(16, int(FAISS_HNSW_EF_CONSTRUCTION))
            index.hnsw.efSearch = max(8, int(FAISS_HNSW_EF_SEARCH))
            logger.info(
                "Using FAISS HNSW index (M=%s, efConstruction=%s, efSearch=%s)",
                FAISS_HNSW_M,
                index.hnsw.efConstruction,
                index.hnsw.efSearch,
            )
            return index

        if index_type != "flat":
            logger.warning("Unknown FAISS_INDEX_TYPE '%s'; falling back to flat", index_type)

        logger.info("Using FAISS Flat index (exact search)")
        return faiss.IndexFlatIP(EMBEDDING_DIMENSION)

    def _compute_doc_vectors(self):
        """Compute mean vector for each document from its chunk vectors."""
        from collections import defaultdict

        doc_chunk_indices = defaultdict(list)
        for idx, (doc_id, _) in enumerate(self.index_to_doc):
            doc_chunk_indices[doc_id].append(idx)

        self.doc_vectors = {}
        for doc_id, indices in doc_chunk_indices.items():
            chunk_vecs = self.vectors[indices]
            self.doc_vectors[doc_id] = chunk_vecs.mean(axis=0)

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for documents semantically similar to the query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, similarity_score) tuples.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        self.load_model()
        self._apply_runtime_search_params()

        # Encode query
        query_input = self._embed_text(query, is_query=True)
        query_vector = self.model.encode(
            [query_input],
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

        return self.search_by_vector(query_vector[0], top_k)

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search FAISS using a precomputed normalized query embedding."""
        if self.index is None or self.index.ntotal == 0:
            return []

        self._apply_runtime_search_params()

        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)

        # Request more results than top_k since multiple chunks may be from same doc
        n_search = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(q, n_search)

        # Aggregate scores per document (take max chunk score per doc)
        doc_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id, _ = self.index_to_doc[idx]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = float(score)

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into a vector."""
        self.load_model()
        query_input = self._embed_text(query, is_query=True)
        return self.model.encode(
            [query_input],
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")[0]

    def get_doc_vector(self, doc_id: int) -> Optional[np.ndarray]:
        """Get the mean vector for a document."""
        return self.doc_vectors.get(doc_id)

    def get_all_doc_vectors(self) -> Tuple[List[int], np.ndarray]:
        """Get all document-level vectors as a matrix."""
        doc_ids = sorted(self.doc_vectors.keys())
        if not doc_ids:
            return [], np.array([])
        vectors = np.array([self.doc_vectors[did] for did in doc_ids])
        return doc_ids, vectors

    def save(self, faiss_path: Path, metadata_path: Path):
        """Save the FAISS index and metadata to disk."""
        import faiss

        if self.index is not None:
            faiss.write_index(self.index, str(faiss_path))

        metadata = {
            "index_to_doc": self.index_to_doc,
            "doc_vectors": {
                k: v.tolist() for k, v in self.doc_vectors.items()
            },
            "faiss_index_type": FAISS_INDEX_TYPE,
            "hnsw": {
                "m": FAISS_HNSW_M,
                "ef_construction": FAISS_HNSW_EF_CONSTRUCTION,
                "ef_search": FAISS_HNSW_EF_SEARCH,
            },
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved vector index to {faiss_path}")

    def load(self, faiss_path: Path, metadata_path: Path):
        """Load the FAISS index and metadata from disk."""
        import faiss

        self.index = faiss.read_index(str(faiss_path))

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.index_to_doc = metadata["index_to_doc"]
        self.doc_vectors = {
            k: np.array(v, dtype="float32")
            for k, v in metadata["doc_vectors"].items()
        }

        self._apply_runtime_search_params()

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def _apply_runtime_search_params(self):
        """Apply runtime FAISS search params (mainly for HNSW)."""
        if self.index is None:
            return

        hnsw = getattr(self.index, "hnsw", None)
        if hnsw is not None:
            hnsw.efSearch = max(8, int(FAISS_HNSW_EF_SEARCH))
