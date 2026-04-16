"""
FAISS Vector Index with sentence-transformers embeddings.

CO2/CO3 Alignment: Vector Space Model - representing documents and queries
as dense vectors and using cosine similarity for retrieval.

Uses paraphrase-multilingual-MiniLM-L12-v2 (~120MB) which supports
50+ languages including English, Hindi, and Telugu in a shared
vector space, enabling cross-language retrieval.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, MAX_CHUNK_TOKENS

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Dense vector index using FAISS for efficient similarity search.

    Documents are:
    1. Split into chunks (max 256 tokens each)
    2. Encoded into 384-dimensional vectors using MiniLM
    3. Stored in a FAISS index for fast nearest-neighbor search

    Queries are encoded the same way and matched via cosine similarity.
    """

    def __init__(self):
        self.model = None
        self.index = None
        # Maps FAISS internal index → (doc_id, chunk_id)
        self.index_to_doc: List[Tuple[int, int]] = []
        # All document vectors (for clustering etc.)
        self.vectors: Optional[np.ndarray] = None
        # Unique doc-level vectors (mean of all chunk vectors per doc)
        self.doc_vectors: dict = {}
        self._model_loaded = False

    def load_model(self):
        """Load the sentence-transformer model into memory."""
        if self._model_loaded:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._model_loaded = True
        logger.info("Embedding model loaded successfully")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for embedding.

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
            batch = all_chunks[i : i + batch_size]
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
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.index.add(self.vectors)

        # Compute document-level vectors (mean of all chunk vectors per doc)
        self._compute_doc_vectors()

        logger.info(
            f"Built FAISS index: {self.index.ntotal} vectors, "
            f"{EMBEDDING_DIMENSION} dimensions"
        )

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

        # Encode query
        query_vector = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Search FAISS index
        # Request more results than top_k since multiple chunks may be from same doc
        n_search = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_vector, n_search)

        # Aggregate scores per document (take max chunk score per doc)
        doc_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id, _ = self.index_to_doc[idx]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = float(score)

        # Sort by score descending
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into a vector."""
        self.load_model()
        return self.model.encode(
            [query], normalize_embeddings=True
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

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
