"""
Semantic Search using dense vector embeddings.

CO2/CO3 Alignment: Vector Space Model with dense embeddings.
Unlike TF-IDF (sparse vectors), semantic search uses dense vectors
from a neural language model to capture meaning, not just keywords.

This enables:
- Synonym matching: "car" finds documents about "automobile"
- Cross-language: "house permission" finds Telugu "ఇంటి పట్టా"
- Conceptual: "machine learning" finds docs about "neural networks"
"""

from typing import List, Tuple
import logging
import numpy as np

from src.indexer.vector_index import VectorIndex
from src.search.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic document search using MiniLM multilingual embeddings + FAISS.
    """

    def __init__(self, vector_index: VectorIndex):
        self.vector_index = vector_index
        self.query_processor = QueryProcessor()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search documents by semantic similarity to the query.

        The query is encoded into the same vector space as the documents
        using the multilingual MiniLM model. FAISS finds the nearest
        document vectors using cosine similarity.

        Args:
            query: Natural language query in any supported language.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, similarity_score) tuples.
        """
        parsed = self.query_processor.parse(query)
        weighted_terms = parsed.get("weighted_terms", {})

        # If expansions exist, compose a weighted query embedding so original terms dominate.
        if parsed.get("expanded_terms") and weighted_terms:
            return self._search_with_weighted_terms(weighted_terms, top_k)

        # Clean query for embedding (remove Boolean operators etc.)
        clean_query = self.query_processor.get_raw_text_for_embedding(query)

        if not clean_query:
            return []

        results = self.vector_index.search(clean_query, top_k)
        return results

    def _search_with_weighted_terms(
        self,
        weighted_terms: dict,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        vectors = []
        weights = []

        for term, weight in weighted_terms.items():
            if not term or float(weight) <= 0:
                continue
            vectors.append(self.vector_index.encode_query(term))
            weights.append(float(weight))

        if not vectors:
            return []

        mat = np.vstack(vectors).astype("float32")
        w = np.asarray(weights, dtype="float32").reshape(-1, 1)
        q = (mat * w).sum(axis=0)

        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        return self.vector_index.search_by_vector(q, top_k)

    def search_with_language_info(
        self, query: str, top_k: int = 10
    ) -> List[dict]:
        """
        Search with additional language detection information.

        Returns:
            List of dicts with 'doc_id', 'score', 'query_language'.
        """
        parsed = self.query_processor.parse(query)
        results = self.search(query, top_k)

        return [
            {
                "doc_id": doc_id,
                "score": score,
                "query_language": parsed["language"],
            }
            for doc_id, score in results
        ]
