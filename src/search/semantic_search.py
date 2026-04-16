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
        # Clean query for embedding (remove Boolean operators etc.)
        clean_query = self.query_processor.get_raw_text_for_embedding(query)

        if not clean_query:
            return []

        results = self.vector_index.search(clean_query, top_k)
        return results

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
