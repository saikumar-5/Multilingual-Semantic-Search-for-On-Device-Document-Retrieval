"""
Keyword Search engine using Inverted Index and TF-IDF ranking.

CO1/CO2 Alignment: Uses inverted index for lookup and TF-IDF for ranking.
Supports Boolean queries, phrase queries, and free-text ranked retrieval.
"""

from typing import List, Tuple
import logging

from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine
from src.search.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class KeywordSearch:
    """
    Keyword-based search combining Inverted Index lookup with TF-IDF ranking.
    """

    def __init__(self, inverted_index: InvertedIndex, tfidf_engine: TFIDFEngine):
        self.inv_index = inverted_index
        self.tfidf = tfidf_engine
        self.query_processor = QueryProcessor()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search documents using keyword matching with TF-IDF ranking.

        Args:
            query: User's search query string.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples sorted by relevance.
        """
        parsed = self.query_processor.parse(query)

        if not parsed["terms"]:
            return []

        if parsed["type"] == "boolean":
            return self._boolean_search(query, top_k)
        elif parsed["type"] == "phrase":
            return self._phrase_search(parsed["raw"], top_k)
        else:
            return self._ranked_search(
                parsed["terms"],
                top_k,
                weighted_terms=parsed.get("weighted_terms"),
            )

    def _ranked_search(
        self,
        terms: List[str],
        top_k: int,
        weighted_terms=None,
    ) -> List[Tuple[int, float]]:
        """Free-text search with TF-IDF cosine similarity ranking."""
        return self.tfidf.rank_documents(terms, top_k, weighted_terms=weighted_terms)

    def _boolean_search(
        self, query: str, top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Boolean search: find matching docs, then rank by TF-IDF.

        We first use Boolean retrieval to get candidate docs,
        then rank them using TF-IDF for ordering.
        """
        matching_docs = self.inv_index.boolean_search(query)

        if not matching_docs:
            return []

        # Rank the Boolean results using TF-IDF
        terms = self.query_processor.get_terms_for_tfidf(query)
        query_vector = self.tfidf.get_query_vector(terms)

        scored = []
        for doc_id in matching_docs:
            score = self.tfidf.cosine_similarity(query_vector, doc_id)
            scored.append((doc_id, max(score, 0.01)))  # Ensure non-zero for Boolean match

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _phrase_search(
        self, phrase: str, top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Phrase search: find docs with exact phrase, then rank by TF-IDF.
        """
        matching_docs = self.inv_index.phrase_search(phrase)

        if not matching_docs:
            return []

        # Rank the phrase results using TF-IDF
        terms = self.query_processor.get_terms_for_tfidf(phrase)
        query_vector = self.tfidf.get_query_vector(terms)

        scored = []
        for doc_id in matching_docs:
            score = self.tfidf.cosine_similarity(query_vector, doc_id)
            scored.append((doc_id, max(score, 0.01)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
