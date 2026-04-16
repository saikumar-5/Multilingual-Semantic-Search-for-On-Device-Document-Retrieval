"""
Hybrid Search - Combines keyword (TF-IDF) and semantic (vector) search.

CO3 Alignment: Combined retrieval using multiple ranking signals.

Hybrid search fuses scores from two complementary approaches:
1. Keyword Search (TF-IDF): Good for exact matches, specific terms, IDs
2. Semantic Search (MiniLM): Good for meaning, synonyms, cross-language

Score Fusion Formula:
    final_score = alpha * semantic_score + (1 - alpha) * keyword_score

Where alpha controls the balance (default 0.6 = slightly favor semantic).
"""

from typing import List, Tuple, Dict
import logging

from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch
from src.config import HYBRID_ALPHA

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search combining keyword and semantic retrieval.

    This approach leverages the strengths of both methods:
    - Keyword search catches exact matches (PAN numbers, specific names)
    - Semantic search catches conceptual matches (synonyms, translations)
    """

    def __init__(
        self,
        keyword_search: KeywordSearch,
        semantic_search: SemanticSearch,
        alpha: float = HYBRID_ALPHA,
    ):
        """
        Args:
            keyword_search: KeywordSearch instance.
            semantic_search: SemanticSearch instance.
            alpha: Weight for semantic score (0-1). Higher = more semantic.
        """
        self.keyword_search = keyword_search
        self.semantic_search = semantic_search
        self.alpha = alpha

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search using both keyword and semantic methods, fuse results.

        Args:
            query: User's search query.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, fused_score) tuples sorted by relevance.
        """
        # Get results from both engines
        keyword_results = self.keyword_search.search(query, top_k * 2)
        semantic_results = self.semantic_search.search(query, top_k * 2)

        # Normalize scores to [0, 1] range
        kw_scores = self._normalize_scores(keyword_results)
        sem_scores = self._normalize_scores(semantic_results)

        # Fuse scores
        all_doc_ids = set(kw_scores.keys()) | set(sem_scores.keys())
        fused = {}

        for doc_id in all_doc_ids:
            kw = kw_scores.get(doc_id, 0.0)
            sem = sem_scores.get(doc_id, 0.0)
            fused[doc_id] = self.alpha * sem + (1 - self.alpha) * kw

        # Sort by fused score
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_with_details(
        self, query: str, top_k: int = 10
    ) -> List[dict]:
        """
        Search with detailed score breakdown for each result.

        Returns:
            List of dicts with 'doc_id', 'final_score',
            'keyword_score', 'semantic_score'.
        """
        keyword_results = self.keyword_search.search(query, top_k * 2)
        semantic_results = self.semantic_search.search(query, top_k * 2)

        kw_scores = self._normalize_scores(keyword_results)
        sem_scores = self._normalize_scores(semantic_results)

        all_doc_ids = set(kw_scores.keys()) | set(sem_scores.keys())
        results = []

        for doc_id in all_doc_ids:
            kw = kw_scores.get(doc_id, 0.0)
            sem = sem_scores.get(doc_id, 0.0)
            final = self.alpha * sem + (1 - self.alpha) * kw

            results.append({
                "doc_id": doc_id,
                "final_score": final,
                "keyword_score": kw,
                "semantic_score": sem,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def _normalize_scores(
        self, results: List[Tuple[int, float]]
    ) -> Dict[int, float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        This is necessary because keyword scores (TF-IDF cosine) and
        semantic scores (MiniLM cosine) are on different scales.
        """
        if not results:
            return {}

        scores = [s for _, s in results]
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s

        if range_s == 0:
            return {doc_id: 1.0 for doc_id, _ in results}

        return {
            doc_id: (score - min_s) / range_s for doc_id, score in results
        }
