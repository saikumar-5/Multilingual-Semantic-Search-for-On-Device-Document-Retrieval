"""
Hybrid Search - Combines keyword (TF-IDF) and semantic (vector) search.

CO3 Alignment: Combined retrieval using multiple ranking signals.

Hybrid search fuses two ranked result lists using Reciprocal Rank Fusion (RRF):
1. Keyword Search (TF-IDF): Good for exact matches, specific terms, IDs
2. Semantic Search (MiniLM): Good for meaning, synonyms, cross-language

RRF Formula:
    score(d) = sum(1 / (k + rank_i(d)))

Where rank_i(d) is the 1-based rank of document d in ranking i, and k is a
stability constant (default: 60).
"""

from typing import List, Tuple, Dict
import logging
import re

from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch
from src.config import RRF_K

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
        rrf_k: int = RRF_K,
        reranker=None,
        rerank_candidates: int = 50,
        rerank_english_only: bool = True,
    ):
        """
        Args:
            keyword_search: KeywordSearch instance.
            semantic_search: SemanticSearch instance.
            rrf_k: RRF constant (higher values smooth rank differences).
            reranker: Optional second-stage reranker with rerank() method.
            rerank_candidates: Candidate pool size for reranking.
            rerank_english_only: If True, rerank only for English queries.
        """
        self.keyword_search = keyword_search
        self.semantic_search = semantic_search
        self.rrf_k = max(1, rrf_k)
        self.reranker = reranker
        self.rerank_candidates = max(1, rerank_candidates)
        self.rerank_english_only = rerank_english_only

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search using both keyword and semantic methods, fuse results.

        Args:
            query: User's search query.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, fused_score) tuples sorted by relevance.
        """
        retrieval_k = max(top_k * 2, self.rerank_candidates)

        # Get results from both engines
        keyword_results = self.keyword_search.search(query, retrieval_k)
        semantic_results = self.semantic_search.search(query, retrieval_k)

        # Rank-based fusion via Reciprocal Rank Fusion (RRF).
        fused = self._rrf_fuse(keyword_results, semantic_results)

        # Sort by fused score
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        if self.reranker is not None and self._should_apply_reranker(query):
            return self.reranker.rerank(query, ranked, top_k)

        return ranked[:top_k]

    def _detect_query_language(self, query: str) -> str:
        """
        Lightweight script-based language detection for rerank gating.

        Returns one of: 'en', 'hi', 'te', or 'mixed'.
        """
        if not query:
            return "en"

        devanagari_count = len(re.findall(r"[\u0900-\u097F]", query))
        telugu_count = len(re.findall(r"[\u0C00-\u0C7F]", query))
        latin_count = len(re.findall(r"[a-zA-Z]", query))

        total = devanagari_count + telugu_count + latin_count
        if total == 0:
            return "en"

        if devanagari_count / total > 0.3:
            return "hi"
        if telugu_count / total > 0.3:
            return "te"
        if latin_count / total > 0.5:
            return "en"
        return "mixed"

    def _should_apply_reranker(self, query: str) -> bool:
        """Apply reranker for English queries; skip for non-English queries."""
        if not self.rerank_english_only:
            return True

        language = self._detect_query_language(query)
        should_rerank = language == "en"
        if not should_rerank:
            logger.info(
                "Skipping cross-encoder reranking for non-English query (detected: %s)",
                language,
            )
        return should_rerank

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

        kw_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(keyword_results, 1)}
        sem_ranks = {
            doc_id: rank for rank, (doc_id, _) in enumerate(semantic_results, 1)
        }

        all_doc_ids = set(kw_ranks.keys()) | set(sem_ranks.keys())
        results = []

        for doc_id in all_doc_ids:
            kw_rank = kw_ranks.get(doc_id)
            sem_rank = sem_ranks.get(doc_id)

            kw_rrf = 1.0 / (self.rrf_k + kw_rank) if kw_rank is not None else 0.0
            sem_rrf = 1.0 / (self.rrf_k + sem_rank) if sem_rank is not None else 0.0
            final = kw_rrf + sem_rrf

            results.append({
                "doc_id": doc_id,
                "final_score": final,
                "keyword_rank": kw_rank,
                "semantic_rank": sem_rank,
                "keyword_rrf": kw_rrf,
                "semantic_rrf": sem_rrf,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def _rrf_fuse(
        self,
        keyword_results: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        """Fuse ranked results using Reciprocal Rank Fusion."""
        fused: Dict[int, float] = {}

        for rank, (doc_id, _) in enumerate(keyword_results, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (doc_id, _) in enumerate(semantic_results, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        return fused
