"""
Hybrid Search - Combines keyword (TF-IDF) and semantic (vector) search.

CO3 Alignment: Combined retrieval using multiple ranking signals.

Hybrid search fuses two ranked result lists using Reciprocal Rank Fusion (RRF):
1. Keyword Search (TF-IDF): Good for exact matches, specific terms, IDs
2. Semantic Search (multilingual-e5): Good for meaning, synonyms, cross-language

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
from src.search.query_processor import QueryProcessor
from src.config import (
    RRF_K,
    FUSION_METHOD,
    FUSION_KEYWORD_WEIGHT,
    FUSION_SEMANTIC_WEIGHT,
    FUSION_RERANKER_WEIGHT,
    FUSION_KEYWORD_DIGIT_BOOST,
    FUSION_KEYWORD_SHORT_QUERY_BOOST,
    FUSION_SEMANTIC_LONG_QUERY_BOOST,
    DEFAULT_TOP_K,
    RERANKER_MAX_INFLUENCE,
)

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
        self.query_processor = QueryProcessor()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search using both keyword and semantic methods, fuse results.

        Args:
            query: User's search query.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, fused_score) tuples sorted by relevance.
        """
        effective_top_k = max(1, int(top_k or DEFAULT_TOP_K))
        retrieval_k = max(effective_top_k * 2, self.rerank_candidates)

        # Get results from both engines
        keyword_results = self.keyword_search.search(query, retrieval_k)
        semantic_results = self.semantic_search.search(query, retrieval_k)

        # Rank-based fusion via Reciprocal Rank Fusion (RRF).
        fused = self._rrf_fuse(query, keyword_results, semantic_results)

        # Sort by fused score
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        if self.reranker is not None and self._should_apply_reranker(query):
            reranked = self.reranker.rerank(query, ranked, effective_top_k)
            blended = self._blend_reranker_scores(ranked, reranked, effective_top_k)
            return blended[:effective_top_k]

        return ranked[:effective_top_k]

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
        query: str,
        keyword_results: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        """Fuse ranked results using RRF, optionally weighted by query features."""
        fused: Dict[int, float] = {}

        kw_weight, sem_weight = self._get_fusion_weights(query)

        for rank, (doc_id, _) in enumerate(keyword_results, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + (
                kw_weight * (1.0 / (self.rrf_k + rank))
            )

        for rank, (doc_id, _) in enumerate(semantic_results, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + (
                sem_weight * (1.0 / (self.rrf_k + rank))
            )

        return fused

    def _blend_reranker_scores(
        self,
        fused_ranked: List[Tuple[int, float]],
        reranked: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """Blend fused scores with reranker scores using configured weight."""
        rerank_weight = max(0.0, min(1.0, float(FUSION_RERANKER_WEIGHT)))
        rerank_weight = min(rerank_weight, float(RERANKER_MAX_INFLUENCE))
        if rerank_weight <= 0 or not reranked:
            return reranked[:top_k]

        fused_scores = {doc_id: score for doc_id, score in fused_ranked}
        rerank_scores = {doc_id: score for doc_id, score in reranked}

        fused_norm = self._min_max_normalize(fused_scores)
        rerank_norm = self._min_max_normalize(rerank_scores)

        all_ids = set(fused_norm) | set(rerank_norm)
        blended = []
        for doc_id in all_ids:
            score = (
                (1.0 - rerank_weight) * fused_norm.get(doc_id, 0.0)
                + rerank_weight * rerank_norm.get(doc_id, 0.0)
            )
            blended.append((doc_id, score))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:top_k]

    @staticmethod
    def _min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v <= 1e-12:
            return {doc_id: 0.0 for doc_id in scores}
        return {doc_id: (val - min_v) / (max_v - min_v) for doc_id, val in scores.items()}

    def _get_fusion_weights(self, query: str) -> Tuple[float, float]:
        """Compute fusion weights from query heuristics."""
        if (FUSION_METHOD or "rrf").lower() != "weighted_rrf":
            return 1.0, 1.0

        kw_weight = float(FUSION_KEYWORD_WEIGHT)
        sem_weight = float(FUSION_SEMANTIC_WEIGHT)

        parsed = self.query_processor.parse(query)
        terms = parsed.get("original_terms", []) or parsed.get("terms", [])
        term_count = len(terms)

        if re.search(r"\d", query):
            kw_weight += float(FUSION_KEYWORD_DIGIT_BOOST)

        if term_count > 0 and term_count <= 2:
            kw_weight += float(FUSION_KEYWORD_SHORT_QUERY_BOOST)

        if term_count >= 6:
            sem_weight += float(FUSION_SEMANTIC_LONG_QUERY_BOOST)

        return max(0.1, kw_weight), max(0.1, sem_weight)
