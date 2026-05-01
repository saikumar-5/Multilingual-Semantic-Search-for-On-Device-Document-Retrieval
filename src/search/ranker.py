"""
Result Ranker - Formats and enriches search results with metadata.

CO3 Alignment: Ranking functions and result presentation.
"""

from typing import List, Tuple, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Ranker:
    """
    Formats raw (doc_id, score) results into rich result objects
    with file metadata, text snippets, and language tags.
    """

    def __init__(self, documents: List[dict]):
        """
        Args:
            documents: List of document dicts from ingestion pipeline.
        """
        # Build lookup: doc_id → document dict
        self.doc_lookup: Dict[int, dict] = {}
        for doc in documents:
            self.doc_lookup[doc["doc_id"]] = doc

    def format_results(
        self, ranked_results: List[Tuple[int, float]], query: str = ""
    ) -> List[dict]:
        """
        Convert (doc_id, score) pairs into rich result objects.

        Returns:
            List of dicts with keys:
                'rank', 'doc_id', 'score', 'file_name', 'file_path',
                'file_type', 'snippet', 'language'
        """
        raw_scores = [float(score) for _, score in ranked_results]
        display_scores = self._normalize_display_scores(raw_scores)

        results = []
        for i, (doc_id, score) in enumerate(ranked_results):
            rank = i + 1
            doc = self.doc_lookup.get(doc_id)
            if doc is None:
                continue

            snippet = self._extract_snippet(doc.get("text", ""), query)
            language = self._detect_doc_language(doc.get("text", ""))

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                # Keep UI score in [0, 1] so percentage badges are meaningful.
                "score": round(display_scores[i], 4) if i < len(display_scores) else 0.0,
                "raw_score": round(float(score), 4),
                "file_name": doc.get("file_name", "Unknown"),
                "file_path": doc.get("file_path", ""),
                "file_type": doc.get("file_type", "unknown"),
                "snippet": snippet,
                "language": language,
                "metadata": doc.get("metadata", {}),
            })

        return results

    def _normalize_display_scores(self, raw_scores: List[float]) -> List[float]:
        """Map heterogeneous backend scores to [0, 1] for UI display only."""
        if not raw_scores:
            return []

        # If scores already look like probabilities/similarities, keep as-is.
        if all(0.0 <= s <= 1.0 for s in raw_scores):
            return [float(s) for s in raw_scores]

        min_s = min(raw_scores)
        max_s = max(raw_scores)
        if max_s - min_s <= 1e-12:
            return [0.5 for _ in raw_scores]

        return [(float(s) - min_s) / (max_s - min_s) for s in raw_scores]

    def _extract_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """
        Extract a relevant text snippet around the query terms.

        Tries to center the snippet around the first occurrence
        of a query term for context.
        """
        if not text:
            return ""

        if not query:
            return text[:max_length] + ("..." if len(text) > max_length else "")

        # Find first occurrence of any query word
        text_lower = text.lower()
        query_terms = query.lower().split()

        best_pos = -1
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                if best_pos == -1 or pos < best_pos:
                    best_pos = pos

        if best_pos == -1:
            # No exact match found, return start of document
            return text[:max_length] + ("..." if len(text) > max_length else "")

        # Center snippet around the match
        start = max(0, best_pos - max_length // 3)
        end = min(len(text), start + max_length)

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def _detect_doc_language(self, text: str) -> str:
        """Simple language detection based on character ranges."""
        import re

        if not text:
            return "en"

        sample = text[:500]
        devanagari = len(re.findall(r"[\u0900-\u097F]", sample))
        telugu = len(re.findall(r"[\u0C00-\u0C7F]", sample))
        latin = len(re.findall(r"[a-zA-Z]", sample))

        total = devanagari + telugu + latin
        if total == 0:
            return "en"

        if devanagari / total > 0.3:
            return "Hindi"
        elif telugu / total > 0.3:
            return "Telugu"
        else:
            return "English"
