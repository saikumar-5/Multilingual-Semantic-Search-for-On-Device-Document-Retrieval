"""
Query expansion using local synonym dictionaries.

Expands only short queries and preserves original terms with higher weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

from src.config import (
    QUERY_SYNONYMS_PATH,
    QUERY_EXPANSION_SHORT_QUERY_MAX_TOKENS,
    QUERY_EXPANSION_MAX_SYNONYMS_PER_TERM,
    QUERY_EXPANSION_ORIGINAL_WEIGHT,
    QUERY_EXPANSION_EXPANDED_WEIGHT,
)

logger = logging.getLogger(__name__)


class QueryExpander:
    """Lightweight offline query expansion via local synonym dictionary."""

    # Generic intent words where expansion can help sparse short queries.
    GENERIC_TERMS = {
        "en": {
            "document", "documents", "file", "files", "form", "letter",
            "report", "project", "info", "information", "details", "data",
            "record", "records", "certificate", "application",
        },
        "hi": {
            "दस्तावेज", "दस्तावेज़", "फाइल", "सूचना", "जानकारी", "रिपोर्ट",
            "प्रमाणपत्र", "पत्र", "विवरण", "रिकॉर्ड",
        },
        "te": {
            "పత్రం", "పత్రాలు", "ఫైల్", "సమాచారం", "వివరాలు", "రిపోర్ట్",
            "సర్టిఫికేట్", "రికార్డు", "దరఖాస్తు",
        },
    }

    # Technical/domain-specific terms that should not be expanded in phrases.
    TECHNICAL_TERMS = {
        "en": {
            "network", "security", "tfidf", "faiss", "hnsw", "semantic",
            "cross", "encoder", "index", "indexing", "vector", "embedding",
            "database", "api", "ocr", "retrieval", "ranking",
        },
        "hi": {
            "नेटवर्क", "सुरक्षा", "डेटाबेस", "इंडेक्स", "एम्बेडिंग",
        },
        "te": {
            "నెట్వర్క్", "భద్రత", "డేటాబేస్", "ఇండెక్స్", "ఎంబెడింగ్",
        },
    }

    # Strong multi-term phrases that already carry specific meaning.
    STRONG_PHRASES = {
        "en": {
            "network security",
            "machine learning",
            "software engineering",
            "query expansion",
            "information retrieval",
            "cross encoder",
        },
        "hi": set(),
        "te": set(),
    }

    def __init__(self, synonyms_path: Path = QUERY_SYNONYMS_PATH):
        self.synonyms_path = synonyms_path
        self._synonyms = self._load_synonyms()

    def _load_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        if not self.synonyms_path.exists():
            logger.warning("Synonym dictionary not found at %s", self.synonyms_path)
            return {}

        try:
            with open(self.synonyms_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("Invalid synonym dictionary format; expected object")
                return {}

            return data
        except Exception as e:
            logger.warning("Failed to load synonym dictionary %s: %s", self.synonyms_path, e)
            return {}

    def is_expandable(self, terms: List[str], language: str) -> Tuple[bool, str]:
        """Determine if a query is safe/beneficial for expansion."""
        cleaned = [t for t in terms if t]
        if not cleaned:
            return False, "empty_query"

        if len(cleaned) > QUERY_EXPANSION_SHORT_QUERY_MAX_TOKENS:
            return False, "too_many_tokens"

        # Single-word queries are the primary expansion candidates.
        if len(cleaned) == 1:
            return True, "single_word_query"

        lang_key = language if language in ("en", "hi", "te") else "en"
        normalized = [t.lower() if lang_key == "en" else t for t in cleaned]
        phrase = " ".join(normalized)

        if phrase in self.STRONG_PHRASES.get(lang_key, set()):
            return False, "strong_phrase"

        tech = self.TECHNICAL_TERMS.get(lang_key, set())
        if any(t in tech for t in normalized):
            return False, "technical_phrase"

        # For multi-term queries, expand only when terms are generic.
        generic = self.GENERIC_TERMS.get(lang_key, set())
        if all(t in generic for t in normalized):
            return True, "generic_multi_term_query"

        return False, "specific_multi_term_query"

    def expand(self, terms: List[str], language: str) -> Dict[str, object]:
        """
        Expand query terms and return weighted term mapping.

        Output schema:
        {
          "original_terms": [...],
          "expanded_terms": [...],
          "weighted_terms": {...}
        }
        """
        original_terms = [t for t in terms if t]
        if not original_terms:
            return {
                "original_terms": [],
                "expanded_terms": [],
                "weighted_terms": {},
            }

        weighted_terms: Dict[str, float] = {
            t: float(QUERY_EXPANSION_ORIGINAL_WEIGHT) for t in original_terms
        }

        should_expand, reason = self.is_expandable(original_terms, language)
        if not should_expand:
            return {
                "original_terms": original_terms,
                "expanded_terms": [],
                "weighted_terms": weighted_terms,
                "expandable": False,
                "expand_reason": reason,
            }

        lang_key = language if language in ("en", "hi", "te") else "en"
        lang_dict = self._synonyms.get(lang_key, {})

        expanded_terms: List[str] = []
        max_synonyms = max(0, int(QUERY_EXPANSION_MAX_SYNONYMS_PER_TERM))

        for term in original_terms:
            lookup_term = term.lower() if lang_key == "en" else term
            synonyms = lang_dict.get(lookup_term, [])
            if not isinstance(synonyms, list):
                continue

            added = 0
            for candidate in synonyms:
                if added >= max_synonyms:
                    break
                if not isinstance(candidate, str):
                    continue

                norm_candidate = candidate.strip()
                if not norm_candidate:
                    continue

                if norm_candidate in weighted_terms:
                    continue

                expanded_terms.append(norm_candidate)
                weighted_terms[norm_candidate] = float(QUERY_EXPANSION_EXPANDED_WEIGHT)
                added += 1

        return {
            "original_terms": original_terms,
            "expanded_terms": expanded_terms,
            "weighted_terms": weighted_terms,
            "expandable": True,
            "expand_reason": reason,
        }
