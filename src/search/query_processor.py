"""
Query Processor - Parses, preprocesses, and routes user queries.

CO2 Alignment: Query representation and processing.
Handles free-text queries, Boolean queries, wildcard queries,
and phrase queries.
"""

import re
from typing import List, Tuple
import logging

from src.indexer.preprocessor import Preprocessor
from src.search.query_expander import QueryExpander
from src.config import (
    ENABLE_QUERY_EXPANSION,
    INTENT_KEYWORDS,
    INTENT_CANONICAL_TERMS,
    INTENT_TERM_BOOST,
    INTENT_MAX_ADDED_TERMS,
    QUERY_TERM_BOOSTS,
    QUERY_PHRASE_EXPANSIONS,
    QUERY_PHRASE_BOOST,
    FAILING_QUERY_EXPANSIONS,
    FAILING_QUERY_ORIGINAL_WEIGHT,
    FAILING_QUERY_EXPANDED_WEIGHT,
)

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Parse and preprocess user queries for different search modes."""

    def __init__(self):
        self.preprocessor = Preprocessor(remove_stopwords=False)
        self.preprocessor_filtered = Preprocessor(remove_stopwords=True)
        self.query_expander = QueryExpander()

    def parse(self, query: str) -> dict:
        """
        Parse a raw query string and determine its type and components.

        Returns:
            dict with keys:
                'raw': original query
                'type': 'boolean', 'phrase', 'wildcard', or 'free_text'
                'terms': preprocessed terms
                'language': detected language
        """
        query = query.strip()
        if not query:
            return {
                "raw": "",
                "type": "free_text",
                "terms": [],
                "language": "en",
                "original_terms": [],
                "expanded_terms": [],
                "weighted_terms": {},
                "expandable": False,
                "expand_reason": "empty_query",
                "intent": [],
                "intent_terms": [],
                "use_weighted_embedding": False,
            }

        language = self.preprocessor.detect_language(query)

        # Detect query type
        if self._is_boolean(query):
            terms = self.preprocessor_filtered.preprocess(query)
            return {
                "raw": query,
                "type": "boolean",
                "terms": terms,
                "language": language,
                "original_terms": terms,
                "expanded_terms": [],
                "weighted_terms": {t: 1.0 for t in terms},
                "expandable": False,
                "expand_reason": "boolean_query",
                "intent": [],
                "intent_terms": [],
                "use_weighted_embedding": False,
            }

        if self._is_phrase(query):
            # Remove quotes for phrase search
            phrase = query.strip('"').strip("'")
            phrase_terms = self.preprocessor_filtered.preprocess(phrase)
            return {
                "raw": phrase,
                "type": "phrase",
                "terms": phrase_terms,
                "language": language,
                "original_terms": phrase_terms,
                "expanded_terms": [],
                "weighted_terms": {t: 1.0 for t in phrase_terms},
                "expandable": False,
                "expand_reason": "phrase_query",
                "intent": [],
                "intent_terms": [],
                "use_weighted_embedding": False,
            }

        if self._is_wildcard(query):
            return {
                "raw": query,
                "type": "wildcard",
                "terms": [query.lower()],
                "language": language,
                "original_terms": [query.lower()],
                "expanded_terms": [],
                "weighted_terms": {query.lower(): 1.0},
                "expandable": False,
                "expand_reason": "wildcard_query",
                "intent": [],
                "intent_terms": [],
                "use_weighted_embedding": False,
            }

        # Default: free-text query
        terms = self.preprocessor_filtered.preprocess(query)
        intent = self._detect_intent(query, terms)
        intent_terms = self._build_intent_terms(intent)
        if ENABLE_QUERY_EXPANSION:
            expansion = self.query_expander.expand(terms, language)
        else:
            expansion = {
                "original_terms": terms,
                "expanded_terms": [],
                "weighted_terms": {t: 1.0 for t in terms},
                "expandable": False,
                "expand_reason": "query_expansion_disabled",
            }

        weighted_terms = dict(expansion["weighted_terms"])
        if intent_terms:
            for term in intent_terms:
                if term not in weighted_terms:
                    weighted_terms[term] = float(INTENT_TERM_BOOST)

        weighted_terms = self._apply_term_boosts(weighted_terms)
        weighted_terms = self._apply_phrase_expansions(query, weighted_terms)
        weighted_terms = self._apply_failing_expansions(query, weighted_terms)

        merged_terms = list(weighted_terms.keys())
        return {
            "raw": query,
            "type": "free_text",
            "terms": merged_terms,
            "language": language,
            "original_terms": expansion["original_terms"],
            "expanded_terms": expansion["expanded_terms"],
            "weighted_terms": weighted_terms,
            "expandable": expansion.get("expandable", False),
            "expand_reason": expansion.get("expand_reason", "unknown"),
            "intent": intent,
            "intent_terms": intent_terms,
            "use_weighted_embedding": bool(intent_terms or expansion["expanded_terms"]),
        }

    def _is_boolean(self, query: str) -> bool:
        """Check if query contains Boolean operators."""
        upper = query.upper()
        return " AND " in upper or " OR " in upper or " NOT " in upper

    def _is_phrase(self, query: str) -> bool:
        """Check if query is a phrase (enclosed in quotes)."""
        return (query.startswith('"') and query.endswith('"')) or (
            query.startswith("'") and query.endswith("'")
        )

    def _is_wildcard(self, query: str) -> bool:
        """Check if query contains wildcard characters."""
        return "*" in query or "?" in query

    def _detect_intent(self, query: str, terms: List[str]) -> List[str]:
        """Detect document intent categories based on query tokens."""
        if not query:
            return []

        normalized = query.lower()
        term_set = set(t.lower() for t in terms if t)
        intents = []

        for intent_name, keywords in INTENT_KEYWORDS.items():
            matched = False
            for keyword in keywords:
                kw = keyword.lower()
                if " " in kw:
                    if kw in normalized:
                        matched = True
                        break
                else:
                    if kw in term_set:
                        matched = True
                        break
            if matched:
                intents.append(intent_name)

        return intents

    def _build_intent_terms(self, intents: List[str]) -> List[str]:
        if not intents:
            return []

        terms: List[str] = []
        for intent_name in intents:
            for term in INTENT_CANONICAL_TERMS.get(intent_name, []):
                if term not in terms:
                    terms.append(term)
                if len(terms) >= INTENT_MAX_ADDED_TERMS:
                    return terms

        return terms

    def _apply_term_boosts(self, weighted_terms: dict) -> dict:
        if not weighted_terms:
            return weighted_terms

        boosted = dict(weighted_terms)
        for term, boost in QUERY_TERM_BOOSTS.items():
            if term in boosted:
                boosted[term] = max(float(boosted[term]), float(boost))

        return boosted

    def _apply_phrase_expansions(self, query: str, weighted_terms: dict) -> dict:
        if not query:
            return weighted_terms

        normalized = query.lower()
        expanded = dict(weighted_terms)
        for phrase, terms in QUERY_PHRASE_EXPANSIONS.items():
            if phrase in normalized:
                for term in terms:
                    if term not in expanded:
                        expanded[term] = float(QUERY_PHRASE_BOOST)
                    else:
                        expanded[term] = max(float(expanded[term]), float(QUERY_PHRASE_BOOST))

        return expanded

    def _apply_failing_expansions(self, query: str, weighted_terms: dict) -> dict:
        if not query:
            return weighted_terms

        normalized = query.lower()
        expanded = dict(weighted_terms)

        for trigger, expansions in FAILING_QUERY_EXPANSIONS.items():
            if trigger in normalized:
                for term in expansions:
                    if term not in expanded:
                        expanded[term] = float(FAILING_QUERY_EXPANDED_WEIGHT)

        # Ensure originals stay at configured weight.
        for term in list(expanded.keys()):
            if term in normalized.split():
                expanded[term] = max(expanded[term], float(FAILING_QUERY_ORIGINAL_WEIGHT))

        return expanded

    def get_terms_for_tfidf(self, query: str) -> List[str]:
        """Get preprocessed terms suitable for TF-IDF ranking."""
        return self.preprocessor_filtered.preprocess(query)

    def get_raw_text_for_embedding(self, query: str) -> str:
        """
        Get cleaned query text for semantic embedding.
        Keep more of the original text for better semantic understanding.
        """
        # Remove Boolean operators but keep the words
        text = re.sub(r"\b(AND|OR|NOT)\b", " ", query, flags=re.IGNORECASE)
        # Remove wildcard characters
        text = text.replace("*", "").replace("?", "")
        # Remove quotes
        text = text.strip('"').strip("'")
        return text.strip()
