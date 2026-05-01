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
from src.config import ENABLE_QUERY_EXPANSION

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
            }

        # Default: free-text query
        terms = self.preprocessor_filtered.preprocess(query)
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

        merged_terms = list(expansion["weighted_terms"].keys())
        return {
            "raw": query,
            "type": "free_text",
            "terms": merged_terms,
            "language": language,
            "original_terms": expansion["original_terms"],
            "expanded_terms": expansion["expanded_terms"],
            "weighted_terms": expansion["weighted_terms"],
            "expandable": expansion.get("expandable", False),
            "expand_reason": expansion.get("expand_reason", "unknown"),
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
