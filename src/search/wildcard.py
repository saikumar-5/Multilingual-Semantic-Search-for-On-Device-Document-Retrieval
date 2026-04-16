"""
Wildcard Query Search using bigram (2-gram) index.

CO2 Alignment: Implements wildcard queries and n-gram query formation.

Wildcard queries allow patterns like:
- "comp*"  → matches "computer", "compiler", "company"
- "*tion"  → matches "education", "information", "nation"
- "pro*ing" → matches "processing", "programming"

We use a bigram index to efficiently find candidate terms
that match the wildcard pattern.
"""

import re
from typing import List, Set, Tuple, Dict
from collections import defaultdict
import logging

from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine

logger = logging.getLogger(__name__)


class WildcardSearch:
    """
    Wildcard query support using a bigram (character 2-gram) index.

    The bigram index maps character pairs to terms:
        "$c" → {"cat", "computer", "class"}
        "co" → {"computer", "code", "company"}
        "er$" → {"computer", "compiler", "user"}

    To resolve "comp*":
    1. Generate bigrams: $c, co, om, mp
    2. Intersect posting lists to get candidate terms
    3. Filter candidates against the wildcard pattern
    4. Look up documents for matching terms
    """

    def __init__(self, inverted_index: InvertedIndex, tfidf_engine: TFIDFEngine):
        self.inv_index = inverted_index
        self.tfidf = tfidf_engine
        # Bigram index: {bigram: set of terms}
        self.bigram_index: Dict[str, Set[str]] = defaultdict(set)
        self._built = False

    def build(self):
        """Build the bigram index from the inverted index vocabulary."""
        for term in self.inv_index.vocabulary:
            padded = f"${term}$"
            for i in range(len(padded) - 1):
                bigram = padded[i : i + 2]
                self.bigram_index[bigram].add(term)

        self._built = True
        logger.info(
            f"Built bigram index: {len(self.bigram_index)} bigrams "
            f"from {len(self.inv_index.vocabulary)} terms"
        )

    def find_matching_terms(self, pattern: str) -> Set[str]:
        """
        Find all terms matching a wildcard pattern.

        Args:
            pattern: Wildcard pattern (e.g., "comp*", "*tion", "pro*ing")

        Returns:
            Set of matching terms from the vocabulary.
        """
        if not self._built:
            self.build()

        pattern = pattern.lower()

        # Convert wildcard to regex
        regex_pattern = "^" + pattern.replace("*", ".*").replace("?", ".") + "$"
        regex = re.compile(regex_pattern)

        # If pattern has no wildcard, just check vocabulary
        if "*" not in pattern and "?" not in pattern:
            if pattern in self.inv_index.vocabulary:
                return {pattern}
            return set()

        # Use bigram index to narrow candidates
        candidates = self._get_candidates(pattern)

        # Filter candidates with regex
        matching = {term for term in candidates if regex.match(term)}
        return matching

    def _get_candidates(self, pattern: str) -> Set[str]:
        """Use bigram index to find candidate terms for a wildcard pattern."""
        # Split pattern by wildcards to get fixed parts
        parts = re.split(r"[*?]+", pattern)
        parts = [p for p in parts if p]  # Remove empty parts

        if not parts:
            return set(self.inv_index.vocabulary)

        # Generate bigrams from fixed parts
        all_bigrams = []
        for part in parts:
            padded = part
            # Add $ boundary markers for prefix/suffix parts
            if pattern.startswith(part):
                padded = "$" + padded
            if pattern.endswith(part):
                padded = padded + "$"

            for i in range(len(padded) - 1):
                bigram = padded[i : i + 2]
                if bigram in self.bigram_index:
                    all_bigrams.append(bigram)

        if not all_bigrams:
            return set(self.inv_index.vocabulary)

        # Intersect bigram posting lists
        candidates = self.bigram_index[all_bigrams[0]].copy()
        for bigram in all_bigrams[1:]:
            candidates &= self.bigram_index.get(bigram, set())

        return candidates

    def search(self, pattern: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching a wildcard pattern.

        1. Find matching terms using bigram index
        2. Get documents containing those terms
        3. Rank by TF-IDF

        Returns:
            List of (doc_id, score) tuples.
        """
        matching_terms = self.find_matching_terms(pattern)

        if not matching_terms:
            logger.info(f"No terms match pattern: {pattern}")
            return []

        logger.info(f"Wildcard '{pattern}' matched {len(matching_terms)} terms")

        # Get all documents containing matching terms
        doc_scores: Dict[int, float] = defaultdict(float)

        for term in matching_terms:
            postings = self.inv_index.lookup(term)
            idf = self.tfidf.idf_values.get(term, 0.0)

            for doc_id, positions in postings.items():
                tf = len(positions) / max(
                    self.inv_index.doc_lengths.get(doc_id, 1), 1
                )
                doc_scores[doc_id] += tf * idf

        # Sort by score
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
