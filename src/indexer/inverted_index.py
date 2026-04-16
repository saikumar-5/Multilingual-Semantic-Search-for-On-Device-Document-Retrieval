"""
Inverted Index with positional information and Boolean search support.

CO1 Alignment: Implements the core Inverted Indexing concept.
Structure: {term: {doc_id: [position1, position2, ...]}}

This is a fundamental IR data structure where each term maps to
the documents containing it, along with the positions where it appears.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
import logging

from src.indexer.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class InvertedIndex:
    """
    Positional Inverted Index for document retrieval.

    The index stores: term → {doc_id: [positions]}
    This supports:
    - Simple term lookup
    - Boolean queries (AND, OR, NOT)
    - Phrase queries (using positional information)
    """

    def __init__(self):
        # Main index: {term: {doc_id: [positions]}}
        self.index: Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Document frequency: {term: number of docs containing term}
        self.doc_freq: Dict[str, int] = defaultdict(int)
        # Document lengths: {doc_id: total token count}
        self.doc_lengths: Dict[int, int] = {}
        # Total number of documents
        self.num_docs: int = 0
        # Vocabulary (all unique terms)
        self.vocabulary: Set[str] = set()

        self.preprocessor = Preprocessor()

    def add_document(self, doc_id: int, text: str):
        """
        Add a document to the inverted index.

        Tokenizes the text, records term positions, and updates
        document frequency counts.

        Args:
            doc_id: Unique document identifier.
            text: Raw text content of the document.
        """
        tokens_with_positions = self.preprocessor.preprocess_with_positions(text)

        if not tokens_with_positions:
            return

        # Track unique terms in this document for doc_freq update
        seen_terms = set()

        for token, position in tokens_with_positions:
            self.index[token][doc_id].append(position)
            self.vocabulary.add(token)

            if token not in seen_terms:
                self.doc_freq[token] += 1
                seen_terms.add(token)

        self.doc_lengths[doc_id] = len(tokens_with_positions)

    def build(self, documents: List[dict]):
        """
        Build the index from a list of document dicts.

        Args:
            documents: List of dicts with 'doc_id' and 'text' keys.
        """
        self.num_docs = len(documents)
        for doc in documents:
            self.add_document(doc["doc_id"], doc["text"])

        logger.info(
            f"Built inverted index: {len(self.vocabulary)} terms, "
            f"{self.num_docs} documents"
        )

    def lookup(self, term: str) -> Dict[int, List[int]]:
        """
        Look up a single term in the index.

        Returns:
            Dict mapping doc_id → [positions] for documents containing the term.
        """
        term = term.lower() if term.isascii() else term
        return dict(self.index.get(term, {}))

    def boolean_and(self, terms: List[str]) -> Set[int]:
        """
        Boolean AND: Return doc_ids containing ALL terms.

        CO2 Alignment: Boolean Model retrieval.
        """
        if not terms:
            return set()

        result = set(self.lookup(terms[0]).keys())
        for term in terms[1:]:
            result &= set(self.lookup(term).keys())
        return result

    def boolean_or(self, terms: List[str]) -> Set[int]:
        """
        Boolean OR: Return doc_ids containing ANY term.
        """
        result = set()
        for term in terms:
            result |= set(self.lookup(term).keys())
        return result

    def boolean_not(self, terms: List[str], universe: Set[int] = None) -> Set[int]:
        """
        Boolean NOT: Return doc_ids NOT containing any of the terms.
        """
        if universe is None:
            universe = set(self.doc_lengths.keys())
        exclude = self.boolean_or(terms)
        return universe - exclude

    def boolean_search(self, query: str) -> Set[int]:
        """
        Parse and execute a Boolean query.
        Supports AND, OR, NOT operators.

        Examples:
            "machine AND learning"
            "python OR java"
            "network AND NOT security"

        CO2 Alignment: Boolean Model retrieval.
        """
        query = query.strip()

        # Handle NOT
        if " AND NOT " in query.upper():
            parts = query.upper().split(" AND NOT ", 1)
            include_terms = self._extract_terms(parts[0])
            exclude_terms = self._extract_terms(parts[1])
            include_docs = self.boolean_and(include_terms)
            exclude_docs = self.boolean_or(exclude_terms)
            return include_docs - exclude_docs

        # Handle AND
        if " AND " in query.upper():
            parts = query.upper().split(" AND ")
            all_terms = []
            for part in parts:
                all_terms.extend(self._extract_terms(part))
            return self.boolean_and(all_terms)

        # Handle OR
        if " OR " in query.upper():
            parts = query.upper().split(" OR ")
            all_terms = []
            for part in parts:
                all_terms.extend(self._extract_terms(part))
            return self.boolean_or(all_terms)

        # Simple query (implicit OR across terms)
        terms = self._extract_terms(query)
        return self.boolean_or(terms)

    def phrase_search(self, phrase: str) -> Set[int]:
        """
        Search for an exact phrase using positional information.
        Terms must appear consecutively in the document.
        """
        terms = self.preprocessor.preprocess(phrase)
        if len(terms) < 2:
            return self.boolean_and(terms)

        # Get postings for first term
        first_postings = self.lookup(terms[0])
        if not first_postings:
            return set()

        result = set()
        for doc_id, positions in first_postings.items():
            for start_pos in positions:
                # Check if subsequent terms appear at consecutive positions
                found = True
                for offset, term in enumerate(terms[1:], 1):
                    term_postings = self.lookup(term)
                    if doc_id not in term_postings:
                        found = False
                        break
                    if (start_pos + offset) not in term_postings[doc_id]:
                        found = False
                        break
                if found:
                    result.add(doc_id)
                    break  # Found in this doc, move to next

        return result

    def _extract_terms(self, text: str) -> List[str]:
        """Extract and preprocess search terms from query text."""
        return self.preprocessor.preprocess(text)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get how many times a term appears in a specific document."""
        postings = self.lookup(term)
        return len(postings.get(doc_id, []))

    def get_document_frequency(self, term: str) -> int:
        """Get how many documents contain the term."""
        return self.doc_freq.get(term, 0)

    def save(self, path: Path):
        """Save the index to disk."""
        data = {
            "index": dict(self.index),
            "doc_freq": dict(self.doc_freq),
            "doc_lengths": self.doc_lengths,
            "num_docs": self.num_docs,
            "vocabulary": self.vocabulary,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved inverted index to {path}")

    def load(self, path: Path):
        """Load the index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = defaultdict(lambda: defaultdict(list), data["index"])
        self.doc_freq = defaultdict(int, data["doc_freq"])
        self.doc_lengths = data["doc_lengths"]
        self.num_docs = data["num_docs"]
        self.vocabulary = data["vocabulary"]
        logger.info(
            f"Loaded inverted index: {len(self.vocabulary)} terms, "
            f"{self.num_docs} documents"
        )
