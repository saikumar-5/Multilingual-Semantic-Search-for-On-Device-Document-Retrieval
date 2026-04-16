"""
Term-Document Incidence Matrix.

CO1 Alignment: Implements the Term-Document Incidence Matrix concept.

An incidence matrix is a binary matrix where:
- Rows represent terms in the vocabulary
- Columns represent documents
- Cell (i, j) = 1 if term i appears in document j, else 0

This is a foundational IR concept used for Boolean retrieval.
"""

from typing import Dict, List, Set, Tuple
import logging

from src.indexer.inverted_index import InvertedIndex

logger = logging.getLogger(__name__)


class IncidenceMatrix:
    """
    Binary Term-Document Incidence Matrix built from an Inverted Index.

    Useful for:
    - Visualizing which terms appear in which documents
    - Boolean retrieval operations via matrix operations
    - Understanding the document-term relationship
    """

    def __init__(self, inverted_index: InvertedIndex):
        self.inv_index = inverted_index
        # Ordered lists for matrix axes
        self.terms: List[str] = []
        self.doc_ids: List[int] = []
        # The matrix itself: {term: {doc_id: 0 or 1}}
        self.matrix: Dict[str, Dict[int, int]] = {}

    def build(self):
        """
        Build the incidence matrix from the inverted index.

        For each term in the vocabulary, mark 1 for every document
        that contains the term, and 0 otherwise.
        """
        self.terms = sorted(self.inv_index.vocabulary)
        self.doc_ids = sorted(self.inv_index.doc_lengths.keys())

        for term in self.terms:
            self.matrix[term] = {}
            postings = self.inv_index.lookup(term)
            for doc_id in self.doc_ids:
                self.matrix[term][doc_id] = 1 if doc_id in postings else 0

        logger.info(
            f"Built incidence matrix: {len(self.terms)} terms x "
            f"{len(self.doc_ids)} documents"
        )

    def get_row(self, term: str) -> List[int]:
        """Get the incidence vector for a term (1 row of the matrix)."""
        if term not in self.matrix:
            return [0] * len(self.doc_ids)
        return [self.matrix[term].get(doc_id, 0) for doc_id in self.doc_ids]

    def get_column(self, doc_id: int) -> List[int]:
        """Get the term vector for a document (1 column of the matrix)."""
        return [self.matrix.get(term, {}).get(doc_id, 0) for term in self.terms]

    def boolean_and_matrix(self, terms: List[str]) -> Set[int]:
        """
        Boolean AND using incidence matrix.
        Multiply (AND) the incidence vectors of all terms.
        """
        if not terms:
            return set()

        result_vector = self.get_row(terms[0])
        for term in terms[1:]:
            row = self.get_row(term)
            result_vector = [a & b for a, b in zip(result_vector, row)]

        return {
            self.doc_ids[i] for i, val in enumerate(result_vector) if val == 1
        }

    def boolean_or_matrix(self, terms: List[str]) -> Set[int]:
        """
        Boolean OR using incidence matrix.
        Add (OR) the incidence vectors of all terms.
        """
        if not terms:
            return set()

        result_vector = self.get_row(terms[0])
        for term in terms[1:]:
            row = self.get_row(term)
            result_vector = [a | b for a, b in zip(result_vector, row)]

        return {
            self.doc_ids[i] for i, val in enumerate(result_vector) if val == 1
        }

    def get_stats(self) -> dict:
        """Get statistics about the incidence matrix."""
        total_cells = len(self.terms) * len(self.doc_ids)
        ones = sum(
            1
            for term in self.terms
            for doc_id in self.doc_ids
            if self.matrix.get(term, {}).get(doc_id, 0) == 1
        )
        return {
            "num_terms": len(self.terms),
            "num_docs": len(self.doc_ids),
            "total_cells": total_cells,
            "non_zero_cells": ones,
            "sparsity": 1 - (ones / total_cells) if total_cells > 0 else 0,
        }
