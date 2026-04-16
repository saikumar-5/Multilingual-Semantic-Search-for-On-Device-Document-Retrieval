"""
TF-IDF (Term Frequency - Inverse Document Frequency) Engine.

CO1 Alignment: Implements TF-IDF weighting scheme.

TF-IDF measures how important a word is to a document within a collection:
- TF (Term Frequency): How often a word appears in a document.
  TF(t,d) = count(t in d) / total_terms(d)
- IDF (Inverse Document Frequency): How rare a word is across all documents.
  IDF(t) = log(N / df(t))  where N = total docs, df(t) = docs containing t
- TF-IDF(t,d) = TF(t,d) * IDF(t)

A high TF-IDF means the word is frequent in this document but rare overall,
making it a good distinguishing feature.
"""

import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

from src.indexer.inverted_index import InvertedIndex

logger = logging.getLogger(__name__)


class TFIDFEngine:
    """
    Compute and store TF-IDF weights for all documents.

    Provides cosine similarity scoring between queries and documents
    using the Vector Space Model.

    CO2 Alignment: Vector Space Model with TF-IDF weighting.
    """

    def __init__(self, inverted_index: InvertedIndex):
        self.inv_index = inverted_index
        # TF-IDF weights: {doc_id: {term: tfidf_weight}}
        self.tfidf_weights: Dict[int, Dict[str, float]] = {}
        # IDF values: {term: idf_value}
        self.idf_values: Dict[str, float] = {}
        # Document vector norms for cosine similarity
        self.doc_norms: Dict[int, float] = {}

    def compute(self):
        """
        Compute TF-IDF weights for all terms in all documents.

        Formula:
            TF(t,d) = (count of t in d) / (total terms in d)
            IDF(t)  = log10(N / df(t))
            TF-IDF  = TF * IDF
        """
        n = self.inv_index.num_docs
        if n == 0:
            return

        # Step 1: Compute IDF for each term
        for term in self.inv_index.vocabulary:
            df = self.inv_index.get_document_frequency(term)
            if df > 0:
                self.idf_values[term] = math.log10(n / df)
            else:
                self.idf_values[term] = 0.0

        # Step 2: Compute TF-IDF for each (term, document) pair
        for term, postings in self.inv_index.index.items():
            idf = self.idf_values.get(term, 0.0)
            for doc_id, positions in postings.items():
                tf = len(positions) / max(self.inv_index.doc_lengths.get(doc_id, 1), 1)
                tfidf = tf * idf

                if doc_id not in self.tfidf_weights:
                    self.tfidf_weights[doc_id] = {}
                self.tfidf_weights[doc_id][term] = tfidf

        # Step 3: Compute document vector norms (for cosine similarity)
        for doc_id, weights in self.tfidf_weights.items():
            self.doc_norms[doc_id] = math.sqrt(sum(w ** 2 for w in weights.values()))

        logger.info(
            f"Computed TF-IDF weights for {len(self.tfidf_weights)} documents"
        )

    def get_query_vector(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Compute TF-IDF vector for a query.

        For queries, TF is simply the count of each term in the query.
        IDF values come from the document collection.
        """
        term_counts = defaultdict(int)
        for term in query_terms:
            term_counts[term] += 1

        query_vector = {}
        for term, count in term_counts.items():
            tf = count / len(query_terms)
            idf = self.idf_values.get(term, 0.0)
            query_vector[term] = tf * idf

        return query_vector

    def cosine_similarity(
        self, query_vector: Dict[str, float], doc_id: int
    ) -> float:
        """
        Compute cosine similarity between a query vector and a document vector.

        cosine_sim(q, d) = (q · d) / (|q| * |d|)

        CO2 Alignment: Vector Space Model - cosine similarity scoring.
        """
        doc_weights = self.tfidf_weights.get(doc_id, {})
        if not doc_weights:
            return 0.0

        # Dot product
        dot_product = sum(
            query_vector.get(term, 0.0) * weight
            for term, weight in doc_weights.items()
        )

        # Norms
        query_norm = math.sqrt(sum(w ** 2 for w in query_vector.values()))
        doc_norm = self.doc_norms.get(doc_id, 0.0)

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    def rank_documents(
        self, query_terms: List[str], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Rank all documents by TF-IDF cosine similarity to the query.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.

        CO2/CO3 Alignment: Rank-based retrieval.
        """
        query_vector = self.get_query_vector(query_terms)

        scores = []
        for doc_id in self.tfidf_weights:
            score = self.cosine_similarity(query_vector, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_top_terms(self, doc_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the top-N most important terms in a document by TF-IDF weight."""
        weights = self.tfidf_weights.get(doc_id, {})
        sorted_terms = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:top_n]

    def save(self, path: Path):
        """Save TF-IDF data to disk."""
        data = {
            "tfidf_weights": self.tfidf_weights,
            "idf_values": self.idf_values,
            "doc_norms": self.doc_norms,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved TF-IDF index to {path}")

    def load(self, path: Path):
        """Load TF-IDF data from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.tfidf_weights = data["tfidf_weights"]
        self.idf_values = data["idf_values"]
        self.doc_norms = data["doc_norms"]
        logger.info(f"Loaded TF-IDF index with {len(self.tfidf_weights)} documents")
