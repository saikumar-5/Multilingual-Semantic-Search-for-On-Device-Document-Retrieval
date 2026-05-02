"""
Evaluation Metrics for Information Retrieval system.


Metrics implemented:
- Precision@K: Of the top-K results, how many are relevant?
- Recall@K: Of all relevant docs, how many appeared in top-K?
- F1@K: Harmonic mean of Precision and Recall
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result
- MAP (Mean Average Precision): Mean of average precision across queries
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Compute standard IR evaluation metrics against ground truth.

    Ground truth format (test_queries.json):
    [
        {
            "query": "network security",
            "relevant_docs": ["CSE3714_NETWORK SECURITY_END TERM.pdf", ...]
        },
        ...
    ]
    """

    def __init__(self):
        self.test_queries: List[dict] = []
        self.results_log: List[dict] = []

    def load_test_queries(self, path: Path):
        """Load test queries with ground truth from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.test_queries = json.load(f)
        logger.info(f"Loaded {len(self.test_queries)} test queries")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize filenames for evaluation matching."""
        if not name:
            return ""
        stem = Path(name).stem.lower()
        normalized = stem.replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    def precision_at_k(
        self, retrieved: List[str], relevant: List[str], k: int
    ) -> float:
        """
        Precision@K: fraction of top-K results that are relevant.

        Precision@K = |relevant ∩ retrieved[:K]| / K

        Args:
            retrieved: List of retrieved file names (ordered by rank).
            relevant: List of relevant file names (ground truth).
            k: Number of top results to consider.
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in top_k if doc in relevant_set)
        return hits / k

    def recall_at_k(
        self, retrieved: List[str], relevant: List[str], k: int
    ) -> float:
        """
        Recall@K: fraction of relevant docs found in top-K results.

        Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
        """
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in top_k if doc in relevant_set)
        return hits / len(relevant_set)

    def f1_at_k(
        self, retrieved: List[str], relevant: List[str], k: int
    ) -> float:
        """
        F1@K: Harmonic mean of Precision@K and Recall@K.
        """
        p = self.precision_at_k(retrieved, relevant, k)
        r = self.recall_at_k(retrieved, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def reciprocal_rank(
        self, retrieved: List[str], relevant: List[str]
    ) -> float:
        """
        Reciprocal Rank: 1 / rank of the first relevant result.

        If the first relevant result is at rank 3, RR = 1/3 = 0.333
        """
        relevant_set = set(relevant)
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                return 1.0 / rank
        return 0.0

    def average_precision(
        self, retrieved: List[str], relevant: List[str]
    ) -> float:
        """
        Average Precision: Mean of precision values at each relevant rank.

        AP = (1/|relevant|) * Σ P(k) * rel(k)
        where P(k) is precision at rank k and rel(k) is 1 if doc at k is relevant.
        """
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        hits = 0
        sum_precision = 0.0

        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                hits += 1
                sum_precision += hits / rank

        return sum_precision / len(relevant_set)

    def evaluate_query(
        self,
        query: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = None,
    ) -> dict:
        """
        Evaluate a single query against ground truth.

        Returns:
            Dict with all metrics for this query.
        """
        if k_values is None:
            k_values = [5, 10]

        normalized_retrieved = [self._normalize_name(doc) for doc in retrieved_docs]
        normalized_relevant = [self._normalize_name(doc) for doc in relevant_docs]

        result = {
            "query": query,
            "num_retrieved": len(retrieved_docs),
            "num_relevant": len(relevant_docs),
            "reciprocal_rank": self.reciprocal_rank(
                normalized_retrieved, normalized_relevant
            ),
            "average_precision": self.average_precision(
                normalized_retrieved, normalized_relevant
            ),
        }

        for k in k_values:
            result[f"precision@{k}"] = self.precision_at_k(
                normalized_retrieved, normalized_relevant, k
            )
            result[f"recall@{k}"] = self.recall_at_k(
                normalized_retrieved, normalized_relevant, k
            )
            result[f"f1@{k}"] = self.f1_at_k(
                normalized_retrieved, normalized_relevant, k
            )

        return result

    def evaluate_all(
        self,
        search_function,
        documents: List[dict],
        k_values: List[int] = None,
    ) -> dict:
        """
        Evaluate all test queries and compute aggregate metrics.

        Args:
            search_function: Callable(query) → List[(doc_id, score)]
            documents: Document list for doc_id → filename mapping.
            k_values: List of K values for Precision/Recall@K.

        Returns:
            Dict with per-query and aggregate metrics.
        """
        if k_values is None:
            k_values = [5, 10]

        # Build doc_id → filename lookup
        id_to_name = {}
        for doc in documents:
            id_to_name[doc["doc_id"]] = doc["file_name"]

        all_results = []
        for tq in self.test_queries:
            query = tq["query"]
            relevant = tq["relevant_docs"]

            # Run search
            search_results = search_function(query)
            retrieved = [
                id_to_name.get(doc_id, "") for doc_id, _ in search_results
            ]

            # Evaluate
            metrics = self.evaluate_query(query, retrieved, relevant, k_values)
            all_results.append(metrics)

        # Compute aggregate metrics
        aggregate = self._compute_aggregates(all_results, k_values)
        aggregate["per_query"] = all_results
        aggregate["num_queries"] = len(all_results)

        self.results_log.append(aggregate)
        return aggregate

    def _compute_aggregates(
        self, results: List[dict], k_values: List[int]
    ) -> dict:
        """Compute mean metrics across all queries."""
        if not results:
            return {}

        n = len(results)
        aggregate = {
            "MRR": sum(r["reciprocal_rank"] for r in results) / n,
            "MAP": sum(r["average_precision"] for r in results) / n,
        }

        for k in k_values:
            aggregate[f"mean_precision@{k}"] = (
                sum(r[f"precision@{k}"] for r in results) / n
            )
            aggregate[f"mean_recall@{k}"] = (
                sum(r[f"recall@{k}"] for r in results) / n
            )
            aggregate[f"mean_f1@{k}"] = (
                sum(r[f"f1@{k}"] for r in results) / n
            )

        return aggregate

    def print_report(self, results: dict):
        """Print a formatted evaluation report."""
        print("\n" + "=" * 60)
        print("  EVALUATION REPORT")
        print("=" * 60)
        print(f"  Queries evaluated: {results.get('num_queries', 0)}")
        print(f"  MRR:               {results.get('MRR', 0):.4f}")
        print(f"  MAP:               {results.get('MAP', 0):.4f}")

        for key, val in results.items():
            if key.startswith("mean_"):
                metric_name = key.replace("mean_", "").replace("@", " @ ")
                print(f"  {metric_name:20s} {val:.4f}")

        print("=" * 60)

        # Per-query breakdown
        if "per_query" in results:
            print("\n  Per-Query Breakdown:")
            print(f"  {'Query':40s} {'P@5':>6s} {'R@5':>6s} {'RR':>6s}")
            print("  " + "-" * 58)
            for r in results["per_query"]:
                print(
                    f"  {r['query'][:40]:40s} "
                    f"{r.get('precision@5', 0):6.3f} "
                    f"{r.get('recall@5', 0):6.3f} "
                    f"{r.get('reciprocal_rank', 0):6.3f}"
                )
        print()
