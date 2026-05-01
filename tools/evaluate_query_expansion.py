"""Evaluate query expansion impact for short queries in hybrid search.

Compares BEFORE (expansion disabled) vs AFTER (expansion enabled) on:
- Precision@5
- Recall@10
- MAP
- MRR

Usage:
    python -m tools.evaluate_query_expansion
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from src.config import (
    DOCUMENT_STORE_PATH,
    INVERTED_INDEX_PATH,
    TFIDF_INDEX_PATH,
    DATA_DIR,
    ENABLE_CROSS_ENCODER_RERANK,
    CROSS_ENCODER_CANDIDATES,
    CROSS_ENCODER_TOP_K,
    DEFAULT_TOP_K,
)
from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine
from src.indexer.vector_index import VectorIndex
from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch
from src.search.hybrid_search import HybridSearch
from src.search.query_processor import QueryProcessor
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.evaluation.metrics import EvaluationMetrics
import src.search.query_processor as query_processor_module


def build_engines(documents: List[dict], use_reranker: bool):
    inv_index = InvertedIndex()
    inv_index.load(INVERTED_INDEX_PATH)

    tfidf = TFIDFEngine(inv_index)
    tfidf.load(TFIDF_INDEX_PATH)

    vec_index = VectorIndex()
    faiss_path = DATA_DIR / "vectors.faiss"
    meta_path = DATA_DIR / "vector_meta.pkl"
    vec_index.load(faiss_path, meta_path)

    kw_search = KeywordSearch(inv_index, tfidf)
    sem_search = SemanticSearch(vec_index)

    reranker = None
    if ENABLE_CROSS_ENCODER_RERANK and use_reranker:
        reranker = CrossEncoderReranker(
            documents,
            top_candidates=CROSS_ENCODER_CANDIDATES,
        )

    hybrid = HybridSearch(
        kw_search,
        sem_search,
        reranker=reranker,
        rerank_candidates=CROSS_ENCODER_CANDIDATES,
    )

    return hybrid


def aggregate_metrics(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {
            "precision@5": 0.0,
            "recall@10": 0.0,
            "MAP": 0.0,
            "MRR": 0.0,
        }

    return {
        "precision@5": mean(r["precision@5"] for r in rows),
        "recall@10": mean(r["recall@10"] for r in rows),
        "MAP": mean(r["average_precision"] for r in rows),
        "MRR": mean(r["reciprocal_rank"] for r in rows),
    }


def format_delta(after: float, before: float) -> str:
    d = after - before
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def is_short_query(qp: QueryProcessor, query: str, max_tokens: int = 3) -> bool:
    terms = qp.preprocessor_filtered.preprocess(query)
    return 0 < len(terms) <= max_tokens


def run_eval(
    hybrid: HybridSearch,
    evaluator: EvaluationMetrics,
    short_queries: List[dict],
    id_to_name: Dict[int, str],
    expansion_enabled: bool,
) -> Tuple[Dict[str, float], List[dict]]:
    query_processor_module.ENABLE_QUERY_EXPANSION = expansion_enabled

    per_query = []
    top_k = CROSS_ENCODER_TOP_K if ENABLE_CROSS_ENCODER_RERANK else DEFAULT_TOP_K

    for q in short_queries:
        query = q["query"]
        relevant = q["relevant_docs"]

        raw = hybrid.search(query, top_k=top_k)
        retrieved = [id_to_name.get(doc_id, "") for doc_id, _ in raw]

        metrics = evaluator.evaluate_query(
            query=query,
            retrieved_docs=retrieved,
            relevant_docs=relevant,
            k_values=[5, 10],
        )
        per_query.append(metrics)

    return aggregate_metrics(per_query), per_query


def print_report(
    before: Dict[str, float],
    after: Dict[str, float],
    before_rows: List[dict],
    after_rows: List[dict],
):
    print("\nQuery Expansion Impact (Short Queries <= 3 tokens)")
    print("=" * 72)
    print("| Metric      | Before | After  | Delta   |")
    print("|-------------|--------|--------|---------|")
    for metric in ["precision@5", "recall@10", "MAP", "MRR"]:
        print(
            f"| {metric:<11} | {before[metric]:.4f} | {after[metric]:.4f} | {format_delta(after[metric], before[metric]):>7} |"
        )

    print("\nObservations")
    print("-" * 72)

    by_query_before = {r["query"]: r for r in before_rows}
    by_query_after = {r["query"]: r for r in after_rows}

    improved = []
    hurt = []
    unchanged = []

    for query in by_query_before:
        b = by_query_before[query]
        a = by_query_after[query]

        # Primary impact signal: AP, then RR as tie-breaker
        d_ap = a["average_precision"] - b["average_precision"]
        d_rr = a["reciprocal_rank"] - b["reciprocal_rank"]

        if d_ap > 1e-9 or (abs(d_ap) <= 1e-9 and d_rr > 1e-9):
            improved.append((query, d_ap, d_rr))
        elif d_ap < -1e-9 or (abs(d_ap) <= 1e-9 and d_rr < -1e-9):
            hurt.append((query, d_ap, d_rr))
        else:
            unchanged.append((query, d_ap, d_rr))

    print(f"- Improved queries: {len(improved)}")
    print(f"- Hurt queries: {len(hurt)}")
    print(f"- Unchanged queries: {len(unchanged)}")

    if improved:
        improved_sorted = sorted(improved, key=lambda x: x[1], reverse=True)[:5]
        print("- Top improvements (query, ΔAP, ΔRR):")
        for q, d_ap, d_rr in improved_sorted:
            print(f"  - {q} | {d_ap:+.4f} | {d_rr:+.4f}")

    if hurt:
        hurt_sorted = sorted(hurt, key=lambda x: x[1])[:5]
        print("- Top degradations (query, ΔAP, ΔRR):")
        for q, d_ap, d_rr in hurt_sorted:
            print(f"  - {q} | {d_ap:+.4f} | {d_rr:+.4f}")

    print("\nSuggestions (Expansion Weight Tuning)")
    print("-" * 72)

    p5_delta = after["precision@5"] - before["precision@5"]
    r10_delta = after["recall@10"] - before["recall@10"]

    if p5_delta < 0 and r10_delta > 0:
        print("- Recall improved but precision dropped: lower expanded-term weight (e.g., 0.50 -> 0.35).")
        print("- Keep original-term weight at 1.0 and cap expansion fan-out strictly at 1-2 terms.")
    elif p5_delta > 0 and r10_delta <= 0:
        print("- Precision improved but recall did not: slightly raise expanded-term weight (e.g., 0.50 -> 0.60).")
        print("- Add missing synonyms for low-recall short queries to widen candidate coverage.")
    elif p5_delta > 0 and r10_delta > 0:
        print("- Both precision and recall improved: keep current weights as baseline.")
        print("- Try small A/B around expanded-term weight (0.45, 0.55) to confirm stability.")
    else:
        print("- No aggregate gain: tighten synonym quality and remove ambiguous terms.")
        print("- Reduce expansion for noisy terms or apply language-specific weights.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate query expansion impact")
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Include cross-encoder reranking during evaluation (slower)",
    )
    args = parser.parse_args()

    if not DOCUMENT_STORE_PATH.exists():
        raise FileNotFoundError("No index found. Run indexing before evaluation.")

    with open(DOCUMENT_STORE_PATH, "rb") as f:
        documents = pickle.load(f)

    id_to_name = {doc["doc_id"]: doc["file_name"] for doc in documents}

    evaluator = EvaluationMetrics()
    test_queries_path = Path(__file__).resolve().parent.parent / "src" / "evaluation" / "test_queries.json"
    evaluator.load_test_queries(test_queries_path)

    qp = QueryProcessor()
    short_queries = [
        q for q in evaluator.test_queries if is_short_query(qp, q.get("query", ""), max_tokens=3)
    ]

    if not short_queries:
        raise RuntimeError("No short queries (<=3 tokens) found in test set.")

    hybrid = build_engines(documents, use_reranker=args.use_reranker)

    before, before_rows = run_eval(
        hybrid,
        evaluator,
        short_queries,
        id_to_name,
        expansion_enabled=False,
    )

    after, after_rows = run_eval(
        hybrid,
        evaluator,
        short_queries,
        id_to_name,
        expansion_enabled=True,
    )

    print(f"Evaluated queries: {len(short_queries)}")
    print_report(before, after, before_rows, after_rows)


if __name__ == "__main__":
    main()
