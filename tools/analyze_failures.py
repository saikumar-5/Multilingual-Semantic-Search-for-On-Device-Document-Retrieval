"""
Analyze per-query failures and their impact on hybrid metrics.
"""

from __future__ import annotations

from pathlib import Path
import pickle

from src.config import (
    DATA_DIR,
    DOCUMENT_STORE_PATH,
    INVERTED_INDEX_PATH,
    TFIDF_INDEX_PATH,
    CROSS_ENCODER_CANDIDATES,
    CROSS_ENCODER_TOP_K,
    DEFAULT_TOP_K,
    ENABLE_CROSS_ENCODER_RERANK,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_MODEL_LOCAL_DIR,
    MULTILINGUAL_CROSS_ENCODER_MODEL_NAME,
    MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR,
    USE_MULTILINGUAL_CROSS_ENCODER,
    OFFLINE_MODE,
)
from src.evaluation.metrics import EvaluationMetrics
from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine
from src.indexer.vector_index import VectorIndex
from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch
from src.search.hybrid_search import HybridSearch
from src.search.cross_encoder_reranker import CrossEncoderReranker


def _resolve_reranker_config():
    if not USE_MULTILINGUAL_CROSS_ENCODER:
        return CROSS_ENCODER_MODEL_NAME, CROSS_ENCODER_MODEL_LOCAL_DIR

    multilingual_available = (
        MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR.exists() or not OFFLINE_MODE
    )
    if multilingual_available:
        return (
            MULTILINGUAL_CROSS_ENCODER_MODEL_NAME,
            MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR,
        )

    return CROSS_ENCODER_MODEL_NAME, CROSS_ENCODER_MODEL_LOCAL_DIR


def main():
    if not DOCUMENT_STORE_PATH.exists():
        raise SystemExit("No index found. Run indexing first.")

    with open(DOCUMENT_STORE_PATH, "rb") as f:
        documents = pickle.load(f)

    inv_index = InvertedIndex()
    inv_index.load(INVERTED_INDEX_PATH)

    tfidf = TFIDFEngine(inv_index)
    tfidf.load(TFIDF_INDEX_PATH)

    vec_index = VectorIndex()
    faiss_path = DATA_DIR / "vectors.faiss"
    meta_path = DATA_DIR / "vector_meta.pkl"
    if faiss_path.exists() and meta_path.exists():
        vec_index.load(faiss_path, meta_path)

    kw_search = KeywordSearch(inv_index, tfidf)
    sem_search = SemanticSearch(vec_index)

    reranker = None
    rerank_english_only = True
    if ENABLE_CROSS_ENCODER_RERANK:
        model_name, model_local_dir = _resolve_reranker_config()
        reranker = CrossEncoderReranker(
            documents,
            model_name=model_name,
            model_local_dir=model_local_dir,
            top_candidates=CROSS_ENCODER_CANDIDATES,
        )
        rerank_english_only = False

    hybrid = HybridSearch(
        kw_search,
        sem_search,
        reranker=reranker,
        rerank_candidates=CROSS_ENCODER_CANDIDATES,
        rerank_english_only=rerank_english_only,
    )

    evaluator = EvaluationMetrics()
    test_queries_path = Path(__file__).resolve().parent.parent / "src" / "evaluation" / "test_queries.json"
    evaluator.load_test_queries(test_queries_path)

    top_k = CROSS_ENCODER_TOP_K if ENABLE_CROSS_ENCODER_RERANK else DEFAULT_TOP_K

    id_to_name = {doc["doc_id"]: doc["file_name"] for doc in documents}

    per_query = []
    for tq in evaluator.test_queries:
        query = tq["query"]
        relevant = tq["relevant_docs"]
        results = hybrid.search(query, top_k)
        retrieved = [id_to_name.get(doc_id, "") for doc_id, _ in results]
        metrics = evaluator.evaluate_query(query, retrieved, relevant, k_values=[5, 10])
        per_query.append(metrics)

    failing = [q for q in per_query if q.get("reciprocal_rank", 0.0) == 0.0]
    zero_p5 = [q for q in per_query if q.get("precision@5", 0.0) == 0.0]

    print("\nHybrid per-query analysis")
    print("Queries with RR=0 (no relevant result):")
    for q in failing:
        print(f"- {q['query']}")

    print("\nQueries with P@5=0:")
    for q in zero_p5:
        print(f"- {q['query']}")

    # Estimate impact: remove failing queries and recompute aggregates
    remaining = [q for q in per_query if q.get("reciprocal_rank", 0.0) > 0.0]
    if remaining:
        agg = evaluator._compute_aggregates(remaining, [5, 10])
        print("\nAggregate without RR=0 queries:")
        print(f"MRR: {agg.get('MRR', 0):.4f}")
        print(f"MAP: {agg.get('MAP', 0):.4f}")
        print(f"P@5: {agg.get('mean_precision@5', 0):.4f}")
        print(f"R@5: {agg.get('mean_recall@5', 0):.4f}")


if __name__ == "__main__":
    main()
