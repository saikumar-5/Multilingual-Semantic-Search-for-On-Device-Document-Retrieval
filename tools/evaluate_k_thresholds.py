"""Evaluate adaptive threshold k values and compare metrics."""

from __future__ import annotations

from pathlib import Path
import pickle

from src import config
from src.evaluation.metrics import EvaluationMetrics
from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine
from src.indexer.vector_index import VectorIndex
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.search.hybrid_search import HybridSearch
from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch


def _resolve_reranker_config():
    if not config.USE_MULTILINGUAL_CROSS_ENCODER:
        return config.CROSS_ENCODER_MODEL_NAME, config.CROSS_ENCODER_MODEL_LOCAL_DIR, True

    multilingual_available = (
        config.MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR.exists() or not config.OFFLINE_MODE
    )
    if multilingual_available:
        return (
            config.MULTILINGUAL_CROSS_ENCODER_MODEL_NAME,
            config.MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR,
            False,
        )

    return config.CROSS_ENCODER_MODEL_NAME, config.CROSS_ENCODER_MODEL_LOCAL_DIR, True


def main():
    if not config.DOCUMENT_STORE_PATH.exists():
        raise SystemExit("No index found. Run indexing first.")

    with open(config.DOCUMENT_STORE_PATH, "rb") as f:
        documents = pickle.load(f)

    inv_index = InvertedIndex()
    inv_index.load(config.INVERTED_INDEX_PATH)

    tfidf = TFIDFEngine(inv_index)
    tfidf.load(config.TFIDF_INDEX_PATH)

    vec_index = VectorIndex()
    faiss_path = config.DATA_DIR / "vectors.faiss"
    meta_path = config.DATA_DIR / "vector_meta.pkl"
    if faiss_path.exists() and meta_path.exists():
        vec_index.load(faiss_path, meta_path)

    keyword_search = KeywordSearch(inv_index, tfidf)
    semantic_search = SemanticSearch(vec_index)

    reranker = None
    rerank_english_only = True
    if config.ENABLE_CROSS_ENCODER_RERANK:
        model_name, model_local_dir, rerank_english_only = _resolve_reranker_config()
        reranker = CrossEncoderReranker(
            documents,
            model_name=model_name,
            model_local_dir=model_local_dir,
            top_candidates=config.CROSS_ENCODER_CANDIDATES,
        )

    hybrid = HybridSearch(
        keyword_search,
        semantic_search,
        reranker=reranker,
        rerank_candidates=config.CROSS_ENCODER_CANDIDATES,
        rerank_english_only=rerank_english_only,
    )

    evaluator = EvaluationMetrics()
    test_queries_path = Path(__file__).resolve().parent.parent / "src" / "evaluation" / "test_queries.json"
    evaluator.load_test_queries(test_queries_path)

    ks = [0.3, 0.5, 0.7]
    results = []

    for k in ks:
        config.ADAPTIVE_THRESHOLD_K = k
        metrics = evaluator.evaluate_all(
            lambda q: hybrid.search(q, config.CROSS_ENCODER_TOP_K),
            documents,
        )
        results.append({
            "k": k,
            "P@5": metrics.get("mean_precision@5", 0.0),
            "R@5": metrics.get("mean_recall@5", 0.0),
            "MAP": metrics.get("MAP", 0.0),
            "MRR": metrics.get("MRR", 0.0),
        })

    print("\nAdaptive threshold tuning")
    print("k   |  P@5    R@5    MAP    MRR")
    print("-" * 36)
    for row in results:
        print(
            f"{row['k']:.1f} |  {row['P@5']:.4f} {row['R@5']:.4f} {row['MAP']:.4f} {row['MRR']:.4f}"
        )

    # Pick best by precision@5 with recall guard (within 3% of best recall)
    best_recall = max(r["R@5"] for r in results) if results else 0.0
    recall_floor = max(0.0, best_recall - 0.03)
    candidates = [r for r in results if r["R@5"] >= recall_floor]
    candidates.sort(key=lambda r: (r["P@5"], r["MAP"], r["MRR"]), reverse=True)

    if candidates:
        best = candidates[0]
        print("\nBest k (precision@5 prioritized with recall guard):")
        print(
            f"k={best['k']:.1f} | P@5={best['P@5']:.4f} R@5={best['R@5']:.4f} MAP={best['MAP']:.4f} MRR={best['MRR']:.4f}"
        )


if __name__ == "__main__":
    main()
