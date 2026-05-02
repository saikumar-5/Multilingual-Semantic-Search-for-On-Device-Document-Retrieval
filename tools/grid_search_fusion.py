"""
Two-stage fast tuning for fusion weights across keyword, semantic, and reranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import time

import pickle

from src.config import (
    CROSS_ENCODER_CANDIDATES,
    CROSS_ENCODER_TOP_K,
    DATA_DIR,
    DEFAULT_TOP_K,
    DOCUMENT_STORE_PATH,
    INVERTED_INDEX_PATH,
    TFIDF_INDEX_PATH,
    OFFLINE_MODE,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_MODEL_LOCAL_DIR,
    MULTILINGUAL_CROSS_ENCODER_MODEL_NAME,
    MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR,
    USE_MULTILINGUAL_CROSS_ENCODER,
)
from src.evaluation.metrics import EvaluationMetrics
from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine
from src.indexer.vector_index import VectorIndex
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch


@dataclass(frozen=True)
class WeightConfig:
    keyword: float
    semantic: float
    reranker: float


def _resolve_reranker_model() -> Tuple[str, Path]:
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


def _min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_v = min(values)
    max_v = max(values)
    if max_v - min_v <= 1e-12:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: (val - min_v) / (max_v - min_v) for doc_id, val in scores.items()}


def _combine_scores(
    kw_scores: Dict[int, float],
    sem_scores: Dict[int, float],
    ce_scores: Dict[int, float],
    weights: WeightConfig,
) -> List[Tuple[int, float]]:
    all_ids = set(kw_scores) | set(sem_scores) | set(ce_scores)
    results = []
    for doc_id in all_ids:
        score = (
            weights.keyword * kw_scores.get(doc_id, 0.0)
            + weights.semantic * sem_scores.get(doc_id, 0.0)
            + weights.reranker * ce_scores.get(doc_id, 0.0)
        )
        results.append((doc_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _build_candidate_list(
    kw_results: List[Tuple[int, float]],
    sem_results: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    candidates = []
    for doc_id, score in kw_results:
        candidates.append((doc_id, score))
    for doc_id, score in sem_results:
        candidates.append((doc_id, score))

    candidate_map: Dict[int, float] = {}
    for doc_id, score in candidates:
        candidate_map[doc_id] = max(candidate_map.get(doc_id, 0.0), score)

    return sorted(candidate_map.items(), key=lambda x: x[1], reverse=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Two-stage fast fusion tuning.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print progress every N combinations.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for embeddings and reranking if available.",
    )
    return parser.parse_args()


def _score_query(
    kw_scores: Dict[int, float],
    sem_scores: Dict[int, float],
    ce_scores: Dict[int, float],
    weights: WeightConfig,
    top_k: int,
) -> List[int]:
    combined = _combine_scores(kw_scores, sem_scores, ce_scores, weights)
    return [doc_id for doc_id, _ in combined[:top_k]]


def main():
    args = _parse_args()
    if not DOCUMENT_STORE_PATH.exists():
        raise SystemExit("No index found. Run indexing first.")

    with open(DOCUMENT_STORE_PATH, "rb") as f:
        documents = pickle.load(f)

    inv_index = InvertedIndex()
    inv_index.load(INVERTED_INDEX_PATH)

    tfidf = TFIDFEngine(inv_index)
    tfidf.load(TFIDF_INDEX_PATH)

    device = "cpu"
    if args.use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                print("GPU requested but CUDA not available; using CPU.")
        except Exception as exc:
            print(f"GPU requested but torch check failed; using CPU: {exc}")

    vec_index = VectorIndex(device=device)
    faiss_path = DATA_DIR / "vectors.faiss"
    meta_path = DATA_DIR / "vector_meta.pkl"
    if faiss_path.exists() and meta_path.exists():
        vec_index.load(faiss_path, meta_path)

    keyword_search = KeywordSearch(inv_index, tfidf)
    semantic_search = SemanticSearch(vec_index)

    reranker = None
    if CROSS_ENCODER_CANDIDATES > 0:
        model_name, model_local_dir = _resolve_reranker_model()
        reranker = CrossEncoderReranker(
            documents,
            model_name=model_name,
            model_local_dir=model_local_dir,
            top_candidates=CROSS_ENCODER_CANDIDATES,
            device=device,
        )

    evaluator = EvaluationMetrics()
    test_queries_path = (
        Path(__file__).resolve().parent.parent / "src" / "evaluation" / "test_queries.json"
    )
    evaluator.load_test_queries(test_queries_path)

    top_k = CROSS_ENCODER_TOP_K if CROSS_ENCODER_TOP_K else DEFAULT_TOP_K
    retrieval_k = max(top_k * 5, CROSS_ENCODER_CANDIDATES)

    # Stage 1: coarse keyword/semantic weights without reranker.
    stage1_weights = [
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
    ]
    stage2_reranker_weights = [0.2, 0.3, 0.4]

    # Build doc_id -> filename lookup.
    id_to_name = {doc["doc_id"]: doc["file_name"] for doc in documents}

    # Cache keyword/semantic results per query.
    kw_cache: Dict[str, Dict[int, float]] = {}
    sem_cache: Dict[str, Dict[int, float]] = {}
    candidate_cache: Dict[str, List[Tuple[int, float]]] = {}
    for tq in evaluator.test_queries:
        query = tq["query"]
        kw_results = keyword_search.search(query, retrieval_k)
        sem_results = semantic_search.search(query, retrieval_k)

        kw_cache[query] = _min_max_normalize({doc_id: score for doc_id, score in kw_results})
        sem_cache[query] = _min_max_normalize({doc_id: score for doc_id, score in sem_results})
        candidate_cache[query] = _build_candidate_list(kw_results, sem_results)

    def _evaluate_weights(weights: WeightConfig, ce_cache: Dict[str, Dict[int, float]]):
        per_query = []
        for tq in evaluator.test_queries:
            query = tq["query"]
            relevant = tq["relevant_docs"]
            retrieved_ids = _score_query(
                kw_cache[query],
                sem_cache[query],
                ce_cache.get(query, {}),
                weights,
                top_k,
            )
            retrieved = [id_to_name.get(doc_id, "") for doc_id in retrieved_ids]
            per_query.append(
                evaluator.evaluate_query(query, retrieved, relevant, k_values=[5, 10])
            )

        aggregate = evaluator._compute_aggregates(per_query, [5, 10])
        return {
            "weights": weights,
            "MRR": aggregate.get("MRR", 0.0),
            "MAP": aggregate.get("MAP", 0.0),
            "P@5": aggregate.get("mean_precision@5", 0.0),
            "R@5": aggregate.get("mean_recall@5", 0.0),
        }

    # Stage 1 evaluation (no reranker scores)
    stage1_start = time.time()
    stage1_results = []
    for i, (kw, sem) in enumerate(stage1_weights, 1):
        weights = WeightConfig(keyword=kw, semantic=sem, reranker=0.0)
        stage1_results.append(_evaluate_weights(weights, {}))
        if args.log_every and i % args.log_every == 0:
            print(f"Stage 1: evaluated {i}/{len(stage1_weights)}")
    stage1_time = time.time() - stage1_start

    stage1_results.sort(key=lambda r: (r["MAP"] + r["MRR"]) / 2.0, reverse=True)
    top_stage1 = stage1_results[:3]

    # Stage 2: precompute reranker scores once per query.
    stage2_start = time.time()
    ce_cache: Dict[str, Dict[int, float]] = {}
    if reranker is not None:
        for i, tq in enumerate(evaluator.test_queries, 1):
            query = tq["query"]
            candidate_list = candidate_cache[query]
            reranked = reranker.rerank(query, candidate_list, retrieval_k)
            ce_cache[query] = _min_max_normalize({doc_id: score for doc_id, score in reranked})
            if args.log_every and i % args.log_every == 0:
                print(f"Stage 2: reranked {i}/{len(evaluator.test_queries)} queries")

    stage2_results = []
    for base in top_stage1:
        base_w = base["weights"]
        for ce in stage2_reranker_weights:
            weights = WeightConfig(keyword=base_w.keyword, semantic=base_w.semantic, reranker=ce)
            stage2_results.append(_evaluate_weights(weights, ce_cache))

    stage2_time = time.time() - stage2_start

    stage2_results.sort(key=lambda r: (r["MAP"] + r["MRR"]) / 2.0, reverse=True)
    best_map = max(stage2_results, key=lambda r: r["MAP"], default=None)
    best_mrr = max(stage2_results, key=lambda r: r["MRR"], default=None)
    best_combined = stage2_results[0] if stage2_results else None

    total_time = stage1_time + stage2_time

    print("\nStage 1 (coarse, no reranker) results:")
    print("kw  sem  |  MRR    MAP    P@5   R@5")
    print("-" * 40)
    for row in stage1_results:
        w = row["weights"]
        print(
            f"{w.keyword:>3.2f} {w.semantic:>4.2f}  |  "
            f"{row['MRR']:.4f} {row['MAP']:.4f} {row['P@5']:.4f} {row['R@5']:.4f}"
        )

    print("\nStage 2 (reranker sweep) results:")
    print("kw  sem  ce  |  MRR    MAP    P@5   R@5")
    print("-" * 44)
    for row in stage2_results:
        w = row["weights"]
        print(
            f"{w.keyword:>3.2f} {w.semantic:>4.2f} {w.reranker:>4.2f}  |  "
            f"{row['MRR']:.4f} {row['MAP']:.4f} {row['P@5']:.4f} {row['R@5']:.4f}"
        )

    print(f"\nTime taken: {total_time/60:.2f} minutes")

    print("\nBest by MAP:")
    if best_map:
        w = best_map["weights"]
        print(
            f"kw={w.keyword:.2f}, sem={w.semantic:.2f}, ce={w.reranker:.2f}  "
            f"MAP={best_map['MAP']:.4f}, MRR={best_map['MRR']:.4f}"
        )

    print("\nBest by MRR:")
    if best_mrr:
        w = best_mrr["weights"]
        print(
            f"kw={w.keyword:.2f}, sem={w.semantic:.2f}, ce={w.reranker:.2f}  "
            f"MRR={best_mrr['MRR']:.4f}, MAP={best_mrr['MAP']:.4f}"
        )

    print("\nRecommended (avg of MAP and MRR):")
    if best_combined:
        w = best_combined["weights"]
        print(
            f"kw={w.keyword:.2f}, sem={w.semantic:.2f}, ce={w.reranker:.2f}  "
            f"MRR={best_combined['MRR']:.4f}, MAP={best_combined['MAP']:.4f}"
        )


if __name__ == "__main__":
    main()
