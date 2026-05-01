"""Benchmark FAISS Flat vs HNSW for this project.

Measures:
- Average query latency (ms)
- Throughput (queries/sec)
- Approximate recall@k against Flat baseline

Usage:
    python -m tools.benchmark_hnsw --top-k 10
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import faiss
import numpy as np

from src.config import DATA_DIR


def load_index_to_doc(metadata_path: Path) -> List[Tuple[int, int]]:
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata["index_to_doc"]


def extract_all_vectors(index) -> np.ndarray:
    """Extract all vectors from a FAISS index for benchmark reproducibility."""
    n = int(index.ntotal)
    if n == 0:
        return np.zeros((0, 0), dtype="float32")

    # Fast path for indexes exposing reconstruct_n.
    try:
        vecs = index.reconstruct_n(0, n)
        return np.asarray(vecs, dtype="float32")
    except Exception:
        pass

    # Fallback path.
    first = np.asarray(index.reconstruct(0), dtype="float32")
    d = int(first.shape[0])
    out = np.empty((n, d), dtype="float32")
    out[0] = first
    for i in range(1, n):
        out[i] = np.asarray(index.reconstruct(i), dtype="float32")
    return out


def sample_query_vectors(vectors: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if vectors.size == 0:
        return vectors

    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    take = min(sample_size, n)
    idx = rng.choice(n, size=take, replace=False)

    # Add tiny perturbation so queries are realistic but still in-distribution.
    q = vectors[idx].copy()
    noise = rng.normal(loc=0.0, scale=0.001, size=q.shape).astype("float32")
    q = q + noise
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (q / norms).astype("float32")


def aggregate_doc_topk(
    scores: np.ndarray,
    indices: np.ndarray,
    index_to_doc: Sequence[Tuple[int, int]],
    top_k: int,
) -> List[int]:
    doc_scores: Dict[int, float] = {}
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        doc_id, _ = index_to_doc[int(idx)]
        prior = doc_scores.get(doc_id)
        if prior is None or float(score) > prior:
            doc_scores[doc_id] = float(score)

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]


def recall_at_k(pred: Sequence[int], gold: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_set = set(gold[:k])
    if not gold_set:
        return 0.0
    pred_set = set(pred[:k])
    return len(pred_set & gold_set) / float(min(k, len(gold_set)))


def benchmark_index(
    index,
    query_vectors: np.ndarray,
    index_to_doc: Sequence[Tuple[int, int]],
    exact_doc_rankings: Sequence[List[int]],
    top_k: int,
    search_multiplier: int,
) -> dict:
    n_search = min(top_k * search_multiplier, index.ntotal)

    # Warm-up
    for i in range(min(3, len(query_vectors))):
        _ = index.search(query_vectors[i : i + 1], n_search)

    t0 = time.perf_counter()
    recalls = []
    for q_idx, qv in enumerate(query_vectors):
        scores, indices = index.search(qv.reshape(1, -1), n_search)
        pred_doc_ids = aggregate_doc_topk(scores[0], indices[0], index_to_doc, top_k)
        recalls.append(recall_at_k(pred_doc_ids, exact_doc_rankings[q_idx], top_k))
    elapsed = time.perf_counter() - t0

    q_count = len(query_vectors)
    avg_ms = (elapsed / q_count) * 1000.0 if q_count else 0.0
    qps = (q_count / elapsed) if elapsed > 0 else 0.0

    return {
        "avg_latency_ms": avg_ms,
        "qps": qps,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS Flat vs HNSW")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--search-multiplier", type=int, default=5)
    parser.add_argument("--hnsw-m", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--ef-search", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--query-sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    faiss_path = DATA_DIR / "vectors.faiss"
    metadata_path = DATA_DIR / "vector_meta.pkl"
    if not faiss_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing vector index files: {faiss_path} and/or {metadata_path}. Run indexing first."
        )

    source_index = faiss.read_index(str(faiss_path))
    index_to_doc = load_index_to_doc(metadata_path)
    vectors = extract_all_vectors(source_index)
    if vectors.shape[0] != len(index_to_doc):
        raise RuntimeError(
            "Vector count does not match metadata index_to_doc length. Rebuild index and retry."
        )

    query_vectors = sample_query_vectors(vectors, args.query_sample, args.seed)

    d = vectors.shape[1]
    flat = faiss.IndexFlatIP(d)
    flat.add(vectors)

    n_search = min(args.top_k * args.search_multiplier, flat.ntotal)
    exact_doc_rankings = []
    for qv in query_vectors:
        s, i = flat.search(qv.reshape(1, -1), n_search)
        exact_doc_rankings.append(
            aggregate_doc_topk(s[0], i[0], index_to_doc, args.top_k)
        )

    print("\nFAISS Benchmark (doc-level)\n" + "=" * 56)
    print(f"Vectors:   {len(vectors)}")
    print(f"Queries:   {len(query_vectors)}")
    print(f"top_k:     {args.top_k}")

    flat_stats = benchmark_index(
        flat,
        query_vectors,
        index_to_doc,
        exact_doc_rankings,
        args.top_k,
        args.search_multiplier,
    )

    print("\nFlat (exact baseline)")
    print(f"  avg_latency_ms: {flat_stats['avg_latency_ms']:.3f}")
    print(f"  qps:            {flat_stats['qps']:.2f}")
    print(f"  recall@{args.top_k}:      {flat_stats['recall_at_k']:.4f}")

    print("\nHNSW configs")
    print("  M  efSearch  efConstr   avg_latency_ms   qps      recall@k")

    for m in args.hnsw_m:
        for ef_search in args.ef_search:
            hnsw = faiss.index_factory(d, f"HNSW{m},Flat", faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = max(16, int(args.ef_construction))
            hnsw.hnsw.efSearch = max(8, int(ef_search))
            hnsw.add(vectors)

            stats = benchmark_index(
                hnsw,
                query_vectors,
                index_to_doc,
                exact_doc_rankings,
                args.top_k,
                args.search_multiplier,
            )

            print(
                f"  {m:2d}  {ef_search:8d}  {args.ef_construction:8d}"
                f"   {stats['avg_latency_ms']:14.3f}   {stats['qps']:7.2f}   {stats['recall_at_k']:.4f}"
            )

    print("\nNotes:")
    print("- recall@k is against Flat exact results at doc level (max chunk score per doc).")
    print("- higher M and efSearch usually improve recall but increase latency/memory.")


if __name__ == "__main__":
    main()
