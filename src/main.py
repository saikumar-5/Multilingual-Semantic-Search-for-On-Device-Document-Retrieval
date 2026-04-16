"""
Main entry point for the DocSearch application.

Usage:
    GUI Mode (default):
        python -m src.main

    CLI Mode:
        python -m src.main --index <directory>       Index a directory
        python -m src.main --search <query>           Search the index
        python -m src.main --evaluate                 Run evaluation
        python -m src.main --mode keyword|semantic|hybrid
"""

import argparse
import sys
import pickle
import logging
from pathlib import Path

from src.config import (
    INVERTED_INDEX_PATH,
    TFIDF_INDEX_PATH,
    FAISS_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    DATA_DIR,
    DEFAULT_TOP_K,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_index(directory: str):
    """Index a directory from the command line."""
    from src.ingestion.file_router import FileRouter
    from src.indexer.inverted_index import InvertedIndex
    from src.indexer.tfidf import TFIDFEngine
    from src.indexer.incidence_matrix import IncidenceMatrix
    from src.indexer.vector_index import VectorIndex
    from src.search.wildcard import WildcardSearch

    print(f"\n  Indexing directory: {directory}")
    print("  " + "=" * 50)

    # Step 1: Parse files
    print("\n  [1/5] Scanning and parsing files...")
    router = FileRouter()
    documents = router.process_directory(
        directory,
        progress_callback=lambda cur, total, name: print(
            f"    ({cur}/{total}) {name}", end="\r"
        ),
    )
    print(f"\n  Parsed {len(documents)} documents successfully.")

    # Re-assign sequential doc_ids
    for i, doc in enumerate(documents):
        doc["doc_id"] = i

    # Step 2: Build inverted index
    print("\n  [2/5] Building inverted index...")
    inv_index = InvertedIndex()
    inv_index.build(documents)
    print(f"  Vocabulary size: {len(inv_index.vocabulary)} terms")
    inv_index.save(INVERTED_INDEX_PATH)

    # Step 3: Compute TF-IDF
    print("\n  [3/5] Computing TF-IDF weights...")
    tfidf = TFIDFEngine(inv_index)
    tfidf.compute()
    tfidf.save(TFIDF_INDEX_PATH)

    # Step 4: Build incidence matrix (for CO1 demo)
    print("\n  [4/5] Building incidence matrix...")
    inc_matrix = IncidenceMatrix(inv_index)
    inc_matrix.build()
    stats = inc_matrix.get_stats()
    print(f"  Matrix: {stats['num_terms']} terms x {stats['num_docs']} docs")
    print(f"  Sparsity: {stats['sparsity']:.2%}")

    # Step 5: Build vector index
    print("\n  [5/5] Building semantic vector index...")
    vec_index = VectorIndex()
    vec_index.build(
        documents,
        progress_callback=lambda cur, total, status: print(
            f"    {status} ({cur}/{total})", end="\r"
        ),
    )
    vec_index.save(DATA_DIR / "vectors.faiss", DATA_DIR / "vector_meta.pkl")

    # Save documents
    with open(DOCUMENT_STORE_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"\n  Indexing complete! {len(documents)} documents indexed.")
    print(f"  Index saved to: {DATA_DIR}")
    print()


def cmd_search(query: str, mode: str = "hybrid", top_k: int = DEFAULT_TOP_K):
    """Search from the command line."""
    from src.indexer.inverted_index import InvertedIndex
    from src.indexer.tfidf import TFIDFEngine
    from src.indexer.vector_index import VectorIndex
    from src.search.keyword_search import KeywordSearch
    from src.search.semantic_search import SemanticSearch
    from src.search.hybrid_search import HybridSearch
    from src.search.wildcard import WildcardSearch
    from src.search.ranker import Ranker
    from src.search.query_processor import QueryProcessor

    # Load index
    if not DOCUMENT_STORE_PATH.exists():
        print("\n  No index found. Run --index <directory> first.\n")
        return

    print(f"\n  Loading index...")
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

    # Build search engines
    kw_search = KeywordSearch(inv_index, tfidf)
    sem_search = SemanticSearch(vec_index)
    hybrid = HybridSearch(kw_search, sem_search)
    wildcard = WildcardSearch(inv_index, tfidf)
    wildcard.build()
    ranker = Ranker(documents)
    qp = QueryProcessor()

    # Run search
    parsed = qp.parse(query)

    if parsed["type"] == "wildcard":
        raw_results = wildcard.search(parsed["raw"], top_k)
    elif mode == "keyword":
        raw_results = kw_search.search(query, top_k)
    elif mode == "semantic":
        raw_results = sem_search.search(query, top_k)
    else:
        raw_results = hybrid.search(query, top_k)

    results = ranker.format_results(raw_results, query)

    # Display results
    print(f"\n  Search: \"{query}\" (mode: {mode})")
    print(f"  Language detected: {parsed['language']}")
    print("  " + "=" * 60)

    if not results:
        print("  No results found.\n")
        return

    for r in results:
        print(
            f"  #{r['rank']:2d} [{r['score']:.3f}] [{r['language']:>7s}] "
            f"[{r['file_type'].upper():>5s}] {r['file_name']}"
        )
        if r["snippet"]:
            snippet = r["snippet"][:120].replace("\n", " ")
            print(f"       {snippet}")
        print()


def cmd_evaluate():
    """Run evaluation from the command line."""
    from src.indexer.inverted_index import InvertedIndex
    from src.indexer.tfidf import TFIDFEngine
    from src.indexer.vector_index import VectorIndex
    from src.search.keyword_search import KeywordSearch
    from src.search.semantic_search import SemanticSearch
    from src.search.hybrid_search import HybridSearch
    from src.evaluation.metrics import EvaluationMetrics

    if not DOCUMENT_STORE_PATH.exists():
        print("\n  No index found. Run --index <directory> first.\n")
        return

    # Load everything
    print("\n  Loading index for evaluation...")
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
    hybrid = HybridSearch(kw_search, sem_search)

    # Load test queries
    test_queries_path = Path(__file__).parent / "evaluation" / "test_queries.json"
    evaluator = EvaluationMetrics()
    evaluator.load_test_queries(test_queries_path)

    # Evaluate each mode
    for mode_name, search_fn in [
        ("Keyword (TF-IDF)", lambda q: kw_search.search(q, DEFAULT_TOP_K)),
        ("Semantic (Vector)", lambda q: sem_search.search(q, DEFAULT_TOP_K)),
        ("Hybrid", lambda q: hybrid.search(q, DEFAULT_TOP_K)),
    ]:
        print(f"\n  Evaluating: {mode_name}")
        results = evaluator.evaluate_all(search_fn, documents)
        evaluator.print_report(results)


def main():
    parser = argparse.ArgumentParser(
        description="DocSearch - Multilingual Semantic Search for On-Device Document Retrieval",
    )
    parser.add_argument("--index", metavar="DIR", help="Index a directory")
    parser.add_argument("--search", metavar="QUERY", help="Search the index")
    parser.add_argument(
        "--mode",
        choices=["keyword", "semantic", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (default)")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results"
    )

    args = parser.parse_args()

    if args.index:
        cmd_index(args.index)
    elif args.search:
        cmd_search(args.search, args.mode, args.top_k)
    elif args.evaluate:
        cmd_evaluate()
    else:
        # Default: launch GUI
        from src.ui.app import run_app
        run_app()


if __name__ == "__main__":
    main()
