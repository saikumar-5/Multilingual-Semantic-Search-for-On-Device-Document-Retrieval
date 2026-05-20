from pathlib import Path
import pickle
from src.config import DATA_DIR, DOCUMENT_STORE_PATH, INVERTED_INDEX_PATH, TFIDF_INDEX_PATH, DEFAULT_TOP_K
from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine
from src.indexer.vector_index import VectorIndex
from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch
from src.search.hybrid_search import HybridSearch
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.evaluation.metrics import EvaluationMetrics
from src.main import _resolve_reranker_config


def load_all():
    with open(DOCUMENT_STORE_PATH, "rb") as f:
        documents = pickle.load(f)
    inv_index = InvertedIndex()
    inv_index.load(INVERTED_INDEX_PATH)
    tfidf = TFIDFEngine(inv_index)
    tfidf.load(TFIDF_INDEX_PATH)
    vec_index = VectorIndex()
    vec_index.load(DATA_DIR / "vectors.faiss", DATA_DIR / "vector_meta.pkl")
    return documents, inv_index, tfidf, vec_index


documents, inv_index, tfidf, vec_index = load_all()
print("Loaded indexes and documents")

keyword_search = KeywordSearch(inv_index, tfidf)
semantic_search = SemanticSearch(vec_index)

hybrid_no_rerank = HybridSearch(keyword_search, semantic_search, reranker=None, rerank_candidates=30)

# Create reranker with the same configuration used by CLI and UI.
model_name, model_local_dir, rerank_english_only = _resolve_reranker_config()
cross_encoder = CrossEncoderReranker(documents, model_name=model_name, model_local_dir=model_local_dir)
hybrid_with_rerank = HybridSearch(
    keyword_search,
    semantic_search,
    reranker=cross_encoder,
    rerank_candidates=30,
    rerank_english_only=rerank_english_only,
)

evaluator = EvaluationMetrics()
test_queries_path = Path(__file__).resolve().parent.parent / "src" / "evaluation" / "test_queries.json"
evaluator.load_test_queries(test_queries_path)

for label, search_fn in [
    ("Hybrid without rerank", lambda q: hybrid_no_rerank.search(q, DEFAULT_TOP_K)),
    ("Hybrid with rerank", lambda q: hybrid_with_rerank.search(q, DEFAULT_TOP_K)),
]:
    print("\n=== {} ===".format(label))
    results = evaluator.evaluate_all(search_fn, documents)
    evaluator.print_report(results)
