"""
Configuration module for the Multilingual Semantic Search application.
Stores all paths, constants, and settings used across the project.
"""

import os
import json
from pathlib import Path


# ── Base Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TEST_CORPUS_DIR = PROJECT_ROOT / "IR_DOCUMNETS"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Index File Paths ────────────────────────────────────────────────────────
INVERTED_INDEX_PATH = DATA_DIR / "inverted_index.pkl"
TFIDF_INDEX_PATH = DATA_DIR / "tfidf_index.pkl"
FAISS_INDEX_PATH = DATA_DIR / "vectors.faiss"
METADATA_PATH = DATA_DIR / "metadata.json"
DOCUMENT_STORE_PATH = DATA_DIR / "doc_store.pkl"
SETTINGS_PATH = DATA_DIR / "settings.json"
QUERY_SYNONYMS_PATH = DATA_DIR / "query_synonyms.json"

# ── Model Configuration ────────────────────────────────────────────────────
# multilingual-e5-small: multilingual retrieval model, CPU-efficient (384 dims)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_LOCAL_DIR = MODELS_DIR / "sentence-transformers" / "multilingual-e5-small"
EMBEDDING_DIMENSION = 384  # Output dimension of multilingual-e5-small

# E5 models perform best with explicit query/document prefixes.
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_DOCUMENT_PREFIX = "passage: "

# ── OCR Configuration ───────────────────────────────────────────────────────
# Tesseract OCR settings
PADDLE_OCR_LANGS = ["en", "hi", "te"]
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = True

# ── Supported File Extensions ───────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    "pdf": [".pdf"],
    "docx": [".docx"],
    "text": [".txt", ".md", ".csv", ".log"],
    "excel": [".xlsx", ".xls"],
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".avif", ".webp"],
}

ALL_SUPPORTED_EXTENSIONS = []
for exts in SUPPORTED_EXTENSIONS.values():
    ALL_SUPPORTED_EXTENSIONS.extend(exts)

# ── Search Configuration ───────────────────────────────────────────────────
DEFAULT_TOP_K = 10  # Number of results to return
HYBRID_ALPHA = 0.6  # Weight for semantic score (1-alpha for keyword score)
MIN_SCORE_THRESHOLD = 0.01  # Minimum score to include in results
SCORE_THRESHOLD = 0.5  # Fixed threshold fallback (normalized scores)
ADAPTIVE_THRESHOLD_K = 0.3  # Adaptive threshold: mean + k * std
MIN_RESULTS = 3  # Minimum results to return when threshold allows
MAX_RESULTS = 10  # Maximum results to return
SCORE_GAP_THRESHOLD = 0.15  # Cut tail if score drop exceeds this

# ── FAISS Vector Index Configuration ───────────────────────────────────────
# Supported index types: "flat", "hnsw"
FAISS_INDEX_TYPE = "hnsw"

# HNSW tuning knobs
# M controls graph connectivity: higher M improves recall, increases memory/build time.
FAISS_HNSW_M = 32
# efConstruction controls exploration depth while adding vectors.
FAISS_HNSW_EF_CONSTRUCTION = 200
# efSearch controls search-time recall/speed tradeoff.
FAISS_HNSW_EF_SEARCH = 64

# Reciprocal Rank Fusion (RRF) configuration
RRF_K = 60  # Standard RRF constant

# ── Cross-Encoder Re-ranking Configuration ─────────────────────────────────
ENABLE_CROSS_ENCODER_RERANK = True
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL_LOCAL_DIR = MODELS_DIR / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"
# Optional multilingual cross-encoder for non-English reranking.
USE_MULTILINGUAL_CROSS_ENCODER = True
MULTILINGUAL_CROSS_ENCODER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR = (
    MODELS_DIR / "cross-encoder" / "mmarco-mMiniLMv2-L12-H384-v1"
)
CROSS_ENCODER_CANDIDATES = 30  # Re-rank top-N fused results
CROSS_ENCODER_TOP_K = 10  # Final results after re-ranking
CROSS_ENCODER_BATCH_SIZE = 16
CROSS_ENCODER_MAX_LENGTH = 256

# Privacy-first mode: load models only from local disk when enabled.
OFFLINE_MODE = True

# Keep CPU thread usage modest for desktop responsiveness.
CPU_THREADS = max(1, min(4, os.cpu_count() or 1))

# ── Preprocessing Configuration ────────────────────────────────────────────
MIN_TOKEN_LENGTH = 2  # Minimum word length to keep
MAX_TOKEN_LENGTH = 50  # Maximum word length to keep

# ── Chunking Configuration ─────────────────────────────────────────────────
# Supported: "fixed" (word window), "semantic" (sentence-similarity based)
CHUNKING_STRATEGY = "semantic"

# Hard limit for chunk size in words (applies to both strategies)
MAX_CHUNK_TOKENS = 180

# Semantic chunking controls
SEMANTIC_CHUNK_SIMILARITY_THRESHOLD = 0.68
SEMANTIC_CHUNK_MIN_TOKENS = 60
SEMANTIC_CHUNK_OVERLAP_SENTENCES = 1

# ── Query Expansion Configuration ──────────────────────────────────────────
ENABLE_QUERY_EXPANSION = True
QUERY_EXPANSION_SHORT_QUERY_MAX_TOKENS = 3
QUERY_EXPANSION_MAX_SYNONYMS_PER_TERM = 2
QUERY_EXPANSION_ORIGINAL_WEIGHT = 1.0
QUERY_EXPANSION_EXPANDED_WEIGHT = 0.3

# ── Query Intent Configuration ─────────────────────────────────────────────
# Intent keyword lists are ASCII-only for portability; expand as needed.
INTENT_KEYWORDS = {
    "marksheet": ["marksheet", "transcript", "grade sheet", "score sheet", "semester"],
    "invoice": ["invoice", "bill", "billing", "receipt", "tax invoice"],
    "permission": ["permission", "approval", "noc", "no objection", "permit", "house permission"],
    "certificate": ["certificate", "cert", "provisional", "bonafide"],
    "id": ["id", "identity", "aadhaar", "aadhar", "pan", "passport"],
    "address": ["address", "residence", "residential proof", "ration card"],
    "property": ["property", "land", "plot", "survey", "patta"],
}
INTENT_CANONICAL_TERMS = {
    "marksheet": ["marksheet", "transcript"],
    "invoice": ["invoice", "bill"],
    "permission": ["permission", "approval"],
    "certificate": ["certificate"],
    "id": ["id", "identity"],
    "address": ["address", "residence"],
    "property": ["property", "land"],
}
INTENT_TERM_BOOST = 1.4
INTENT_MAX_ADDED_TERMS = 4

# ── Query Term Boosts ──────────────────────────────────────────────────────
QUERY_TERM_BOOSTS = {
    "marksheet": 1.6,
    "result": 1.3,
    "transcript": 1.3,
    "id": 1.4,
    "identity": 1.4,
    "card": 1.2,
    "electricity": 1.3,
    "power": 1.2,
    "current": 1.2,
}

QUERY_PHRASE_EXPANSIONS = {
    "marksheet semester": ["marksheet", "result", "transcript"],
    "id card": ["identity", "identity card"],
    "house permission": ["permission", "approval", "noc"],
    "electricity power": ["electricity", "power", "current", "power supply"],
}
QUERY_PHRASE_BOOST = 1.5

# ── Filename Injection & Tags ─────────────────────────────────────────────
ENABLE_FILENAME_INJECTION = True
ENABLE_FILENAME_TAGS = True
FILENAME_TAGS = {
    "id": ["identity", "card", "student"],
    "identity": ["id", "card"],
    "card": ["id", "identity"],
    "electricity": ["power", "bill"],
    "power": ["electricity", "bill"],
    "permission": ["approval", "permit"],
    "permit": ["permission", "approval"],
    "schedule": ["timetable", "calendar"],
    "timetable": ["schedule", "calendar"],
}

# ── Keyword Phrase Boost ─────────────────────────────────────────────────-
PHRASE_BOOST = 0.15
PHRASE_BOOST_TERMS = [
    "id card",
    "identity card",
    "student id",
    "electricity bill",
    "power bill",
    "house permission",
    "building permit",
    "timetable",
    "class routine",
]

# ── Selective Expansion for Failing Queries ───────────────────────────────
FAILING_QUERY_EXPANSIONS = {
    "electricity": ["power supply", "current", "electric bill"],
    "house permission": ["building permit", "construction approval"],
    "id card": ["identity card", "student id", "id proof"],
    "environment": ["environmental", "climate", "pollution control"],
    "timetable": ["schedule", "calendar", "class routine"],
}
FAILING_QUERY_ORIGINAL_WEIGHT = 1.0
FAILING_QUERY_EXPANDED_WEIGHT = 0.3

# ── Fusion Tuning Configuration ────────────────────────────────────────────
# Supported methods: "rrf", "weighted_rrf"
FUSION_METHOD = "weighted_rrf"
FUSION_KEYWORD_WEIGHT = 0.2
FUSION_SEMANTIC_WEIGHT = 0.8
FUSION_RERANKER_WEIGHT = 0.4
RERANKER_MAX_INFLUENCE = 0.35
FUSION_KEYWORD_DIGIT_BOOST = 0.35
FUSION_KEYWORD_SHORT_QUERY_BOOST = 0.2
FUSION_SEMANTIC_LONG_QUERY_BOOST = 0.2

# ── Application Settings ───────────────────────────────────────────────────
APP_NAME = "DocSearch - Multilingual Semantic Search"
APP_VERSION = "1.0.0"
APP_WINDOW_SIZE = "1100x700"


def load_settings() -> dict:
    """Load user settings from disk."""
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"indexed_directories": [], "search_mode": "hybrid", "top_k": DEFAULT_TOP_K}


def save_settings(settings: dict):
    """Save user settings to disk."""
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
