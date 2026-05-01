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

# ── Tesseract OCR Configuration ─────────────────────────────────────────────
# Default Tesseract path on Windows
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Languages for OCR: English + Hindi + Telugu
OCR_LANGUAGES = "eng+hin+tel"

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
CROSS_ENCODER_CANDIDATES = 50  # Re-rank top-N fused results
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
MAX_CHUNK_TOKENS = 256

# Semantic chunking controls
SEMANTIC_CHUNK_SIMILARITY_THRESHOLD = 0.62
SEMANTIC_CHUNK_MIN_TOKENS = 80
SEMANTIC_CHUNK_OVERLAP_SENTENCES = 1

# ── Query Expansion Configuration ──────────────────────────────────────────
ENABLE_QUERY_EXPANSION = True
QUERY_EXPANSION_SHORT_QUERY_MAX_TOKENS = 3
QUERY_EXPANSION_MAX_SYNONYMS_PER_TERM = 2
QUERY_EXPANSION_ORIGINAL_WEIGHT = 1.0
QUERY_EXPANSION_EXPANDED_WEIGHT = 0.3

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
