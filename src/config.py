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

# ── Model Configuration ────────────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2: ~120MB, supports 50+ languages
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384  # Output dimension of MiniLM-L12-v2
MAX_CHUNK_TOKENS = 256  # Max tokens per document chunk for embedding

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

# ── Preprocessing Configuration ────────────────────────────────────────────
MIN_TOKEN_LENGTH = 2  # Minimum word length to keep
MAX_TOKEN_LENGTH = 50  # Maximum word length to keep

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
