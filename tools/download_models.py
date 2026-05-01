"""Download all NLP models to local project storage for offline use."""

from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_LOCAL_DIR,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_MODEL_LOCAL_DIR,
)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_embedding_model():
    ensure_parent(EMBEDDING_MODEL_LOCAL_DIR)
    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.save(str(EMBEDDING_MODEL_LOCAL_DIR))
    print(f"Saved embedding model to: {EMBEDDING_MODEL_LOCAL_DIR}")


def download_cross_encoder_model():
    ensure_parent(CROSS_ENCODER_MODEL_LOCAL_DIR)
    print(f"Downloading cross-encoder model: {CROSS_ENCODER_MODEL_NAME}")
    model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    model.save_pretrained(str(CROSS_ENCODER_MODEL_LOCAL_DIR))
    print(f"Saved cross-encoder model to: {CROSS_ENCODER_MODEL_LOCAL_DIR}")


def main():
    download_embedding_model()
    download_cross_encoder_model()
    print("All models downloaded for offline usage.")


if __name__ == "__main__":
    main()
