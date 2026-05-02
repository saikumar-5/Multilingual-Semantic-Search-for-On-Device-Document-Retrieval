# save as tools/download_multilingual_reranker.py
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
TARGET_DIR = Path("models") / "cross-encoder" / "mmarco-mMiniLMv2-L12-H384-v1"

TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(TARGET_DIR),
    local_dir_use_symlinks=False,
)

print(f"Downloaded to: {TARGET_DIR}")