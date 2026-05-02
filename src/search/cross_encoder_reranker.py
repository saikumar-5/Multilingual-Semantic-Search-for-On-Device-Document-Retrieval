"""
Cross-encoder re-ranker for second-stage document ranking.

This module applies a lightweight cross-encoder model over top candidate
results produced by first-stage retrieval (hybrid fusion).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import logging
import re
from pathlib import Path

import numpy as np

from src.config import (
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_CANDIDATES,
    CROSS_ENCODER_MAX_LENGTH,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_MODEL_LOCAL_DIR,
    CPU_THREADS,
    OFFLINE_MODE,
)

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranks candidate documents using query-document cross-encoding."""

    def __init__(
        self,
        documents: List[dict],
        model_name: str = CROSS_ENCODER_MODEL_NAME,
        model_local_dir: Optional[Path] = None,
        top_candidates: int = CROSS_ENCODER_CANDIDATES,
        batch_size: int = CROSS_ENCODER_BATCH_SIZE,
        max_length: int = CROSS_ENCODER_MAX_LENGTH,
        offline_mode: bool = OFFLINE_MODE,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.model_local_dir = model_local_dir
        self.top_candidates = max(1, top_candidates)
        self.batch_size = max(1, batch_size)
        self.max_length = max(32, max_length)
        self.offline_mode = offline_mode
        self.device = device

        # Store compact text per doc_id for fast pair construction.
        self.doc_text: Dict[int, str] = {
            doc["doc_id"]: self._prepare_text(doc.get("text", ""))
            for doc in documents
            if "doc_id" in doc
        }

        self._model = None
        self._load_error = None

    def rerank(
        self,
        query: str,
        fused_results: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank the top fused results and return the final top_k.

        Falls back to fused ranking if model is unavailable.
        """
        if not query or not fused_results:
            return fused_results[:top_k]

        candidates = fused_results[: self.top_candidates]
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            return candidates[:top_k]

        pairs = []
        valid_ids = []
        prior_scores = {}

        for doc_id, fused_score in candidates:
            text = self.doc_text.get(doc_id, "")
            if not text:
                continue
            pairs.append((query, text))
            valid_ids.append(doc_id)
            prior_scores[doc_id] = fused_score

        if not pairs:
            return candidates[:top_k]

        try:
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            logger.warning("Cross-encoder reranking failed, using fused results: %s", e)
            return candidates[:top_k]

        # Rank primarily by cross-encoder score, then by fused score as tie-breaker.
        reranked = []
        for doc_id, ce_score in zip(valid_ids, np.asarray(scores).tolist()):
            reranked.append((doc_id, float(ce_score), prior_scores.get(doc_id, 0.0)))

        reranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(doc_id, ce_score) for doc_id, ce_score, _ in reranked[:top_k]]

    def _prepare_text(self, text: str, max_chars: int = 1200) -> str:
        """Normalize whitespace and cap text length for faster inference."""
        clean = re.sub(r"\s+", " ", text or "").strip()
        if len(clean) > max_chars:
            return clean[:max_chars]
        return clean

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None

        try:
            import torch
            from sentence_transformers import CrossEncoder

            try:
                torch.set_num_threads(CPU_THREADS)
            except Exception as thread_err:
                logger.debug("Could not set torch num threads: %s", thread_err)

            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except Exception as interop_err:
                    logger.debug(
                        "Could not set torch interop threads: %s", interop_err
                    )

            model_source = self.model_name
            local_files_only = False

            local_dir = self.model_local_dir or CROSS_ENCODER_MODEL_LOCAL_DIR
            if local_dir and local_dir.exists():
                model_source = str(local_dir)
                local_files_only = True
            elif self.offline_mode:
                raise FileNotFoundError(
                    f"Offline mode enabled but cross-encoder model not found at: {local_dir}"
                )

            self._model = CrossEncoder(
                model_source,
                device=self.device,
                max_length=self.max_length,
                local_files_only=local_files_only,
            )
            logger.info("Loaded cross-encoder re-ranker from: %s", model_source)
            return self._model
        except Exception as e:
            self._load_error = e
            logger.warning("Failed to load cross-encoder model %s: %s", self.model_name, e)
            return None
