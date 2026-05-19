from __future__ import annotations

import threading
import time
from typing import Optional

from loguru import logger
from sentence_transformers import CrossEncoder

from neural_search.config import get_settings

settings = get_settings()

_MODEL: Optional[CrossEncoder] = None
_MODEL_LOCK = threading.Lock()
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                model_name = getattr(settings, "reranker_model", _DEFAULT_MODEL)
                t0 = time.perf_counter()
                _MODEL = CrossEncoder(model_name)
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                logger.info(f"Reranker loaded: {model_name} in {elapsed}ms")
    return _MODEL


class CrossEncoderReranker:
    """
    Wraps a cross-encoder for reranking.
    Kept as a class so existing routes.py (_get_reranker()) and
    test_phase4_reranker.py continue to work without changes.
    """

    def __init__(self) -> None:
        self._model = _get_model()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        if not candidates:
            return candidates

        top_k = top_k or len(candidates)
        top_k = min(top_k, len(candidates))

        t0 = time.perf_counter()
        pairs = [(query, c["text"]) for c in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for rank, (chunk, score) in enumerate(scored[:top_k], start=1):
            entry = dict(chunk)
            entry["rerank_score"] = round(float(score), 6)
            entry["rerank_rank"] = rank
            entry["rank"] = rank
            results.append(entry)

        logger.debug(
            f"Reranked {len(candidates)} → {top_k} in {elapsed_ms}ms | "
            f"top score: {results[0]['rerank_score']:.4f}"
        )
        return results
