from __future__ import annotations

import numpy as np
from loguru import logger

_SIMILARITY_THRESHOLD = 0.85


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def deduplicate_web_results(
    web_results: list[dict],
    local_results: list[dict],
    model,
    threshold: float = _SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Drop web results whose content is already covered by local results.

    Embeds texts once per call. A web result is dropped if its cosine
    similarity to any local result exceeds `threshold`.

    Args:
        web_results:  Results from TavilyRetriever.
        local_results: Combined sparse + dense local results.
        model:        SentenceTransformer instance (shared, not re-loaded).
        threshold:    Similarity cutoff. Default 0.85.

    Returns:
        Filtered web results that add genuinely new information.
    """
    if not web_results or not local_results:
        return web_results

    local_texts = [r["text"] for r in local_results]
    web_texts = [r["text"] for r in web_results]

    # Encode all at once — one forward pass each
    local_embs = model.encode(local_texts, normalize_embeddings=True)
    web_embs = model.encode(web_texts, normalize_embeddings=True)

    kept = []
    for web_result, web_emb in zip(web_results, web_embs):
        max_sim = max(_cosine_sim(web_emb, loc_emb) for loc_emb in local_embs)
        if max_sim < threshold:
            kept.append(web_result)
        else:
            logger.debug(
                f"Dedup: dropped web chunk (sim={max_sim:.3f}) "
                f"'{web_result['text'][:60]}...'"
            )

    logger.debug(f"Dedup: {len(web_results)} web → {len(kept)} kept after dedup")
    return kept
