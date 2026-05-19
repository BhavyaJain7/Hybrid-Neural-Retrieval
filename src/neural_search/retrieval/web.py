from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

from loguru import logger

from neural_search.config import get_settings
from neural_search.ingestion.chunker import _count_tokens

settings = get_settings()

_FRESHNESS_CUTOFF_DAYS = 180


def _web_chunk_id(url: str) -> str:
    return "web::" + hashlib.sha1(url.encode()).hexdigest()[:12]


def _freshness_weight(published_date: str | None) -> float:
    """
    Returns weight in [0.5, 1.0].
    Unknown date → 0.8 (mild penalty).
    Older than cutoff → 0.5.
    """
    if not published_date:
        return 0.8
    try:
        pub = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - pub).days
        if age_days > _FRESHNESS_CUTOFF_DAYS:
            return 0.5
        return round(1.0 - 0.5 * (age_days / _FRESHNESS_CUTOFF_DAYS), 3)
    except Exception:
        return 0.8


class TavilyRetriever:
    """
    Fetches web snippets via Tavily and scores them using the shared
    SentenceTransformer model. Model is injected — not imported from dense.py.
    Results follow the same dict schema as BM25sRetriever / QdrantRetriever.
    """

    def __init__(self, model) -> None:
        if settings.tavily_api_key == "not-set":
            raise ValueError("TAVILY_API_KEY is not configured in .env")
        from tavily import TavilyClient
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        self._model = model

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or settings.tavily_max_results
        t0 = time.perf_counter()

        try:
            response = self._client.search(
                query=query,
                max_results=k,
                search_depth="basic",
                include_answer=False,
            )
        except Exception as e:
            logger.warning(f"Tavily search failed for '{query}': {e}")
            return []

        raw_results = response.get("results", [])
        if not raw_results:
            return []

        texts = [r.get("content", "") for r in raw_results]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        query_emb = self._model.encode(query, normalize_embeddings=True)

        output = []
        for i, (result, emb) in enumerate(zip(raw_results, embeddings)):
            text = result.get("content", "").strip()
            if not text:
                continue
            raw_score = float(query_emb @ emb)
            fw = _freshness_weight(result.get("published_date"))
            output.append({
                "chunk_id": _web_chunk_id(result.get("url", str(i))),
                "source_file": result.get("url", "web"),
                "source_url": result.get("url"),
                "page": 0,
                "text": text,
                "token_count": _count_tokens(text),
                "score": round(raw_score * fw, 6),
                "rank": i + 1,
                "source": "web",
                "freshness_weight": fw,
            })

        output.sort(key=lambda x: x["score"], reverse=True)

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"Tavily: {len(output)} results in {elapsed}ms for '{query}'")
        return output
