from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from neural_search.config import get_settings
from neural_search.retrieval.dense import QdrantRetriever, _get_model
from neural_search.retrieval.sparse import BM25sRetriever

settings = get_settings()


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf(
    result_lists: list[tuple[list[dict], float]],
    rrf_k: int | None = None,
) -> list[dict]:
    """
    Weighted Reciprocal Rank Fusion over N result lists.

    Args:
        result_lists: List of (results, weight) tuples.
        rrf_k:        RRF constant (default from settings).

    Returns:
        Fused and ranked list with rrf_score on each entry.
    """
    k = rrf_k or settings.rrf_k
    scores: dict[str, float] = {}
    sources: dict[str, set] = {}
    meta: dict[str, dict] = {}

    for results, weight in result_lists:
        for rank, result in enumerate(results, start=1):
            cid = result["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            sources.setdefault(cid, set()).add(result.get("source", "unknown"))
            if cid not in meta:
                meta[cid] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for final_rank, (cid, rrf_score) in enumerate(ranked, start=1):
        entry = dict(meta[cid])
        entry["rrf_score"] = round(rrf_score, 6)
        entry["rank"] = final_rank
        # Use "+" separator — matches existing test expectations
        entry["source"] = "+".join(sorted(sources[cid]))
        fused.append(entry)

    return fused


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    3-way hybrid retriever: BM25 + Qdrant + optional Tavily web.

    Public API:
        search()       → list[dict]   (backward compatible — all existing tests pass)
        search_full()  → dict         (enriched result with metadata for routes.py)
        search_debug() → dict         (per-source breakdown)
    """

    def __init__(self, sparse: BM25sRetriever, dense: QdrantRetriever) -> None:
        self._sparse = sparse
        self._dense = dense
        self._web: Optional[object] = None
        self._reranker: Optional[object] = None  # lazy-loaded on first use

        if settings.tavily_enabled:
            try:
                from neural_search.retrieval.web import TavilyRetriever
                self._web = TavilyRetriever(model=_get_model())
                logger.info("TavilyRetriever initialised")
            except Exception as e:
                logger.warning(f"Tavily disabled: {e}")

    def _should_trigger_web(self, top_local_score: float, force_web: bool) -> bool:
        if not settings.tavily_enabled or self._web is None:
            return False
        if force_web:
            return True
        return top_local_score < settings.web_trigger_threshold

    # ── Primary public method — backward compatible ───────────────────────────

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """
        Standard 2-way RRF search. Returns list[dict].
        Signature unchanged — all existing tests continue to pass.
        """
        k = k or settings.top_k
        sparse_results = self._sparse.search(query, k=k)
        dense_results = self._dense.search(query, k=k)
        fused = _rrf([(sparse_results, 1.0), (dense_results, 1.0)])
        return fused[:k]

    # ── Extended method for routes.py ─────────────────────────────────────────

    def search_full(
        self,
        query: str,
        k: int | None = None,
        expand: bool = False,
        query_type: str = "semantic",
        web_search: bool = False,
        rerank: bool = False,
        rerank_top_k: int = 5,
    ) -> dict:
        """
        Full pipeline: expansion → 3-way RRF → optional reranking.

        Args:
            query_type: One of "keyword", "semantic", "vague".
                        Expansion only fires when query_type == "vague" (or expand
                        is True AND type is not "keyword").

        Returns:
            {
                results: list[dict],
                web_results_used: bool,
                retrieval_confidence: float,
                expansion_queries: list[str],
                reranked: bool,
                latency_ms: dict,
            }
        """
        k = k or settings.top_k
        timings: dict[str, float] = {}

        # ── Query expansion ───────────────────────────────────────────────────
        # Expansion only fires for vague queries; keyword queries must never expand
        # (expansion adds noise on precise terms).
        should_expand = expand and query_type != "keyword"
        queries = [query]
        if should_expand:
            from neural_search.retrieval.expander import expand_query
            t0 = time.perf_counter()
            queries = expand_query(query, n=2)
            timings["expansion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Local retrieval — merge across expanded queries ───────────────────
        t0 = time.perf_counter()
        sparse_seen: dict[str, dict] = {}
        dense_seen: dict[str, dict] = {}

        for q in queries:
            for r in self._sparse.search(q, k=k):
                sparse_seen.setdefault(r["chunk_id"], r)
            for r in self._dense.search(q, k=k):
                dense_seen.setdefault(r["chunk_id"], r)

        sparse_results = list(sparse_seen.values())
        dense_results = list(dense_seen.values())
        timings["local_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Local RRF to get confidence score ─────────────────────────────────
        local_fused = _rrf([(sparse_results, 1.0), (dense_results, 1.0)])
        top_local_score = local_fused[0]["rrf_score"] if local_fused else 0.0

        # ── Web retrieval (gated) ─────────────────────────────────────────────
        web_results: list[dict] = []
        web_used = False

        if self._should_trigger_web(top_local_score, web_search):
            from neural_search.retrieval.deduplicator import deduplicate_web_results
            t0 = time.perf_counter()
            raw_web = self._web.search(query, k=settings.tavily_max_results)
            all_local = sparse_results + dense_results
            web_results = deduplicate_web_results(raw_web, all_local, model=_get_model())
            timings["web_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            web_used = bool(web_results)
            logger.info(
                f"Web search: {len(raw_web)} raw → {len(web_results)} after dedup "
                f"(local confidence={top_local_score:.3f})"
            )

        # ── 3-way RRF fusion ──────────────────────────────────────────────────
        result_lists: list[tuple[list[dict], float]] = [
            (sparse_results, 1.0),
            (dense_results, 1.0),
        ]
        if web_results:
            result_lists.append((web_results, settings.tavily_weight))

        fused = _rrf(result_lists)
        retrieval_confidence = fused[0]["rrf_score"] if fused else 0.0

        # ── Optional reranking ────────────────────────────────────────────────
        reranked_flag = False
        if rerank and len(fused) > 1:
            if self._reranker is None:
                from neural_search.retrieval.reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
                logger.debug("CrossEncoderReranker loaded and cached on HybridRetriever")
            t0 = time.perf_counter()
            fused = self._reranker.rerank(query, fused[: k * 2], top_k=rerank_top_k)
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            reranked_flag = True
        else:
            fused = fused[:k]
            for i, r in enumerate(fused, start=1):
                r["rank"] = i

        return {
            "results": fused,
            "web_results_used": web_used,
            "retrieval_confidence": round(retrieval_confidence, 6),
            "expansion_queries": queries if expand else [],
            "reranked": reranked_flag,
            "latency_ms": timings,
        }

    # ── Debug ─────────────────────────────────────────────────────────────────

    def search_debug(self, query: str, k: int | None = None) -> dict:
        k = k or settings.top_k
        sparse_results = self._sparse.search(query, k=k)
        dense_results = self._dense.search(query, k=k)
        web_results: list[dict] = []

        if self._web:
            try:
                web_results = self._web.search(query, k=settings.tavily_max_results)
            except Exception as e:
                logger.warning(f"Tavily debug search failed: {e}")

        result_lists: list[tuple[list[dict], float]] = [
            (sparse_results, 1.0),
            (dense_results, 1.0),
        ]
        if web_results:
            result_lists.append((web_results, settings.tavily_weight))

        fused = _rrf(result_lists)[:k]

        return {
            "sparse": sparse_results,
            "dense": dense_results,
            "web": web_results,
            "hybrid_rrf": fused,
        }
