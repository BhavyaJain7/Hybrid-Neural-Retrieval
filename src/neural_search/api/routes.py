from __future__ import annotations

import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger

from neural_search.api.schemas import (
    CollectionMeta,
    CreateCollectionRequest,
    DebugResponse,
    HealthResponse,
    IngestResponse,
    LatencyBreakdown,
    SearchRequest,
    SearchResponse,
)
from neural_search.collections.manager import CollectionManager
from neural_search.config import get_settings
from neural_search.ingestion.pipeline import run_ingestion
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.learned import LearnedHybridFusion
from neural_search.retrieval.reranker import CrossEncoderReranker
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.synthesis.groq_client import GroqSynthesizer

router = APIRouter()
collection_manager = CollectionManager()
settings = get_settings()

# ── Singletons ────────────────────────────────────────────────────────────────

_reranker: CrossEncoderReranker | None = None
_hybrid_cache: dict[str, HybridRetriever] = {}


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_hybrid(slug: str) -> HybridRetriever:
    """Return a cached HybridRetriever for the given collection.

    The cache is keyed by slug. It is evicted in `ingest()` whenever a new
    document is indexed so the BM25 model always reflects the latest corpus.
    """
    if slug not in _hybrid_cache:
        sparse = BM25sRetriever(collection_slug=slug)
        sparse.load()
        dense = QdrantRetriever(collection_slug=slug)
        _hybrid_cache[slug] = HybridRetriever(sparse=sparse, dense=dense)
        logger.debug(f"HybridRetriever cached for collection '{slug}'")
    return _hybrid_cache[slug]


def _require_collection(slug: str) -> dict:
    col = collection_manager.get_collection(slug)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{slug}' not found")
    return col


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    cols = collection_manager.list_collections()
    return HealthResponse(
        status="ok",
        collections_count=len(cols),
        tavily_enabled=settings.tavily_enabled,
    )


# ── Collections ───────────────────────────────────────────────────────────────

@router.get("/collections", response_model=list[CollectionMeta])
def list_collections():
    return collection_manager.list_collections()


@router.post("/collections", response_model=CollectionMeta, status_code=201)
def create_collection(body: CreateCollectionRequest):
    try:
        return collection_manager.create_collection(body.name, body.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/collections/{slug}", response_model=CollectionMeta)
def get_collection(slug: str):
    return _require_collection(slug)


@router.delete("/collections/{slug}", status_code=204)
def delete_collection(slug: str):
    _require_collection(slug)
    collection_manager.delete_collection(slug)


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post("/collections/{slug}/ingest", response_model=IngestResponse)
async def ingest(slug: str, file: UploadFile = File(...), force: bool = False):
    _require_collection(slug)
    filename = file.filename
    warnings: list[str] = []

    dest_dir = settings.documents_path_for(slug)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists() and not force:
        raise HTTPException(
            status_code=409,
            detail=f"{filename} already ingested. Use force=true to re-ingest.",
        )

    content = await file.read()
    dest.write_bytes(content)

    sparse = BM25sRetriever(collection_slug=slug)
    dense = QdrantRetriever(collection_slug=slug)
    chunks = run_ingestion(
        source=dest,
        sparse_retriever=sparse,
        dense_retriever=dense,
        collection_slug=slug,
    )
    if not chunks:
        warnings.append(f"No chunks produced from {filename}")

    total_tokens = sum(c.token_count for c in chunks)
    page_count = len({c.page for c in chunks})

    collection_manager.add_file_record(slug, {
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks),
        "tokens": total_tokens,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
    })

    # Evict cached retriever — BM25 index has changed
    _hybrid_cache.pop(slug, None)
    logger.debug(f"HybridRetriever cache evicted for '{slug}' after ingest")

    return IngestResponse(
        status="ok",
        chunks_indexed=len(chunks),
        warnings=warnings,
    )


# ── Search ────────────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request):
    t_total = time.perf_counter()
    _require_collection(body.collection)
    hybrid = _get_hybrid(body.collection)

    retrieval_ms: float = 0.0
    rerank_ms: float | None = None
    synthesis_ms: float | None = None
    reranked = False
    web_used = False
    retrieval_confidence = 0.0
    expansion_queries: list[str] = []
    synthesis = None
    synthesis_triggered = False

    # ── Retrieval ─────────────────────────────────────────────────────────────
    t_retrieval = time.perf_counter()

    if body.mode == "sparse":
        results = hybrid._sparse.search(body.query, k=body.k)
        retrieval_confidence = results[0]["score"] if results else 0.0

    elif body.mode == "dense":
        results = hybrid._dense.search(body.query, k=body.k)
        retrieval_confidence = results[0]["score"] if results else 0.0

    elif body.mode == "learned":
        fusion = LearnedHybridFusion(sparse=hybrid._sparse, dense=hybrid._dense)
        results = fusion.search(body.query, k=body.k)
        retrieval_confidence = results[0].get("score", 0.0) if results else 0.0

    else:  # hybrid — use search_full() for all new features
        outcome = hybrid.search_full(
            query=body.query,
            k=body.k,
            expand=body.expand,
            query_type=body.query_type,
            web_search=body.web_search,
            rerank=body.rerank,
            rerank_top_k=body.rerank_top_k,
        )
        results = outcome["results"]
        web_used = outcome["web_results_used"]
        retrieval_confidence = outcome["retrieval_confidence"]
        expansion_queries = outcome["expansion_queries"]
        reranked = outcome["reranked"]

        if "rerank_ms" in outcome["latency_ms"]:
            rerank_ms = outcome["latency_ms"]["rerank_ms"]

        for key, val in outcome["latency_ms"].items():
            logger.debug(f"  {key}: {val}ms")

    retrieval_ms = round((time.perf_counter() - t_retrieval) * 1000, 2)

    # ── Reranking for non-hybrid modes (original behaviour kept) ─────────────
    if body.rerank and body.mode != "hybrid":
        t_rerank = time.perf_counter()
        results = _get_reranker().rerank(body.query, results, top_k=body.rerank_top_k)
        rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 2)
        reranked = True

    # ── Confidence-gated synthesis ────────────────────────────────────────────
    if body.synthesize and settings.synthesis_enabled:
        if retrieval_confidence >= settings.synthesis_threshold:
            try:
                t_synth = time.perf_counter()
                synthesizer: GroqSynthesizer = request.app.state.synthesizer
                synthesis = synthesizer.synthesize(body.query, results)
                synthesis_ms = round((time.perf_counter() - t_synth) * 1000, 2)
                synthesis_triggered = True
            except Exception as e:
                logger.warning(f"Synthesis failed: {e}")
        else:
            logger.info(
                f"Synthesis skipped — confidence {retrieval_confidence:.3f} "
                f"< threshold {settings.synthesis_threshold}"
            )

    total_ms = round((time.perf_counter() - t_total) * 1000, 2)

    # Attach collection to every result (original behaviour)
    enriched = [{**r, "collection": body.collection} for r in results]

    return SearchResponse(
        query=body.query,
        mode=body.mode,
        reranked=reranked,
        results=enriched,
        synthesis=synthesis,
        latency_ms=total_ms,
        latency=LatencyBreakdown(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            synthesis_ms=synthesis_ms,
            total_ms=total_ms,
        ),
        web_results_used=web_used,
        retrieval_confidence=retrieval_confidence,
        synthesis_triggered=synthesis_triggered,
        expansion_queries=expansion_queries,
    )


# ── Debug ─────────────────────────────────────────────────────────────────────

@router.get("/search/debug", response_model=DebugResponse)
def search_debug(query: str, collection: str, k: int = 10):
    _require_collection(collection)
    hybrid = _get_hybrid(collection)
    debug = hybrid.search_debug(query, k=k)
    return DebugResponse(
        sparse=debug["sparse"],
        dense=debug["dense"],
        hybrid_rrf=debug["hybrid_rrf"],
        web=debug.get("web", []),
    )
