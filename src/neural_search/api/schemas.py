from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=256)


class FileRecord(BaseModel):
    filename: str
    pages: int
    chunks: int
    tokens: int
    ingested_at: str
    status: str = "ok"


class CollectionMeta(BaseModel):
    slug: str
    name: str
    description: str
    created_at: str
    updated_at: str
    files: list[FileRecord]
    total_chunks: int
    total_tokens: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str
    k: int = Field(default=10, ge=1, le=50)
    synthesize: bool = False
    # "learned" kept for backward compat with test_phase4_reranker.py
    mode: Literal["hybrid", "sparse", "dense", "learned"] = "hybrid"
    # original rerank fields kept — still honoured in routes.py
    rerank: bool = False
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    # new fields
    expand: bool = False
    web_search: bool = False
    query_type: Literal["keyword", "semantic", "vague"] = "semantic"


class ChunkResult(BaseModel):
    chunk_id: str
    source_file: str
    page: int
    token_count: int
    text: str
    score: float
    rank: int
    source: str
    collection: Optional[str] = None          # added by routes.py on every result
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None
    # new web fields
    source_url: Optional[str] = None
    freshness_weight: Optional[float] = None


class SynthesisResult(BaseModel):
    answer: str
    sources_used: list[dict]
    model: str


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    rerank_ms: Optional[float] = None
    synthesis_ms: Optional[float] = None
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    reranked: bool = False
    results: list[ChunkResult]
    synthesis: Optional[SynthesisResult] = None
    latency_ms: float
    latency: Optional[LatencyBreakdown] = None
    # new response fields
    web_results_used: bool = False
    retrieval_confidence: float = 0.0
    synthesis_triggered: bool = False
    expansion_queries: list[str] = []


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    warnings: list[str] = []


class DebugResponse(BaseModel):
    sparse: list[dict]
    dense: list[dict]
    hybrid_rrf: list[dict]
    web: list[dict] = []                      # new
    expansion_queries: list[str] = []         # new


class HealthResponse(BaseModel):
    status: str = "ok"
    collections_count: int
    tavily_enabled: bool = False
