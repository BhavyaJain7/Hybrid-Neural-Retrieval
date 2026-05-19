from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Path:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return cwd / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_find_env_file()),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = "not-set"
    groq_model: str = "llama-3.1-8b-instant"

    # ── Tavily (new) ──────────────────────────────────────────────────────────
    tavily_api_key: str = "not-set"
    tavily_enabled: bool = False
    tavily_max_results: int = 5
    tavily_weight: float = 0.6
    web_trigger_threshold: float = 0.010  # RRF max≈0.033; 0.010 ≈ bottom 20% — fires only on weak retrieval

    # ── Synthesis gating (new) ────────────────────────────────────────────────
    synthesis_threshold: float = 0.01  # RRF scores max ~0.033 — 0.4 was unreachable
    synthesis_enabled: bool = True

    # ── Retrieval ─────────────────────────────────────────────────────────────
    qdrant_path: Path = Path("./data/qdrant")
    bm25_index_path: Path = Path("./data/bm25_index")
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # kept
    top_k: int = 10
    rrf_k: int = 60

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Path("./data")

    def bm25_path_for(self, collection_slug: str) -> Path:
        return self.bm25_index_path / collection_slug

    def documents_path_for(self, collection_slug: str) -> Path:
        return self.data_dir / "documents" / collection_slug

    def snapshot_path_for(self, collection_slug: str) -> Path:
        return self.data_dir / "snapshots" / collection_slug / "chunks.jsonl"

    def ensure_dirs(self) -> None:
        for path in [
            self.qdrant_path,
            self.bm25_index_path,
            self.data_dir / "documents",
            self.data_dir / "snapshots",
            self.data_dir / "collections",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def assert_groq_configured(self) -> None:
        if self.groq_api_key == "not-set":
            raise RuntimeError("GROQ_API_KEY is not set in .env")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
