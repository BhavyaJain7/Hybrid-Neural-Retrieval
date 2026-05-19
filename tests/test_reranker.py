"""Unit tests for CrossEncoderReranker (Phase 4)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_search.retrieval.reranker import CrossEncoderReranker


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunks(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"Chunk {i} content about AI agents and tools.",
            "source_file": "doc.pdf",
            "page": i,
            "token_count": 10,
            "score": float(n - i),
            "rank": i + 1,
            "source": "hybrid",
            "collection": "base",
        }
        for i in range(n)
    ]


@pytest.fixture()
def reranker():
    """Reranker with mocked cross-encoder that reverses input order."""
    with patch("neural_search.retrieval.reranker._get_model") as mock_get:
        model = MagicMock()
        model.predict = lambda pairs: np.array(
            [float(len(pairs) - i) for i in range(len(pairs))]
        )
        mock_get.return_value = model
        yield CrossEncoderReranker()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rerank_respects_top_k(reranker):
    assert len(reranker.rerank("query", _chunks(10), top_k=5)) == 5


def test_rerank_sorted_descending(reranker):
    results = reranker.rerank("query", _chunks(6), top_k=6)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_rank_is_1_based_sequential(reranker):
    results = reranker.rerank("query", _chunks(5), top_k=5)
    assert [r["rerank_rank"] for r in results] == [1, 2, 3, 4, 5]


def test_rerank_enriches_score_and_rank_fields(reranker):
    results = reranker.rerank("query", _chunks(3), top_k=3)
    for r in results:
        assert isinstance(r["rerank_score"], float)
        assert isinstance(r["rerank_rank"], int)
        assert "rerank_latency_ms" in r


def test_rerank_preserves_original_fields(reranker):
    results = reranker.rerank("query", _chunks(3), top_k=3)
    for r in results:
        for field in ("chunk_id", "text", "source_file", "page", "collection"):
            assert field in r


def test_rerank_empty_input_returns_empty(reranker):
    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_top_k_exceeds_candidates_returns_all(reranker):
    assert len(reranker.rerank("query", _chunks(3), top_k=20)) == 3


def test_rerank_default_top_k_returns_all(reranker):
    assert len(reranker.rerank("query", _chunks(7))) == 7
