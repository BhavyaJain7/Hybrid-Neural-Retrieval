"""
Unit tests for retrieval/hybrid.py
Tests: RRF score calculation, source attribution, ranking order, edge cases.
"""
import pytest
from unittest.mock import MagicMock, create_autospec
from neural_search.retrieval.hybrid import HybridRetriever, _rrf
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever


def make_result(chunk_id: str, rank: int, source: str, score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "score": score,
        "source": source,
        "text": f"text for {chunk_id}",
        "source_file": "test.pdf",
        "page": 1,
        "token_count": 10,
        "collection": "test",
    }


def make_sparse_mock(results: list[dict]) -> MagicMock:
    mock = create_autospec(BM25sRetriever, instance=True)
    mock.search.return_value = results
    return mock


def make_dense_mock(results: list[dict]) -> MagicMock:
    mock = create_autospec(QdrantRetriever, instance=True)
    mock.search.return_value = results
    return mock


class TestRRF:
    def test_chunk_in_both_gets_higher_score_than_single(self):
        sparse = [make_result("a", 1, "sparse"), make_result("b", 2, "sparse")]
        dense  = [make_result("a", 1, "dense"),  make_result("c", 2, "dense")]
        fused  = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=60)
        # "a" is in both lists — must outrank "b" and "c" which appear in only one
        assert fused[0]["chunk_id"] == "a"

    def test_source_attribution_both(self):
        sparse = [make_result("a", 1, "sparse")]
        dense  = [make_result("a", 1, "dense")]
        fused  = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=60)
        assert fused[0]["source"] in ("dense+sparse", "sparse+dense")

    def test_source_attribution_sparse_only(self):
        sparse = [make_result("x", 1, "sparse")]
        dense  = [make_result("y", 1, "dense")]
        fused  = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=60)
        by_id  = {r["chunk_id"]: r["source"] for r in fused}
        assert by_id["x"] == "sparse"
        assert by_id["y"] == "dense"

    def test_scores_are_strictly_descending(self):
        sparse = [make_result(f"s{i}", i + 1, "sparse") for i in range(5)]
        dense  = [make_result(f"d{i}", i + 1, "dense")  for i in range(5)]
        fused  = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=60)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_is_one_indexed_and_sequential(self):
        sparse = [make_result("a", 1, "sparse")]
        dense  = [make_result("b", 1, "dense")]
        fused  = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=60)
        assert [r["rank"] for r in fused] == list(range(1, len(fused) + 1))

    def test_empty_inputs_returns_empty_list(self):
        assert _rrf([([], 1.0), ([], 1.0)], rrf_k=60) == []

    def test_sparse_only_input(self):
        sparse = [make_result("a", 1, "sparse")]
        fused  = _rrf([(sparse, 1.0), ([], 1.0)], rrf_k=60)
        assert len(fused) == 1
        assert fused[0]["chunk_id"] == "a"

    def test_higher_k_produces_lower_rrf_score(self):
        sparse = [make_result("a", 1, "sparse")]
        dense  = [make_result("a", 1, "dense")]
        score_k1   = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=1)[0]["rrf_score"]
        score_k100 = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=100)[0]["rrf_score"]
        assert score_k1 > score_k100

    def test_rrf_score_present_in_all_results(self):
        sparse = [make_result("a", 1, "sparse")]
        dense  = [make_result("b", 1, "dense")]
        fused  = _rrf([(sparse, 1.0), (dense, 1.0)], rrf_k=60)
        for r in fused:
            assert "rrf_score" in r
            assert isinstance(r["rrf_score"], float)


class TestHybridRetriever:
    def test_search_calls_both_retrievers(self):
        sparse = make_sparse_mock([make_result("a", 1, "sparse")])
        dense  = make_dense_mock([make_result("b", 1, "dense")])
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        hybrid.search("test query", k=5)
        sparse.search.assert_called_once_with("test query", k=5)
        dense.search.assert_called_once_with("test query", k=5)

    def test_search_returns_merged_results(self):
        sparse = make_sparse_mock([make_result("a", 1, "sparse")])
        dense  = make_dense_mock([make_result("b", 1, "dense")])
        results = HybridRetriever(sparse=sparse, dense=dense).search("q", k=5)
        chunk_ids = {r["chunk_id"] for r in results}
        assert "a" in chunk_ids
        assert "b" in chunk_ids

    def test_search_respects_k_limit(self):
        sparse = make_sparse_mock([make_result(f"s{i}", i + 1, "sparse") for i in range(10)])
        dense  = make_dense_mock([make_result(f"d{i}", i + 1, "dense")  for i in range(10)])
        results = HybridRetriever(sparse=sparse, dense=dense).search("q", k=3)
        assert len(results) <= 3

    def test_search_result_has_rrf_score(self):
        sparse = make_sparse_mock([make_result("a", 1, "sparse")])
        dense  = make_dense_mock([make_result("a", 1, "dense")])
        results = HybridRetriever(sparse=sparse, dense=dense).search("q", k=5)
        assert all("rrf_score" in r for r in results)

    def test_search_debug_returns_expected_keys(self):
        sparse = make_sparse_mock([make_result("a", 1, "sparse")])
        dense  = make_dense_mock([make_result("b", 1, "dense")])
        debug  = HybridRetriever(sparse=sparse, dense=dense).search_debug("q", k=5)
        assert "sparse"     in debug
        assert "dense"      in debug
        assert "hybrid_rrf" in debug
        assert "web"        in debug

    def test_both_retrievers_return_empty_gives_empty(self):
        sparse = make_sparse_mock([])
        dense  = make_dense_mock([])
        results = HybridRetriever(sparse=sparse, dense=dense).search("q", k=5)
        assert results == []
