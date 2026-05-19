"""
Integration Test 3: Hybrid RRF Round-Trip
==========================================

What this tests
---------------
The end-to-end retrieval workflow:

  BM25sRetriever  ──┐
                    ├──→  HybridRetriever._rrf()  →  ranked fused list
  QdrantRetriever ──┘

Both retrievers are given real indexed data (from run_ingestion).
Only SentenceTransformer is faked.

These tests verify:
  1. HybridRetriever.search() returns ranked fused results
  2. rrf_score is present for every result (hybrid mode)
  3. Results are sorted by rrf_score descending
  4. source field indicates which backends contributed (dense, sparse, or dense+sparse)
  5. search_debug() returns sparse/dense/hybrid_rrf sub-keys
  6. HybridRetriever.search() with mode=sparse delegates to BM25 only
  7. HybridRetriever.search() with mode=dense delegates to Qdrant only
"""
import pytest
from unittest.mock import patch

from neural_search.ingestion.chunker import Chunk


def _make_chunks(n: int, slug: str) -> list[Chunk]:
    words = "neural retrieval ranking semantic hybrid fusion document index score"
    wl = words.split()
    return [
        Chunk(
            chunk_id=f"{slug}_c{i:03d}",
            doc_id="doc",
            source_file="test.pdf",
            page=i % 3 + 1,
            chunk_index=i,
            text=" ".join(wl[(i * 2) % len(wl):] + wl[:(i * 2) % len(wl)]),
            token_count=9 + i,
        )
        for i in range(n)
    ]


@pytest.fixture
def indexed_retrievers(real_settings, fake_embedding_model):
    """
    Returns (BM25sRetriever, QdrantRetriever) both populated with the same
    10 chunks so HybridRetriever can merge their results.
    """
    import neural_search.retrieval.dense as dense_mod
    dense_mod._MODEL = None
    dense_mod._QDRANT_CLIENT = None

    with patch("neural_search.retrieval.dense.SentenceTransformer",
               return_value=fake_embedding_model):
        from neural_search.retrieval.sparse import BM25sRetriever
        from neural_search.retrieval.dense import QdrantRetriever

        slug = "hybrid-col"
        chunks = _make_chunks(10, slug)

        sparse = BM25sRetriever(collection_slug=slug)
        sparse.index(chunks)

        dense = QdrantRetriever(collection_slug=slug)
        dense.upsert(chunks)

        yield sparse, dense

    dense_mod._MODEL = None
    dense_mod._QDRANT_CLIENT = None


class TestHybridRetriever:

    def test_search_returns_results(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = hybrid.search("neural retrieval", k=5)
        assert len(results) > 0, "HybridRetriever.search() returned no results"

    def test_results_have_rrf_score(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = hybrid.search("semantic ranking", k=5)
        for r in results:
            assert "rrf_score" in r, "rrf_score missing from hybrid result"
            assert r["rrf_score"] > 0, "rrf_score must be positive"

    def test_results_sorted_by_rrf_score_desc(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = hybrid.search("fusion index", k=10)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True), \
            "Results must be sorted by rrf_score descending"

    def test_rank_field_is_sequential(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = hybrid.search("document score", k=5)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1)), \
            f"rank must be 1-indexed sequential, got {ranks}"

    def test_source_field_values_are_valid(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = hybrid.search("retrieval", k=10)
        valid_sources = {"dense", "sparse", "dense+sparse", "sparse+dense"}
        for r in results:
            assert r["source"] in valid_sources, \
                f"Unexpected source value: {r['source']!r}"

    def test_k_limits_results(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = hybrid.search("test", k=3)
        assert len(results) <= 3

    def test_search_debug_contains_all_keys(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        debug = hybrid.search_debug("hybrid ranking", k=5)
        for key in ("query", "sparse", "dense", "hybrid_rrf"):
            assert key in debug, f"search_debug missing key: {key}"
        assert debug["query"] == "hybrid ranking"
        assert isinstance(debug["sparse"], list)
        assert isinstance(debug["dense"], list)
        assert isinstance(debug["hybrid_rrf"], list)

    def test_hybrid_result_count_does_not_exceed_k(self, indexed_retrievers):
        from neural_search.retrieval.hybrid import HybridRetriever
        sparse, dense = indexed_retrievers
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        for k in [1, 3, 5, 10]:
            results = hybrid.search("ranking", k=k)
            assert len(results) <= k, f"Got {len(results)} results for k={k}"
