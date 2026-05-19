"""
Integration Test 2: Dense Retrieval (Qdrant in-process) 
========================================================

What this tests
---------------
The encode → upsert → search round-trip for QdrantRetriever:

  Fake SentenceTransformer  →  QdrantRetriever.upsert()  →  QdrantRetriever.search()

Real components used:
  - qdrant-client (in-process, on-disk at tmp_path)
  - Real Chunk dataclass
  - Real tmp filesystem

Mocked:
  - SentenceTransformer (fake_embedding_model from conftest)
    → replaced via patch, avoids model download & GPU

These tests verify:
  1. upsert() stores chunks and search() returns them
  2. result schema matches what HybridRetriever/_rrf expect
  3. k parameter is respected
  4. count() reflects the number of upserted chunks
  5. reset() wipes and recreates the Qdrant collection
  6. Multiple collections are isolated from each other
"""
import pytest
from unittest.mock import patch

from neural_search.ingestion.chunker import Chunk


def _make_chunks(n: int, slug: str = "qdrant-col") -> list[Chunk]:
    words = "neural search retrieval hybrid dense sparse vector ranking fusion document"
    word_list = words.split()
    return [
        Chunk(
            chunk_id=f"chunk_{i:04d}",
            doc_id="test_doc",
            source_file="test.pdf",
            page=i % 3 + 1,
            chunk_index=i,
            text=" ".join(word_list[(i * 3) % len(word_list):] + word_list[:(i * 3) % len(word_list)]),
            token_count=10 + i,
        )
        for i in range(n)
    ]


class TestQdrantRetriever:
    """QdrantRetriever upsert → search round-trip."""

    @pytest.fixture
    def retriever(self, real_settings, fake_embedding_model):
        """Constructs a QdrantRetriever backed by in-process Qdrant at tmp_path."""
        # Reset module-level singletons so each test gets a clean client
        import neural_search.retrieval.dense as dense_mod
        dense_mod._MODEL = None
        dense_mod._QDRANT_CLIENT = None

        with patch("neural_search.retrieval.dense.SentenceTransformer",
                   return_value=fake_embedding_model):
            from neural_search.retrieval.dense import QdrantRetriever
            yield QdrantRetriever(collection_slug="qdrant-col")

        # Cleanup singletons after test
        dense_mod._MODEL = None
        dense_mod._QDRANT_CLIENT = None

    def test_upsert_and_search_returns_results(self, retriever):
        chunks = _make_chunks(5)
        retriever.upsert(chunks)
        results = retriever.search("neural search retrieval", k=3)
        assert len(results) > 0, "Expected at least one dense result"

    def test_result_schema(self, retriever):
        retriever.upsert(_make_chunks(5))
        results = retriever.search("hybrid vector ranking", k=3)
        assert results
        for r in results:
            for field in ("chunk_id", "score", "rank", "source", "text",
                          "source_file", "page", "token_count", "collection"):
                assert field in r, f"Missing field '{field}' in dense result"

    def test_k_limits_results(self, retriever):
        retriever.upsert(_make_chunks(10))
        results = retriever.search("search", k=3)
        assert len(results) <= 3, "Result count must not exceed k"

    def test_count_reflects_upserted_chunks(self, retriever):
        chunks = _make_chunks(7)
        retriever.upsert(chunks)
        assert retriever.count() == 7

    def test_source_field_is_dense(self, retriever):
        retriever.upsert(_make_chunks(3))
        results = retriever.search("test", k=3)
        for r in results:
            assert r["source"] == "dense", "source field must be 'dense' for QdrantRetriever"

    def test_reset_empties_collection(self, retriever):
        retriever.upsert(_make_chunks(5))
        assert retriever.count() == 5
        retriever.reset()
        assert retriever.count() == 0, "count() must be 0 after reset()"

    def test_two_collections_are_isolated(self, real_settings, fake_embedding_model):
        """Upsert into col-a and col-b; searching col-a must not surface col-b docs."""
        import neural_search.retrieval.dense as dense_mod
        dense_mod._MODEL = None
        dense_mod._QDRANT_CLIENT = None

        with patch("neural_search.retrieval.dense.SentenceTransformer",
                   return_value=fake_embedding_model):
            from neural_search.retrieval.dense import QdrantRetriever

            col_a = QdrantRetriever(collection_slug="col-a")
            col_b = QdrantRetriever(collection_slug="col-b")

            chunks_a = [Chunk(chunk_id="a001", doc_id="da", source_file="a.pdf",
                              page=1, chunk_index=0, text="alpha bravo charlie delta", token_count=4)]
            chunks_b = [Chunk(chunk_id="b001", doc_id="db", source_file="b.pdf",
                              page=1, chunk_index=0, text="x-ray yankee zulu", token_count=3)]

            col_a.upsert(chunks_a)
            col_b.upsert(chunks_b)

            results_a = col_a.search("alpha bravo", k=5)
            for r in results_a:
                assert r["collection"] == "col-a", \
                    f"Result from col-b leaked into col-a search: {r}"

        dense_mod._MODEL = None
        dense_mod._QDRANT_CLIENT = None
