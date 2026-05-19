"""
Integration Test 1: Ingestion Pipeline (Parse → Chunk → BM25 Index)
=====================================================================

What this tests
---------------
The full parse-to-index journey for the BM25 (sparse) path:

  Parser  →  Chunker  →  BM25sRetriever.index()  →  BM25sRetriever.search()

Real components used:
  - pymupdf / python-docx parsers
  - LangChain TokenTextSplitter chunker
  - bm25s BM25 index (on-disk pickle)
  - Real tmp filesystem (pytest tmp_path)

Mocked / skipped:
  - QdrantRetriever (no dense_retriever passed to run_ingestion)
  - Settings (real, but pointing at tmp dirs via conftest.real_settings)

These tests verify:
  1. run_ingestion() produces non-empty chunks for a valid PDF/DOCX
  2. chunks are written to the JSONL snapshot
  3. BM25 index persists to disk and can be reloaded in a fresh instance
  4. A keyword query returns a non-empty ranked result list
  5. ranking is consistent across repeated calls (determinism)
  6. An empty/invalid source returns [] without raising
"""
import pytest
from pathlib import Path

# ── Fixtures are inherited from integration/conftest.py ───────────────────────


class TestIngestionToBM25:
    """pipeline.run_ingestion() → BM25sRetriever round-trip."""

    def test_pdf_produces_chunks(self, small_pdf, real_settings):
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        sparse = BM25sRetriever(collection_slug="test-col")
        chunks = run_ingestion(
            source=small_pdf,
            sparse_retriever=sparse,
            dense_retriever=None,
            collection_slug="test-col",
            settings_obj=real_settings,
        )
        assert len(chunks) > 0, "Expected at least one chunk from a 3-page PDF"

    def test_docx_produces_chunks(self, small_docx, real_settings):
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        sparse = BM25sRetriever(collection_slug="test-col")
        chunks = run_ingestion(
            source=small_docx,
            sparse_retriever=sparse,
            dense_retriever=None,
            collection_slug="test-col",
            settings_obj=real_settings,
        )
        assert len(chunks) > 0, "Expected at least one chunk from a DOCX"

    def test_snapshot_jsonl_is_written(self, small_pdf, real_settings):
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever
        import json

        sparse = BM25sRetriever(collection_slug="snap-col")
        run_ingestion(
            source=small_pdf,
            sparse_retriever=sparse,
            dense_retriever=None,
            export_snapshot=True,
            collection_slug="snap-col",
            settings_obj=real_settings,
        )
        snap = real_settings.snapshot_path_for("snap-col")
        assert snap.exists(), "JSONL snapshot was not written to disk"

        lines = snap.read_text().strip().splitlines()
        assert len(lines) > 0
        record = json.loads(lines[0])
        for field in ("chunk_id", "doc_id", "source_file", "page", "text"):
            assert field in record, f"Missing field '{field}' in snapshot"

    def test_bm25_index_persists_to_disk(self, small_pdf, real_settings):
        """Index once, then reload in a fresh BM25sRetriever instance."""
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        slug = "persist-col"
        sparse1 = BM25sRetriever(collection_slug=slug)
        chunks = run_ingestion(
            source=small_pdf,
            sparse_retriever=sparse1,
            dense_retriever=None,
            collection_slug=slug,
            settings_obj=real_settings,
        )
        assert len(chunks) > 0

        # Fresh instance — must load from disk
        sparse2 = BM25sRetriever(collection_slug=slug)
        loaded = sparse2.load()
        assert loaded, "BM25sRetriever.load() returned False — index files missing"
        assert sparse2.count() == len(chunks)

    def test_keyword_search_returns_results(self, small_pdf, real_settings):
        """After indexing, a relevant keyword query must return ranked results."""
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        slug = "search-col"
        sparse = BM25sRetriever(collection_slug=slug)
        run_ingestion(
            source=small_pdf,
            sparse_retriever=sparse,
            dense_retriever=None,
            collection_slug=slug,
            settings_obj=real_settings,
        )

        results = sparse.search("neural search retrieval", k=3)
        assert len(results) > 0, "Expected at least one BM25 result"

    def test_search_result_schema(self, small_pdf, real_settings):
        """Every result dict must contain the fields that HybridRetriever and API expect."""
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        slug = "schema-col"
        sparse = BM25sRetriever(collection_slug=slug)
        run_ingestion(
            source=small_pdf,
            sparse_retriever=sparse,
            dense_retriever=None,
            collection_slug=slug,
            settings_obj=real_settings,
        )

        results = sparse.search("hybrid ranking", k=3)
        assert results, "No results returned"
        for r in results:
            for field in ("chunk_id", "score", "rank", "source", "text",
                          "source_file", "page", "token_count", "collection"):
                assert field in r, f"Missing field '{field}' in BM25 result"

    def test_search_ranking_is_deterministic(self, small_pdf, real_settings):
        """Two consecutive searches with the same query must return the same order."""
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        slug = "det-col"
        sparse = BM25sRetriever(collection_slug=slug)
        run_ingestion(
            source=small_pdf,
            sparse_retriever=sparse,
            dense_retriever=None,
            collection_slug=slug,
            settings_obj=real_settings,
        )

        r1 = sparse.search("retrieval ranking fusion", k=5)
        r2 = sparse.search("retrieval ranking fusion", k=5)
        assert [x["chunk_id"] for x in r1] == [x["chunk_id"] for x in r2]

    def test_missing_source_returns_empty(self, tmp_path, real_settings):
        """run_ingestion over a non-existent path must return [] not raise."""
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        bad_path = tmp_path / "does_not_exist.pdf"
        sparse = BM25sRetriever(collection_slug="empty-col")
        result = run_ingestion(
            source=bad_path,
            sparse_retriever=sparse,
            dense_retriever=None,
            collection_slug="empty-col",
            settings_obj=real_settings,
        )
        assert result == [], "Expected empty list for missing source file"

    def test_reset_clears_bm25_index(self, small_pdf, real_settings):
        """After reset(), search must return [] because the index is gone."""
        from neural_search.ingestion.pipeline import run_ingestion
        from neural_search.retrieval.sparse import BM25sRetriever

        slug = "reset-col"
        sparse = BM25sRetriever(collection_slug=slug)
        run_ingestion(
            source=small_pdf,
            sparse_retriever=sparse,
            dense_retriever=None,
            collection_slug=slug,
            settings_obj=real_settings,
        )
        assert sparse.count() > 0

        sparse.reset()
        assert sparse.count() == 0, "BM25 count should be 0 after reset"
        results = sparse.search("test", k=5)
        assert results == [], "search() should return [] after reset"
