"""
Integration tests for the full ingestion pipeline.
Tests parse → chunk → index flow end-to-end with real file I/O.
Retrievers are mocked — no real BM25/Qdrant needed.
"""
import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, create_autospec
from neural_search.ingestion.pipeline import run_ingestion
from neural_search.ingestion.parser import parse_document
from neural_search.ingestion.chunker import chunk_pages
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever


class TestParseThenChunk:
    def test_pdf_parse_and_chunk_produces_chunks(self, sample_pdf):
        pages = parse_document(sample_pdf)
        assert len(pages) > 0
        chunks = chunk_pages(pages)
        assert len(chunks) > 0

    def test_docx_parse_and_chunk_produces_chunks(self, sample_docx):
        pages = parse_document(sample_docx)
        chunks = chunk_pages(pages)
        assert len(chunks) > 0

    def test_chunks_have_valid_metadata(self, sample_pdf):
        pages = parse_document(sample_pdf)
        chunks = chunk_pages(pages)
        for c in chunks:
            assert len(c.chunk_id) == 16
            assert c.source_file == sample_pdf.name
            assert c.token_count > 0
            assert len(c.text.strip()) > 0

    def test_chunk_ids_unique_across_documents(self, sample_pdf, sample_docx):
        all_chunks = (
            chunk_pages(parse_document(sample_pdf)) +
            chunk_pages(parse_document(sample_docx))
        )
        ids = [c.chunk_id for c in all_chunks]
        assert len(ids) == len(set(ids))


class TestRunIngestion:
    def test_ingestion_calls_sparse_index(self, sample_pdf):
        sparse = create_autospec(BM25sRetriever, instance=True)
        chunks = run_ingestion(source=sample_pdf, sparse_retriever=sparse)
        assert len(chunks) > 0
        sparse.index.assert_called_once()
        assert sparse.index.call_args[0][0] == chunks

    def test_ingestion_calls_dense_upsert(self, sample_pdf):
        dense = create_autospec(QdrantRetriever, instance=True)
        chunks = run_ingestion(source=sample_pdf, dense_retriever=dense)
        assert len(chunks) > 0
        dense.upsert.assert_called_once()

    def test_ingestion_with_both_retrievers(self, sample_pdf):
        sparse = create_autospec(BM25sRetriever, instance=True)
        dense  = create_autospec(QdrantRetriever, instance=True)
        chunks = run_ingestion(source=sample_pdf, sparse_retriever=sparse, dense_retriever=dense)
        assert len(chunks) > 0
        sparse.index.assert_called_once()
        dense.upsert.assert_called_once()

    def test_reset_wipes_before_indexing(self, sample_pdf):
        sparse = create_autospec(BM25sRetriever, instance=True)
        dense  = create_autospec(QdrantRetriever, instance=True)
        run_ingestion(source=sample_pdf, sparse_retriever=sparse, dense_retriever=dense, reset=True)
        sparse.reset.assert_called_once()
        dense.reset.assert_called_once()

    def test_nonexistent_path_returns_empty(self, tmp_path):
        chunks = run_ingestion(source=tmp_path / "ghost.pdf")
        assert chunks == []

    def test_exports_jsonl_snapshot(self, sample_pdf, tmp_path, patch_settings):
        # patch_settings is the autouse fixture from conftest — data_dir is already tmp_path/data
        patch_settings.data_dir = tmp_path / "data"
        snapshot_path = tmp_path / "data" / "snapshots" / "chunks.jsonl"

        chunks = run_ingestion(source=sample_pdf, export_snapshot=True)

        assert snapshot_path.exists(), f"Snapshot not found at {snapshot_path}"
        lines = snapshot_path.read_text().strip().splitlines()
        assert len(lines) == len(chunks)

        # Verify each line is valid JSON with required fields
        for line in lines:
            record = json.loads(line)
            for field in ("chunk_id", "source_file", "page", "text", "token_count"):
                assert field in record, f"Missing field '{field}' in snapshot record"

    def test_snapshot_disabled_creates_no_file(self, sample_pdf, tmp_path, patch_settings):
        patch_settings.data_dir = tmp_path / "data"
        snapshot_path = tmp_path / "data" / "snapshots" / "chunks.jsonl"
        run_ingestion(source=sample_pdf, export_snapshot=False)
        assert not snapshot_path.exists()

    def test_directory_ingestion_indexes_all_files(self, tmp_path, sample_pdf, sample_docx):
        shutil.copy(sample_pdf,  tmp_path / "doc1.pdf")
        shutil.copy(sample_docx, tmp_path / "doc2.docx")
        sparse = create_autospec(BM25sRetriever, instance=True)
        chunks = run_ingestion(source=tmp_path, sparse_retriever=sparse)
        assert len(chunks) > 0
        sparse.index.assert_called_once()
        # All chunks from both docs passed in one index call
        indexed = sparse.index.call_args[0][0]
        source_files = {c.source_file for c in indexed}
        assert "doc1.pdf" in source_files
        assert "doc2.docx" in source_files
