"""
Unit tests for ingestion/chunker.py
Tests: chunk splitting, token count, chunk_id determinism, overlap, edge cases.
"""
import pytest
from neural_search.ingestion.parser import ParsedPage
from neural_search.ingestion.chunker import chunk_pages, Chunk, _make_chunk_id


@pytest.fixture
def sample_pages() -> list[ParsedPage]:
    return [
        ParsedPage(
            doc_id="doc1",
            source_file="doc1.pdf",
            page=1,
            text="Word " * 200,   # long enough to force multiple chunks
        ),
        ParsedPage(
            doc_id="doc1",
            source_file="doc1.pdf",
            page=2,
            text="Short page content.",
        ),
    ]


class TestChunkId:
    def test_chunk_id_is_deterministic(self):
        id1 = _make_chunk_id("file.pdf", 0)
        id2 = _make_chunk_id("file.pdf", 0)
        assert id1 == id2

    def test_different_inputs_produce_different_ids(self):
        assert _make_chunk_id("file.pdf", 0) != _make_chunk_id("file.pdf", 1)
        assert _make_chunk_id("a.pdf", 0) != _make_chunk_id("b.pdf", 0)

    def test_chunk_id_is_16_chars(self):
        cid = _make_chunk_id("file.pdf", 0)
        assert len(cid) == 16


class TestChunkPages:
    def test_returns_list_of_chunks(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_long_page_produces_multiple_chunks(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        page1_chunks = [c for c in chunks if c.page == 1]
        assert len(page1_chunks) > 1

    def test_all_chunks_have_token_count(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        for c in chunks:
            assert c.token_count > 0
            assert c.token_count == len(c.text.split())

    def test_chunk_indices_are_globally_sequential(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_preserved(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        for c in chunks:
            assert c.doc_id == "doc1"
            assert c.source_file == "doc1.pdf"

    def test_chunk_ids_are_unique(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_pages_returns_empty(self):
        pages = [ParsedPage(doc_id="x", source_file="x.pdf", page=1, text="")]
        chunks = chunk_pages(pages)
        assert chunks == []

    def test_text_not_empty_in_any_chunk(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        for c in chunks:
            assert len(c.text.strip()) > 0
