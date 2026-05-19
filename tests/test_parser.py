"""
Unit tests for ingestion/parser.py
Tests: PDF parsing, DOCX parsing, unsupported types, empty pages, directory scan.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from neural_search.ingestion.parser import parse_document, parse_directory, ParsedPage


class TestPDFParser:
    def test_pdf_returns_parsed_pages(self, sample_pdf):
        pages = parse_document(sample_pdf)
        assert len(pages) > 0
        assert all(isinstance(p, ParsedPage) for p in pages)

    def test_pdf_page_numbers_are_sequential(self, sample_pdf):
        pages = parse_document(sample_pdf)
        page_nums = [p.page for p in pages]
        assert page_nums == sorted(page_nums)
        assert page_nums[0] == 1

    def test_pdf_metadata_populated(self, sample_pdf):
        pages = parse_document(sample_pdf)
        for p in pages:
            assert p.source_file == sample_pdf.name
            assert p.doc_id == sample_pdf.stem
            assert len(p.text) > 0

    def test_pdf_skips_empty_pages(self, tmp_path):
        """A PDF with only empty pages returns no ParsedPages."""
        import fitz
        doc = fitz.open()
        doc.new_page()   # blank page
        path = tmp_path / "empty.pdf"
        doc.save(str(path))
        doc.close()
        pages = parse_document(path)
        assert len(pages) == 0

    def test_corrupt_pdf_returns_empty_not_crash(self, tmp_path):
        path = tmp_path / "corrupt.pdf"
        path.write_bytes(b"not a pdf at all")
        pages = parse_document(path)
        assert pages == []


class TestDOCXParser:
    def test_docx_returns_parsed_sections(self, sample_docx):
        pages = parse_document(sample_docx)
        assert len(pages) > 0
        assert all(isinstance(p, ParsedPage) for p in pages)

    def test_docx_metadata_populated(self, sample_docx):
        pages = parse_document(sample_docx)
        for p in pages:
            assert p.source_file == sample_docx.name
            assert p.doc_id == sample_docx.stem
            assert len(p.text) > 0

    def test_docx_sections_split_on_headings(self, sample_docx):
        pages = parse_document(sample_docx)
        # sample_docx has 2 headings → expect at least 2 sections
        assert len(pages) >= 2

    def test_corrupt_docx_returns_empty_not_crash(self, tmp_path):
        path = tmp_path / "corrupt.docx"
        path.write_bytes(b"not a docx")
        pages = parse_document(path)
        assert pages == []


class TestUnsupportedTypes:
    def test_unsupported_extension_returns_empty(self, tmp_path):
        path = tmp_path / "file.csv"
        path.write_text("col1,col2\n1,2")
        pages = parse_document(path)
        assert pages == []

    def test_txt_file_returns_empty(self, tmp_path):
        path = tmp_path / "file.txt"
        path.write_text("some text")
        pages = parse_document(path)
        assert pages == []


class TestParseDirectory:
    def test_scans_mixed_directory(self, tmp_path, sample_pdf, sample_docx):
        import shutil
        shutil.copy(sample_pdf, tmp_path / "doc1.pdf")
        shutil.copy(sample_docx, tmp_path / "doc2.docx")
        pages = parse_directory(tmp_path)
        assert len(pages) > 0
        source_files = {p.source_file for p in pages}
        assert "doc1.pdf" in source_files
        assert "doc2.docx" in source_files

    def test_empty_directory_returns_empty(self, tmp_path):
        pages = parse_directory(tmp_path)
        assert pages == []

    def test_ignores_unsupported_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        pages = parse_directory(tmp_path)
        assert pages == []
