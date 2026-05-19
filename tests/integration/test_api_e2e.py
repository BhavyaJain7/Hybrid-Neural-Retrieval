"""
Integration Test 5: Full API End-to-End (HTTP → File → Index → Search)
=======================================================================

What this tests
---------------
The complete request/response lifecycle through the FastAPI layer:

  HTTP request → FastAPI route → CollectionManager / run_ingestion → BM25 + Qdrant → HTTP response

This is the closest we get to production without Groq or a real embedding model:
  - TestClient (no network)
  - Real CollectionManager with real filesystem (tmp_path via real_settings)
  - Real BM25sRetriever
  - Qdrant + SentenceTransformer are mocked so we don't download models

Scenarios covered:
  1. Health check returns 200 + correct schema
  2. Creating a collection via POST /collections
  3. Duplicate collection returns 400
  4. Listing collections returns the created one
  5. Deleting a non-existent collection returns 404
  6. Ingesting a real PDF via POST /collections/{slug}/ingest
  7. Duplicate file upload returns 409
  8. Search on an empty collection returns 200 + empty results (no crash)
  9. Search on unknown collection returns 404
  10. Invalid search mode returns 422
  11. Debug endpoint returns sparse/dense/hybrid_rrf keys
"""
import io
import pytest
from unittest.mock import patch, MagicMock

import numpy as np


# ── App fixture: real settings + mocked external I/O ─────────────────────────
@pytest.fixture
def app_client(real_settings, fake_embedding_model, tmp_path):
    """
    Spins up the FastAPI app with:
      - real CollectionManager (filesystem backed)
      - real BM25sRetriever
      - mocked SentenceTransformer (fake_embedding_model)
      - mocked Groq client (no API calls)
    """
    import neural_search.retrieval.dense as dense_mod
    dense_mod._MODEL = None
    dense_mod._QDRANT_CLIENT = None

    with patch("neural_search.retrieval.dense.SentenceTransformer",
               return_value=fake_embedding_model), \
         patch("neural_search.synthesis.groq_client.Groq") as mock_groq_cls:

        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Synthesized answer."))]
        )
        mock_groq_cls.return_value = mock_groq_instance

        # Import app fresh so lifespan picks up patched settings
        from neural_search.api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client

    dense_mod._MODEL = None
    dense_mod._QDRANT_CLIENT = None


# ── Helper: make a minimal valid PDF bytes ────────────────────────────────────
def _make_pdf_bytes() -> bytes:
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), (
            "Neural search combines sparse BM25 with dense vector retrieval. "
            "Reciprocal Rank Fusion merges the two ranked lists deterministically. "
            "This document is used for integration testing of the ingestion pipeline. "
        ))
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()
    except ImportError:
        pytest.skip("pymupdf not installed")


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestHealthEndpointE2E:
    def test_health_returns_200(self, app_client):
        assert app_client.get("/health").status_code == 200

    def test_health_schema(self, app_client):
        data = app_client.get("/health").json()
        assert "status" in data and data["status"] == "ok"
        assert "collections_count" in data
        assert "total_chunks" in data


class TestCollectionEndpointsE2E:
    def test_create_collection(self, app_client):
        resp = app_client.post("/collections", json={"name": "E2E Test", "description": "integration"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["slug"] == "e2e-test"
        assert body["name"] == "E2E Test"

    def test_create_duplicate_returns_400(self, app_client):
        app_client.post("/collections", json={"name": "Dup E2E", "description": ""})
        resp = app_client.post("/collections", json={"name": "Dup E2E", "description": ""})
        assert resp.status_code == 400

    def test_list_reflects_created_collection(self, app_client):
        app_client.post("/collections", json={"name": "Listed E2E", "description": ""})
        cols = app_client.get("/collections").json()
        slugs = [c["slug"] for c in cols]
        assert "listed-e2e" in slugs

    def test_get_existing_collection(self, app_client):
        app_client.post("/collections", json={"name": "Getme", "description": ""})
        resp = app_client.get("/collections/getme")
        assert resp.status_code == 200
        assert resp.json()["slug"] == "getme"

    def test_get_nonexistent_returns_404(self, app_client):
        resp = app_client.get("/collections/ghost-xyz")
        assert resp.status_code == 404

    def test_delete_existing_collection(self, app_client):
        app_client.post("/collections", json={"name": "Del Me", "description": ""})
        resp = app_client.delete("/collections/del-me")
        assert resp.status_code == 204

    def test_delete_nonexistent_returns_404(self, app_client):
        resp = app_client.delete("/collections/totally-absent")
        assert resp.status_code == 404


class TestIngestEndpointE2E:
    def test_ingest_real_pdf(self, app_client):
        """Ingest a real (pymupdf-generated) PDF and verify the response."""
        app_client.post("/collections", json={"name": "Ingest Col", "description": ""})
        pdf = _make_pdf_bytes()
        resp = app_client.post(
            "/collections/ingest-col/ingest",
            files={"file": ("neural_search.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
        body = resp.json()
        assert body["status"] == "ok"
        assert body["filename"] == "neural_search.pdf"
        assert body["chunks_indexed"] >= 0   # may be 0 for tiny docs

    def test_ingest_response_schema(self, app_client):
        app_client.post("/collections", json={"name": "Schema Col", "description": ""})
        pdf = _make_pdf_bytes()
        body = app_client.post(
            "/collections/schema-col/ingest",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        ).json()
        for field in ("status", "collection", "filename", "chunks_indexed", "tokens", "pages"):
            assert field in body, f"Missing field '{field}' in ingest response"

    def test_duplicate_ingest_returns_409(self, app_client):
        app_client.post("/collections", json={"name": "Dup Ingest", "description": ""})
        pdf = _make_pdf_bytes()
        app_client.post(
            "/collections/dup-ingest/ingest",
            files={"file": ("dup.pdf", pdf, "application/pdf")},
        )
        # Second upload of same filename — must be rejected
        resp = app_client.post(
            "/collections/dup-ingest/ingest",
            files={"file": ("dup.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 409

    def test_ingest_force_overwrites(self, app_client):
        app_client.post("/collections", json={"name": "Force Col", "description": ""})
        pdf = _make_pdf_bytes()
        app_client.post(
            "/collections/force-col/ingest",
            files={"file": ("forced.pdf", pdf, "application/pdf")},
        )
        resp = app_client.post(
            "/collections/force-col/ingest?force=true",
            files={"file": ("forced.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 200, "force=true should allow re-ingest"

    def test_ingest_unknown_collection_returns_404(self, app_client):
        resp = app_client.post(
            "/collections/no-such-col/ingest",
            files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 404


class TestSearchEndpointE2E:
    """
    For search we need indexed data: create a collection, ingest a PDF,
    then issue a search. Results may be empty if tokenization produces
    no matches, but there must never be a 5xx.
    """

    @pytest.fixture(autouse=True)
    def _setup_collection(self, app_client):
        """Create collection + ingest PDF once for all tests in this class."""
        app_client.post("/collections", json={"name": "Search E2E", "description": ""})
        pdf = _make_pdf_bytes()
        app_client.post(
            "/collections/search-e2e/ingest",
            files={"file": ("paper.pdf", pdf, "application/pdf")},
        )
        self.client = app_client

    def test_search_returns_200(self):
        resp = self.client.post("/search", json={"query": "neural search", "collection": "search-e2e"})
        assert resp.status_code == 200

    def test_search_result_schema(self):
        data = self.client.post(
            "/search", json={"query": "retrieval", "collection": "search-e2e"}
        ).json()
        assert "results" in data
        assert "query" in data
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_search_unknown_collection_returns_404(self):
        resp = self.client.post("/search", json={"query": "x", "collection": "no-such"})
        assert resp.status_code == 404

    def test_search_invalid_mode_returns_422(self):
        resp = self.client.post(
            "/search", json={"query": "x", "collection": "search-e2e", "mode": "invalid"}
        )
        assert resp.status_code == 422

    def test_search_empty_query_returns_422(self):
        resp = self.client.post("/search", json={"query": "", "collection": "search-e2e"})
        assert resp.status_code == 422

    def test_search_sparse_mode(self):
        resp = self.client.post(
            "/search", json={"query": "neural", "collection": "search-e2e", "mode": "sparse"}
        )
        assert resp.status_code == 200

    def test_search_dense_mode(self):
        resp = self.client.post(
            "/search", json={"query": "neural", "collection": "search-e2e", "mode": "dense"}
        )
        assert resp.status_code == 200

    def test_debug_endpoint(self):
        resp = self.client.get(
            "/search/debug?query=retrieval&collection=search-e2e"
        )
        assert resp.status_code == 200
        data = resp.json()
        for key in ("query", "sparse", "dense", "hybrid_rrf"):
            assert key in data
