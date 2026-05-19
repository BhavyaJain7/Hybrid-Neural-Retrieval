"""
Integration tests for the FastAPI layer.
Uses TestClient — no real network, no real indexes (all mocked).
Requires: python-multipart (for file upload endpoints).
"""
import pytest
from unittest.mock import MagicMock, patch, create_autospec
from fastapi.testclient import TestClient
from neural_search.collections.manager import CollectionManager
from neural_search.retrieval.hybrid import HybridRetriever


# ── Shared collection payload ──────────────────────────────────────────────────
COLLECTION = {
    "slug": "hr-policies",
    "name": "HR Policies",
    "description": "HR documents",
    "created_at": "2026-04-20T10:00:00+00:00",
    "updated_at": "2026-04-20T10:00:00+00:00",
    "files": [],
    "total_chunks": 42,
    "total_tokens": 5000,
}

SEARCH_RESULT = {
    "chunk_id": "abc123",
    "score": 0.95,
    "rank": 1,
    "source": "dense+sparse",
    "text": "Payment terms are net 30 days.",
    "source_file": "contract.pdf",
    "page": 3,
    "token_count": 8,
    "collection": "hr-policies",
    "rrf_score": 0.016,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_collection_manager():
    manager = create_autospec(CollectionManager, instance=True)
    manager.list_collections.return_value = [COLLECTION]
    manager.get_collection.return_value = COLLECTION
    manager.create_collection.return_value = COLLECTION
    manager.file_exists.return_value = False
    return manager


@pytest.fixture
def mock_hybrid():
    hybrid = create_autospec(HybridRetriever, instance=True)
    hybrid._sparse = MagicMock()
    hybrid._dense  = MagicMock()
    hybrid.search.return_value = [SEARCH_RESULT]
    hybrid.search_debug.return_value = {
        "query": "payment terms",
        "collection": "hr-policies",
        "sparse": [SEARCH_RESULT],
        "dense": [SEARCH_RESULT],
        "hybrid_rrf": [SEARCH_RESULT],
    }
    return hybrid


@pytest.fixture
def mock_synthesizer():
    synth = MagicMock()
    synth.synthesize.return_value = {
        "answer": "Payment terms are net 30 days.",
        "sources_used": [{"source_file": "contract.pdf", "page": 3}],
        "model": "llama-3.1-8b-instant",
    }
    return synth


@pytest.fixture
def client(mock_collection_manager, mock_hybrid, mock_synthesizer):
    with patch("neural_search.api.routes.collection_manager", mock_collection_manager), \
         patch("neural_search.api.routes._get_hybrid", return_value=mock_hybrid), \
         patch("neural_search.synthesis.groq_client.Groq"), \
         patch("neural_search.api.main.GroqSynthesizer", return_value=mock_synthesizer):
        from neural_search.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            # lifespan sets app.state.synthesizer via GroqSynthesizer() — patched above
            yield c


# ── Health ─────────────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_has_collections_count(self, client):
        data = client.get("/health").json()
        assert "collections_count" in data
        assert isinstance(data["collections_count"], int)

    def test_has_total_chunks(self, client):
        data = client.get("/health").json()
        assert "total_chunks" in data
        assert isinstance(data["total_chunks"], int)

    def test_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


# ── Collections ────────────────────────────────────────────────────────────────
class TestCollectionEndpoints:
    def test_list_returns_200_and_list(self, client):
        resp = client.get("/collections")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_contains_collection_fields(self, client):
        col = client.get("/collections").json()[0]
        for field in ("slug", "name", "total_chunks", "files"):
            assert field in col

    def test_create_returns_201(self, client):
        resp = client.post("/collections", json={"name": "Test", "description": ""})
        assert resp.status_code == 201

    def test_create_missing_name_returns_422(self, client):
        resp = client.post("/collections", json={"description": "no name"})
        assert resp.status_code == 422

    def test_get_existing_returns_200(self, client):
        resp = client.get("/collections/hr-policies")
        assert resp.status_code == 200
        assert resp.json()["slug"] == "hr-policies"

    def test_get_nonexistent_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.get("/collections/ghost")
        assert resp.status_code == 404

    def test_delete_existing_returns_204(self, client):
        resp = client.delete("/collections/hr-policies")
        assert resp.status_code == 204

    def test_delete_calls_manager(self, client, mock_collection_manager):
        client.delete("/collections/hr-policies")
        mock_collection_manager.delete_collection.assert_called_once_with("hr-policies")


# ── Search ─────────────────────────────────────────────────────────────────────
class TestSearchEndpoint:
    BASE_PAYLOAD = {"query": "payment terms", "collection": "hr-policies"}

    def test_returns_200(self, client):
        assert client.post("/search", json=self.BASE_PAYLOAD).status_code == 200

    def test_returns_results_list(self, client):
        data = client.post("/search", json=self.BASE_PAYLOAD).json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_result_has_required_fields(self, client):
        result = client.post("/search", json=self.BASE_PAYLOAD).json()["results"][0]
        for field in ("chunk_id", "source_file", "page", "text", "score", "rank", "source"):
            assert field in result, f"Missing field: {field}"

    def test_synthesis_present_when_requested(self, client):
        data = client.post("/search", json={**self.BASE_PAYLOAD, "synthesize": True}).json()
        assert data["synthesis"] is not None
        assert "answer" in data["synthesis"]
        assert "sources_used" in data["synthesis"]

    def test_synthesis_absent_when_not_requested(self, client):
        data = client.post("/search", json={**self.BASE_PAYLOAD, "synthesize": False}).json()
        assert data["synthesis"] is None

    def test_latency_ms_present_and_non_negative(self, client):
        data = client.post("/search", json=self.BASE_PAYLOAD).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_collection_echoed_in_response(self, client):
        data = client.post("/search", json=self.BASE_PAYLOAD).json()
        assert data["collection"] == "hr-policies"

    def test_nonexistent_collection_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.post("/search", json={**self.BASE_PAYLOAD, "collection": "ghost"})
        assert resp.status_code == 404

    def test_invalid_mode_returns_422(self, client):
        resp = client.post("/search", json={**self.BASE_PAYLOAD, "mode": "bad_mode"})
        assert resp.status_code == 422

    def test_empty_query_returns_422(self, client):
        resp = client.post("/search", json={"query": "", "collection": "hr-policies"})
        assert resp.status_code == 422

    def test_k_out_of_range_returns_422(self, client):
        resp = client.post("/search", json={**self.BASE_PAYLOAD, "k": 0})
        assert resp.status_code == 422


# ── Debug ──────────────────────────────────────────────────────────────────────
class TestDebugEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/search/debug?query=test&collection=hr-policies")
        assert resp.status_code == 200

    def test_returns_sparse_dense_hybrid(self, client):
        data = client.get("/search/debug?query=test&collection=hr-policies").json()
        assert "sparse"     in data
        assert "dense"      in data
        assert "hybrid_rrf" in data

    def test_nonexistent_collection_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.get("/search/debug?query=test&collection=ghost")
        assert resp.status_code == 404

    def test_missing_query_param_returns_422(self, client):
        resp = client.get("/search/debug?collection=hr-policies")
        assert resp.status_code == 422


# ── Ingest ─────────────────────────────────────────────────────────────────────
class TestIngestEndpoint:
    def test_duplicate_file_returns_409(self, client, mock_collection_manager):
        mock_collection_manager.file_exists.return_value = True
        resp = client.post(
            "/collections/hr-policies/ingest",
            files={"file": ("existing.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 409

    def test_nonexistent_collection_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.post(
            "/collections/ghost/ingest",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 404

    def test_valid_upload_is_not_5xx(self, client, tmp_path):
        """Real parsing may fail on fake bytes — acceptable. Must not be a server error."""
        resp = client.post(
            "/collections/hr-policies/ingest",
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code < 500
