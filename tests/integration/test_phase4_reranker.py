"""
Integration tests for Phase 4 — reranker and learned mode.
Requires live API: ./run.sh start
"""
from __future__ import annotations

import pytest
import requests

API_BASE = "http://localhost:8000"
COLLECTION = "base"
QUERY = "What is an AI agent?"


def _api_up() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _has_chunks() -> bool:
    try:
        r = requests.get(f"{API_BASE}/collections/{COLLECTION}", timeout=2)
        return r.status_code == 200 and r.json().get("total_chunks", 0) > 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _api_up(), reason="API not running — ./run.sh start"),
    pytest.mark.skipif(not _has_chunks(), reason="No chunks — ingest documents first"),
]


# ── Reranker ──────────────────────────────────────────────────────────────────

def test_rerank_flag_activates_reranking():
    r = requests.post(f"{API_BASE}/search", json={
        "query": QUERY, "collection": COLLECTION,
        "k": 10, "rerank": True, "rerank_top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["reranked"] is True
    assert len(data["results"]) <= 5


def test_reranked_results_have_rerank_fields():
    r = requests.post(f"{API_BASE}/search", json={
        "query": QUERY, "collection": COLLECTION,
        "k": 10, "rerank": True, "rerank_top_k": 5,
    })
    for chunk in r.json()["results"]:
        assert chunk["rerank_score"] is not None
        assert chunk["rerank_rank"] is not None


def test_no_rerank_leaves_fields_null():
    r = requests.post(f"{API_BASE}/search", json={
        "query": QUERY, "collection": COLLECTION, "k": 5, "rerank": False,
    })
    data = r.json()
    assert data["reranked"] is False
    for chunk in data["results"]:
        assert chunk.get("rerank_score") is None
        assert chunk.get("rerank_rank") is None


def test_latency_breakdown_present_with_rerank():
    r = requests.post(f"{API_BASE}/search", json={
        "query": QUERY, "collection": COLLECTION,
        "k": 10, "rerank": True, "rerank_top_k": 5,
    })
    latency = r.json()["latency"]
    assert latency["retrieval_ms"] > 0
    assert latency["rerank_ms"] is not None
    assert latency["total_ms"] < 5000


def test_latency_rerank_ms_null_without_rerank():
    r = requests.post(f"{API_BASE}/search", json={
        "query": QUERY, "collection": COLLECTION, "k": 5,
    })
    assert r.json()["latency"]["rerank_ms"] is None


# ── Learned mode ──────────────────────────────────────────────────────────────

def test_learned_mode_returns_results():
    r = requests.post(f"{API_BASE}/search", json={
        "query": QUERY, "collection": COLLECTION, "k": 5, "mode": "learned",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "learned"
    assert len(data["results"]) > 0


def test_all_modes_return_consistent_contract():
    required = {"chunk_id", "score", "rank", "text", "source_file", "collection"}
    for mode in ("sparse", "dense", "hybrid", "learned"):
        r = requests.post(f"{API_BASE}/search", json={
            "query": QUERY, "collection": COLLECTION, "k": 5, "mode": mode,
        })
        assert r.status_code == 200, f"mode={mode} returned {r.status_code}"
        for chunk in r.json()["results"]:
            missing = required - set(chunk.keys())
            assert not missing, f"mode={mode} missing fields: {missing}"
