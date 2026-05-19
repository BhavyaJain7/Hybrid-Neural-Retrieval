"""Unit tests for LearnedHybridFusion and feature extraction (Phase 4)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_search.retrieval.learned import (
    LearnedFusionTrainer,
    LearnedHybridFusion,
    _build_rank_score_maps,
    _extract_features,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result(chunk_id: str, score: float, rank: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"Text for {chunk_id}",
        "source_file": "doc.pdf",
        "page": 1,
        "token_count": 20,
        "score": score,
        "rank": rank,
        "source": "sparse",
        "collection": "base",
    }


def _sparse_results() -> list[dict]:
    return [_result("c1", 0.9, 1), _result("c2", 0.7, 2), _result("c3", 0.5, 3)]


def _dense_results() -> list[dict]:
    return [_result("c2", 0.95, 1), _result("c1", 0.80, 2), _result("c4", 0.60, 3)]


@pytest.fixture()
def mock_sparse():
    m = MagicMock()
    m.search.return_value = _sparse_results()
    return m


@pytest.fixture()
def mock_dense():
    m = MagicMock()
    m.search.return_value = _dense_results()
    return m


@pytest.fixture()
def trained_fusion(mock_sparse, mock_dense):
    """LearnedHybridFusion with a mocked trained model."""
    model = MagicMock()
    scaler = MagicMock()
    model.predict_proba.return_value = np.array(
        [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.8, 0.2]]
    )
    scaler.transform.side_effect = lambda x: x
    return LearnedHybridFusion(
        sparse=mock_sparse, dense=mock_dense, model=model, scaler=scaler
    )


# ── _build_rank_score_maps ────────────────────────────────────────────────────

def test_rank_map_is_1_based():
    rank_map, _ = _build_rank_score_maps(_sparse_results())
    assert rank_map == {"c1": 1, "c2": 2, "c3": 3}


def test_score_map_values():
    _, score_map = _build_rank_score_maps(_sparse_results())
    assert score_map["c1"] == pytest.approx(0.9)


def test_empty_results_returns_empty_maps():
    assert _build_rank_score_maps([]) == ({}, {})


# ── _extract_features ─────────────────────────────────────────────────────────

def test_extract_features_values():
    chunk = _result("c1", 0.9, 1)
    f = _extract_features(
        "test query", chunk,
        {"c1": 1}, {"c1": 2},
        {"c1": 0.9}, {"c1": 0.8},
    )
    assert f.bm25_score == pytest.approx(0.9)
    assert f.dense_score == pytest.approx(0.8)
    assert f.bm25_rank == 1
    assert f.dense_rank == 2
    assert f.query_length == 2
    assert f.chunk_length == 20


def test_extract_features_missing_chunk_defaults_to_999():
    f = _extract_features("q", _result("unknown", 0, 1), {}, {}, {}, {})
    assert f.bm25_rank == 999
    assert f.dense_rank == 999
    assert f.bm25_score == pytest.approx(0.0)


def test_feature_array_has_6_elements():
    f = _extract_features("q", _result("c1", 0.9, 1), {"c1": 1}, {"c1": 1}, {"c1": 0.9}, {"c1": 0.8})
    assert len(f.to_array()) == 6


# ── LearnedHybridFusion ───────────────────────────────────────────────────────

def test_no_model_file_is_not_trained(mock_sparse, mock_dense):
    with patch("neural_search.retrieval.learned._MODEL_PATH") as mp, \
         patch("neural_search.retrieval.learned._SCALER_PATH") as sp:
        mp.exists.return_value = False
        sp.exists.return_value = False
        fusion = LearnedHybridFusion(sparse=mock_sparse, dense=mock_dense)
        assert not fusion.is_trained


def test_untrained_falls_back_to_rrf(mock_sparse, mock_dense):
    with patch("neural_search.retrieval.learned._MODEL_PATH") as mp, \
         patch("neural_search.retrieval.learned._SCALER_PATH") as sp:
        mp.exists.return_value = False
        sp.exists.return_value = False
        fusion = LearnedHybridFusion(sparse=mock_sparse, dense=mock_dense)
        results = fusion.search("query", k=3)
        assert isinstance(results, list)


def test_trained_search_returns_correct_count(trained_fusion):
    assert len(trained_fusion.search("query", k=3)) == 3


def test_trained_search_scores_descending(trained_fusion):
    results = trained_fusion.search("query", k=4)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_trained_search_source_is_learned(trained_fusion):
    for r in trained_fusion.search("query", k=2):
        assert r["source"] == "learned"


def test_trained_search_result_has_required_fields(trained_fusion):
    for r in trained_fusion.search("query", k=2):
        for field in ("chunk_id", "score", "rank", "text"):
            assert field in r


# ── LearnedFusionTrainer ──────────────────────────────────────────────────────

def test_trainer_raises_on_no_labeled_queries(mock_sparse, mock_dense):
    dataset = MagicMock()
    dataset.labeled_queries.return_value = []
    with pytest.raises(ValueError, match="No labeled queries"):
        LearnedFusionTrainer(dataset, mock_sparse, mock_dense).train("base")


def test_trainer_raises_when_all_labels_negative(mock_sparse, mock_dense):
    from neural_search.evaluation.dataset import EvalQuery
    dataset = MagicMock()
    dataset.labeled_queries.return_value = [EvalQuery(id="q1", text="query", type="semantic")]
    dataset.get_relevant.return_value = {"nonexistent"}
    with pytest.raises(ValueError, match="All training labels are 0"):
        LearnedFusionTrainer(dataset, mock_sparse, mock_dense).train("base", k=5)


def test_trainer_returns_trained_fusion(mock_sparse, mock_dense, tmp_path):
    from neural_search.evaluation.dataset import EvalQuery
    dataset = MagicMock()
    dataset.labeled_queries.return_value = [EvalQuery(id="q1", text="query", type="semantic")]
    dataset.get_relevant.return_value = {"c1"}

    with patch("neural_search.retrieval.learned._MODEL_DIR", tmp_path), \
         patch("neural_search.retrieval.learned._MODEL_PATH", tmp_path / "model.pkl"), \
         patch("neural_search.retrieval.learned._SCALER_PATH", tmp_path / "scaler.pkl"):
        fusion = LearnedFusionTrainer(dataset, mock_sparse, mock_dense).train("base", k=5)
        assert isinstance(fusion, LearnedHybridFusion)
        assert fusion.is_trained
