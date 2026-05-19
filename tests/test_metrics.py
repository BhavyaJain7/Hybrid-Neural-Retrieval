"""
Unit tests for evaluation/metrics.py
Tests: P@K, Recall@K, MRR, nDCG edge cases and correctness.
"""
import pytest
from neural_search.evaluation.metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, evaluate_run
)


RESULTS   = ["a", "b", "c", "d", "e"]
RELEVANT  = {"a", "c", "e"}        # ranks 1, 3, 5 are relevant


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "c", "e"], {"a", "c", "e"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["b", "d"], {"a", "c"}, k=2) == 0.0

    def test_partial(self):
        # 2 of 4 relevant
        assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == pytest.approx(0.5)

    def test_k_larger_than_results(self):
        # Only 2 results but k=5 — score based on k not len(results)
        assert precision_at_k(["a", "b"], {"a"}, k=5) == pytest.approx(1 / 5)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(RESULTS, RELEVANT, k=0) == 0.0


class TestRecallAtK:
    def test_all_retrieved(self):
        assert recall_at_k(["a", "c", "e"], {"a", "c", "e"}, k=3) == 1.0

    def test_none_retrieved(self):
        assert recall_at_k(["b", "d"], {"a", "c"}, k=2) == 0.0

    def test_partial(self):
        # 1 of 3 relevant retrieved in top 2
        assert recall_at_k(["a", "b"], {"a", "c", "e"}, k=2) == pytest.approx(1 / 3)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(RESULTS, set(), k=5) == 0.0


class TestMRR:
    def test_first_result_relevant(self):
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert mrr(["b", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        assert mrr(["b", "c", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_returns_zero(self):
        assert mrr(["a", "b", "c"], {"z"}) == 0.0

    def test_multiple_relevant_uses_first(self):
        # first relevant is at rank 2
        assert mrr(["x", "a", "c"], {"a", "c"}) == pytest.approx(0.5)


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # All relevant at top
        assert ndcg_at_k(["a", "c", "e"], {"a", "c", "e"}, k=3) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k(["b", "d"], {"a", "c"}, k=2) == 0.0

    def test_worst_ranking_is_less_than_perfect(self):
        # relevant at bottom vs top
        best = ndcg_at_k(["a", "c", "e", "b", "d"], {"a", "c", "e"}, k=5)
        worst = ndcg_at_k(["b", "d", "a", "c", "e"], {"a", "c", "e"}, k=5)
        assert best > worst

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


class TestEvaluateRun:
    def test_returns_all_metrics(self):
        metrics = evaluate_run(RESULTS, RELEVANT, k=5)
        assert "precision@5" in metrics
        assert "recall@5" in metrics
        assert "mrr" in metrics
        assert "ndcg@5" in metrics

    def test_all_values_between_zero_and_one(self):
        metrics = evaluate_run(RESULTS, RELEVANT, k=5)
        for val in metrics.values():
            assert 0.0 <= val <= 1.0

    def test_perfect_run(self):
        results = ["a", "c", "e"]
        relevant = {"a", "c", "e"}
        metrics = evaluate_run(results, relevant, k=3)
        assert metrics["precision@3"] == 1.0
        assert metrics["recall@3"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["ndcg@3"] == pytest.approx(1.0)
