"""
Evaluation metrics: Precision@K, Recall@K, MRR, nDCG.
All functions accept:
  results   — ordered list of chunk_ids returned by retriever
  relevant  — set of chunk_ids marked relevant for this query
"""
import math


def precision_at_k(results: list[str], relevant: set[str], k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def mrr(results: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(results, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[str], relevant: set[str], k: int) -> float:
    def dcg(hits: list[int]) -> float:
        return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

    top_k = results[:k]
    hits = [1 if cid in relevant else 0 for cid in top_k]
    ideal = sorted(hits, reverse=True)
    actual_dcg = dcg(hits)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_run(
    results: list[str],
    relevant: set[str],
    k: int = 10,
) -> dict:
    return {
        f"precision@{k}": round(precision_at_k(results, relevant, k), 4),
        f"recall@{k}": round(recall_at_k(results, relevant, k), 4),
        "mrr": round(mrr(results, relevant), 4),
        f"ndcg@{k}": round(ndcg_at_k(results, relevant, k), 4),
    }
