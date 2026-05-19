"""
Learned hybrid fusion using Logistic Regression.

Replaces static RRF with a model trained on labeled eval data.
Features: BM25 score, dense score, BM25 rank, dense rank, query length, chunk length.
Label: 1 if chunk is relevant, 0 otherwise.

Workflow:
    1. trainer = LearnedFusionTrainer(dataset, sparse_retriever, dense_retriever)
       trainer.train(collection="base", k=20)           # fits and saves model
    2. fusion = LearnedHybridFusion(sparse, dense)
       results = fusion.search(query, k=5)              # uses saved model

Model is persisted to data/learned_fusion/model.pkl.
Falls back to RRF if model file is not found (safe degradation).
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neural_search.config import get_settings
from neural_search.evaluation.dataset import EvalDataset

settings = get_settings()

_MODEL_DIR = settings.data_dir / "learned_fusion"
_MODEL_PATH = _MODEL_DIR / "model.pkl"
_SCALER_PATH = _MODEL_DIR / "scaler.pkl"


# ── Feature extraction ────────────────────────────────────────────────────────

@dataclass
class FusionFeatures:
    bm25_score: float
    dense_score: float
    bm25_rank: int
    dense_rank: int
    query_length: int       # token count
    chunk_length: int       # token count

    def to_array(self) -> list[float]:
        return [
            self.bm25_score,
            self.dense_score,
            self.bm25_rank,
            self.dense_rank,
            self.query_length,
            self.chunk_length,
        ]


def _extract_features(
    query: str,
    chunk: dict,
    sparse_rank_map: dict[str, int],
    dense_rank_map: dict[str, int],
    sparse_score_map: dict[str, float],
    dense_score_map: dict[str, float],
) -> FusionFeatures:
    cid = chunk["chunk_id"]
    return FusionFeatures(
        bm25_score=sparse_score_map.get(cid, 0.0),
        dense_score=dense_score_map.get(cid, 0.0),
        bm25_rank=sparse_rank_map.get(cid, 999),
        dense_rank=dense_rank_map.get(cid, 999),
        query_length=len(query.split()),
        chunk_length=chunk.get("token_count", len(chunk.get("text", "").split())),
    )


def _build_rank_score_maps(results: list[dict]) -> tuple[dict, dict]:
    rank_map = {r["chunk_id"]: i + 1 for i, r in enumerate(results)}
    score_map = {r["chunk_id"]: r.get("score", 0.0) for r in results}
    return rank_map, score_map


# ── Trainer ───────────────────────────────────────────────────────────────────

class LearnedFusionTrainer:
    """
    Trains a Logistic Regression model on labeled evaluation data.

    Each (query, chunk) pair becomes one training example.
    Label = 1 if chunk_id is in the relevant set for that query, else 0.
    """

    def __init__(self, dataset: EvalDataset, sparse, dense) -> None:
        self._dataset = dataset
        self._sparse = sparse
        self._dense = dense

    def train(self, collection: str, k: int = 20) -> "LearnedHybridFusion":
        queries = self._dataset.labeled_queries()
        if not queries:
            raise ValueError("No labeled queries in dataset — cannot train.")

        logger.info(f"Training learned fusion on {len(queries)} labeled queries (k={k})")

        X: list[list[float]] = []
        y: list[int] = []

        for q in queries:
            relevant = self._dataset.get_relevant(q.id)
            sparse_results = self._sparse.search(q.text, k=k)
            dense_results = self._dense.search(q.text, k=k)

            sparse_rank, sparse_score = _build_rank_score_maps(sparse_results)
            dense_rank, dense_score = _build_rank_score_maps(dense_results)

            # Union of all retrieved candidates
            seen: set[str] = set()
            all_candidates: list[dict] = []
            for r in sparse_results + dense_results:
                if r["chunk_id"] not in seen:
                    seen.add(r["chunk_id"])
                    all_candidates.append(r)

            for chunk in all_candidates:
                features = _extract_features(
                    q.text, chunk,
                    sparse_rank, dense_rank,
                    sparse_score, dense_score,
                )
                X.append(features.to_array())
                y.append(1 if chunk["chunk_id"] in relevant else 0)

        X_arr = np.array(X)
        y_arr = np.array(y)

        pos = int(y_arr.sum())
        neg = len(y_arr) - pos
        logger.info(f"Training set: {len(X_arr)} examples | {pos} positive, {neg} negative")

        if pos == 0:
            raise ValueError("All training labels are 0 — check relevance.json has real chunk IDs.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        model = LogisticRegression(
            class_weight="balanced",    # handles label imbalance
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_scaled, y_arr)

        train_acc = model.score(X_scaled, y_arr)
        logger.info(f"Training accuracy: {train_acc:.4f}")

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with _MODEL_PATH.open("wb") as f:
            pickle.dump(model, f)
        with _SCALER_PATH.open("wb") as f:
            pickle.dump(scaler, f)

        logger.info(f"Model saved → {_MODEL_PATH}")
        return LearnedHybridFusion(self._sparse, self._dense, model=model, scaler=scaler)


# ── Inference ─────────────────────────────────────────────────────────────────

class LearnedHybridFusion:
    """
    Hybrid retriever using a trained Logistic Regression scorer.
    Falls back to RRF if no model is found.
    """

    def __init__(self, sparse, dense, model=None, scaler=None) -> None:
        self._sparse = sparse
        self._dense = dense
        self._model: Optional[LogisticRegression] = model
        self._scaler: Optional[StandardScaler] = scaler
        self._loaded = False

        if self._model is None:
            self._load()

    def _load(self) -> None:
        if _MODEL_PATH.exists() and _SCALER_PATH.exists():
            with _MODEL_PATH.open("rb") as f:
                self._model = pickle.load(f)
            with _SCALER_PATH.open("rb") as f:
                self._scaler = pickle.load(f)
            self._loaded = True
            logger.info("Learned fusion model loaded from disk")
        else:
            logger.warning(
                f"No learned model found at {_MODEL_PATH} — "
                "falling back to RRF. Run: python scripts/train_fusion.py"
            )

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or settings.top_k
        candidate_k = max(k * 4, 20)    # retrieve wider pool for scoring

        t0 = time.perf_counter()
        sparse_results = self._sparse.search(query, k=candidate_k)
        dense_results = self._dense.search(query, k=candidate_k)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

        if not self.is_trained:
            # Graceful fallback to RRF
            logger.debug("Using RRF fallback (no trained model)")
            from neural_search.retrieval.hybrid import _rrf
            return _rrf([(sparse_results, 1.0), (dense_results, 1.0)])[:k]

        sparse_rank, sparse_score = _build_rank_score_maps(sparse_results)
        dense_rank, dense_score = _build_rank_score_maps(dense_results)

        seen: set[str] = set()
        candidates: list[dict] = []
        for r in sparse_results + dense_results:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                candidates.append(r)

        X = np.array([
            _extract_features(
                query, chunk,
                sparse_rank, dense_rank,
                sparse_score, dense_score,
            ).to_array()
            for chunk in candidates
        ])

        X_scaled = self._scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)[:, 1]   # P(relevant)

        t_score = time.perf_counter()
        scored = sorted(
            zip(proba, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        scoring_ms = round((time.perf_counter() - t_score) * 1000, 2)

        logger.debug(
            f"Learned fusion | candidates={len(candidates)} "
            f"retrieval={retrieval_ms}ms scoring={scoring_ms}ms"
        )

        results = []
        for rank, (score, chunk) in enumerate(scored[:k], start=1):
            results.append({
                **chunk,
                "score": round(float(score), 6),
                "rank": rank,
                "source": "learned",
            })

        return results
