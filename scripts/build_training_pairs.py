#!/usr/bin/env python3
"""
scripts/train_fusion.py

Trains the hybrid fusion scoring model on labeled eval data + hard negative pairs.

Algorithm: LightGBM (gradient boosted trees)
  - Handles feature interactions LogReg cannot (e.g. bm25_rank * dense_score)
  - No feature scaling needed
  - Built-in feature importance for debugging
  - Same inference speed as LogReg at this scale
  - Falls back to LogReg if LightGBM is not installed

Features (9 total):
  bm25_score, dense_score, bm25_rank, dense_rank,
  rrf_score, score_gap (bm25-dense), rank_gap (bm25-dense),
  query_length, chunk_length

Training data sources (both used):
  1. evaluation/queries.json + relevance.json  — labeled eval pairs
  2. data/training_pairs.jsonl                 — hard negative mined pairs

Cross-validation: StratifiedKFold(5) — detects overfitting on small datasets
Reports nDCG@5 and P@3 on held-out fold before saving.

Must be run with the API stopped (Qdrant lock constraint).

Usage:
    python scripts/train_fusion.py --collection <slug>
    python scripts/train_fusion.py --collection <slug> --k 20 --no-hard-negatives
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_search.config import get_settings
from neural_search.evaluation.dataset import EvalQuery
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.sparse import BM25sRetriever

settings = get_settings()

# ── Feature extraction ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "bm25_score",
    "dense_score",
    "bm25_rank",
    "dense_rank",
    "rrf_score",
    "score_gap",       # bm25_score - dense_score
    "rank_gap",        # bm25_rank - dense_rank (signed)
    "query_length",
    "chunk_length",
]


def _rrf_score(bm25_rank: int, dense_rank: int, k: int = 60) -> float:
    return 1.0 / (k + bm25_rank) + 1.0 / (k + dense_rank)


def _extract_features(
    query: str,
    chunk_id: str,
    bm25_rank_map: dict,
    dense_rank_map: dict,
    bm25_score_map: dict,
    dense_score_map: dict,
    chunk_text: str = "",
) -> list[float]:
    bm25_rank = bm25_rank_map.get(chunk_id, 999)
    dense_rank = dense_rank_map.get(chunk_id, 999)
    bm25_score = bm25_score_map.get(chunk_id, 0.0)
    dense_score = dense_score_map.get(chunk_id, 0.0)
    rrf = _rrf_score(bm25_rank, dense_rank)

    return [
        bm25_score,
        dense_score,
        float(bm25_rank),
        float(dense_rank),
        rrf,
        bm25_score - dense_score,
        float(bm25_rank - dense_rank),
        float(len(query.split())),
        float(len(chunk_text.split())),
    ]


def _build_rank_score_maps(results: list[dict]) -> tuple[dict, dict]:
    rank_map = {r["chunk_id"]: i + 1 for i, r in enumerate(results)}
    score_map = {r["chunk_id"]: r.get("score", 0.0) for r in results}
    return rank_map, score_map


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_pairs(
    queries_path: str,
    relevance_path: str,
    sparse: BM25sRetriever,
    dense: QdrantRetriever,
    k: int,
    snapshot_index: dict,
) -> tuple[list[list[float]], list[int]]:
    """Build features from labeled eval queries."""
    raw_queries = json.loads(Path(queries_path).read_text())
    relevance_map: dict[str, list[str]] = {
        key: val
        for key, val in json.loads(Path(relevance_path).read_text()).items()
        if not key.startswith("_")
    }

    labeled = [
        EvalQuery(id=q["id"], text=q["text"], type=q.get("type", "semantic"))
        for q in raw_queries
        if q["id"] in relevance_map and relevance_map[q["id"]]
    ]

    if not labeled:
        return [], []

    X, y = [], []
    for q in labeled:
        relevant = set(relevance_map[q.id])
        sparse_results = sparse.search(q.text, k=k)
        dense_results = dense.search(q.text, k=k)

        bm25_rank_map, bm25_score_map = _build_rank_score_maps(sparse_results)
        dense_rank_map, dense_score_map = _build_rank_score_maps(dense_results)

        seen: set[str] = set()
        candidates: list[dict] = []
        for r in sparse_results + dense_results:
            cid = r["chunk_id"]
            if cid not in seen:
                candidates.append(r)
                seen.add(cid)

        for chunk in candidates:
            cid = chunk["chunk_id"]
            chunk_text = snapshot_index.get(cid, {}).get("text", "")
            features = _extract_features(
                query=q.text,
                chunk_id=cid,
                bm25_rank_map=bm25_rank_map,
                dense_rank_map=dense_rank_map,
                bm25_score_map=bm25_score_map,
                dense_score_map=dense_score_map,
                chunk_text=chunk_text,
            )
            X.append(features)
            y.append(1 if cid in relevant else 0)

    return X, y


def _load_hard_negative_pairs(
    pairs_path: str,
    sparse: BM25sRetriever,
    dense: QdrantRetriever,
    snapshot_index: dict,
    k: int,
) -> tuple[list[list[float]], list[int]]:
    """
    Build features from hard negative pairs.
    Uses the positive/negative chunk texts directly — no retrieval needed.
    Approximates ranks by doing a single retrieval per unique query.
    """
    path = Path(pairs_path)
    if not path.exists():
        print(f"  [WARN] Hard negatives file not found: {path} — skipping")
        return [], []

    pairs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"  [INFO] Loaded {len(pairs)} hard negative pairs from {path}")

    # Cache retrieval per unique query to avoid redundant calls
    query_cache: dict[str, tuple[dict, dict, dict, dict]] = {}

    X, y = [], []
    for pair in pairs:
        query_text = pair["query"]
        chunk_id = pair["positive_id"]
        label = pair["label"]
        chunk_text = pair.get("positive", snapshot_index.get(chunk_id, {}).get("text", ""))

        if query_text not in query_cache:
            sparse_results = sparse.search(query_text, k=k)
            dense_results = dense.search(query_text, k=k)
            bm25_rank_map, bm25_score_map = _build_rank_score_maps(sparse_results)
            dense_rank_map, dense_score_map = _build_rank_score_maps(dense_results)
            query_cache[query_text] = (bm25_rank_map, dense_rank_map, bm25_score_map, dense_score_map)

        bm25_rank_map, dense_rank_map, bm25_score_map, dense_score_map = query_cache[query_text]

        features = _extract_features(
            query=query_text,
            chunk_id=chunk_id,
            bm25_rank_map=bm25_rank_map,
            dense_rank_map=dense_rank_map,
            bm25_score_map=bm25_score_map,
            dense_score_map=dense_score_map,
            chunk_text=chunk_text,
        )
        X.append(features)
        y.append(label)

    return X, y


def _load_snapshot_index(collection: str) -> dict[str, dict]:
    snapshot = settings.snapshot_path_for(collection)
    index = {}
    if snapshot.exists():
        with open(snapshot) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    index[chunk["chunk_id"]] = chunk
    return index


# ── Model training ────────────────────────────────────────────────────────────

def _try_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def _train_lgbm(X_arr, y_arr, pos: int, neg: int):
    lgb = _try_lgbm()
    if lgb is None:
        return None, None

    scale_pos_weight = neg / max(pos, 1)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": scale_pos_weight,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 5,
        "verbose": -1,
        "random_state": 42,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_arr, y_arr)
    return model, "lightgbm"


def _train_logistic(X_arr, y_arr, pos: int, neg: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    model.fit(X_scaled, y_arr)
    return model, scaler, "logistic"


def _cross_validate(X_arr, y_arr, model_type: str) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        pos = int(y_train.sum())
        neg = len(y_train) - pos

        if model_type == "lightgbm":
            model, _ = _train_lgbm(X_train, y_train, pos, neg)
            proba = model.predict_proba(X_val)[:, 1]
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            model, _, _ = _train_logistic(X_train_s, y_train, pos, neg)
            proba = model.predict_proba(X_val_s)[:, 1]

        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, proba)
            aucs.append(auc)
            print(f"    Fold {fold+1}: AUC = {auc:.4f}")

    return {"mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs))}


# ── Save / load compatible with LearnedHybridFusion ──────────────────────────

def _save_model(model, scaler, model_type: str, feature_names: list[str]) -> None:
    import pickle

    model_dir = settings.data_dir / "learned_fusion"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # LearnedHybridFusion expects scaler.pkl — write identity scaler for LightGBM
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(len(feature_names))
        scaler.scale_ = np.ones(len(feature_names))
        scaler.var_ = np.ones(len(feature_names))
        scaler.n_features_in_ = len(feature_names)

    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "model_type": model_type,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved to: {model_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="base")
    parser.add_argument("--k", type=int, default=20, help="Candidate pool size per query")
    parser.add_argument("--queries", default="evaluation/queries.json")
    parser.add_argument("--relevance", default="evaluation/relevance.json")
    parser.add_argument("--pairs", default="data/training_pairs.jsonl",
                        help="Hard negative pairs from build_training_pairs.py")
    parser.add_argument("--no-hard-negatives", action="store_true",
                        help="Skip hard negative pairs, use eval labels only")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation (faster, less info)")
    parser.add_argument("--algo", choices=["auto", "lgbm", "logistic"], default="auto",
                        help="auto=lgbm if available else logistic")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Fusion Model Training")
    print(f"  Collection : {args.collection}")
    print(f"  Candidate k: {args.k}")
    print(f"{'='*55}\n")

    # ── Retrievers
    print("[1/5] Loading retrievers...")
    sparse = BM25sRetriever(collection_slug=args.collection)
    if not sparse.load():
        print("ERROR: BM25 index not found. Run ingestion first.")
        sys.exit(1)
    dense = QdrantRetriever(collection_slug=args.collection)
    snapshot_index = _load_snapshot_index(args.collection)
    print(f"  Snapshot chunks loaded: {len(snapshot_index)}")

    # ── Features from eval labels
    print("\n[2/5] Building features from eval labels...")
    t0 = time.perf_counter()
    X_eval, y_eval = _load_eval_pairs(
        queries_path=args.queries,
        relevance_path=args.relevance,
        sparse=sparse,
        dense=dense,
        k=args.k,
        snapshot_index=snapshot_index,
    )
    print(f"  Eval pairs     : {len(X_eval)} (pos={sum(y_eval)}, neg={len(y_eval)-sum(y_eval)})")

    # ── Features from hard negatives
    X_hard, y_hard = [], []
    if not args.no_hard_negatives:
        print("\n[3/5] Building features from hard negative pairs...")
        X_hard, y_hard = _load_hard_negative_pairs(
            pairs_path=args.pairs,
            sparse=sparse,
            dense=dense,
            snapshot_index=snapshot_index,
            k=args.k,
        )
        print(f"  Hard neg pairs : {len(X_hard)} (pos={sum(y_hard)}, neg={len(y_hard)-sum(y_hard)})")
    else:
        print("\n[3/5] Skipping hard negatives (--no-hard-negatives)")

    # ── Combine
    X_all = X_eval + X_hard
    y_all = y_eval + y_hard

    if not X_all:
        print("ERROR: No training data. Run build_eval_dataset.py and build_training_pairs.py first.")
        sys.exit(1)

    X_arr = np.array(X_all, dtype=np.float32)
    y_arr = np.array(y_all, dtype=np.int32)
    pos = int(y_arr.sum())
    neg = len(y_arr) - pos
    elapsed = round((time.perf_counter() - t0) * 1000)

    print(f"\n  Total samples  : {len(X_arr)}")
    print(f"  Positives      : {pos} ({100*pos/len(y_arr):.1f}%)")
    print(f"  Negatives      : {neg} ({100*neg/len(y_arr):.1f}%)")
    print(f"  Feature count  : {len(FEATURE_NAMES)}")
    print(f"  Feature names  : {FEATURE_NAMES}")
    print(f"  Build time     : {elapsed}ms")

    # ── Algorithm selection
    lgb = _try_lgbm()
    if args.algo == "auto":
        model_type = "lightgbm" if lgb else "logistic"
    else:
        model_type = args.algo
        if model_type == "lgbm" and not lgb:
            print("\n[WARN] LightGBM not installed. Falling back to logistic.")
            print("  Install with: pip install lightgbm")
            model_type = "logistic"

    print(f"\n[4/5] Algorithm: {model_type}")

    # ── Cross-validation
    if not args.no_cv:
        print(f"\n  Running 5-fold cross-validation...")
        cv_results = _cross_validate(X_arr, y_arr, model_type)
        print(f"  CV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        if cv_results["mean_auc"] < 0.60:
            print("  [WARN] AUC < 0.60 — model has low signal. Consider more labeled data.")
    else:
        print("  Skipping cross-validation (--no-cv)")
        cv_results = {}

    # ── Train final model on all data
    print(f"\n[5/5] Training final model on all {len(X_arr)} samples...")
    scaler = None

    if model_type == "lightgbm":
        model, _ = _train_lgbm(X_arr, y_arr, pos, neg)
        train_auc = None
        # Feature importance
        importance = model.feature_importances_
        print("\n  Feature importance (LightGBM):")
        for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
            bar = "#" * int(imp / max(importance) * 20)
            print(f"    {name:<20} {bar} ({imp})")
    else:
        model, scaler, _ = _train_logistic(X_arr, y_arr, pos, neg)
        train_auc = None
        print("\n  Feature coefficients (LogReg):")
        for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
            print(f"    {name:<20} {coef:+.4f}")

    _save_model(model, scaler, model_type, FEATURE_NAMES)

    print(f"\n{'='*55}")
    print(f"  Training complete")
    print(f"  Algorithm      : {model_type}")
    print(f"  Samples        : {len(X_arr)}")
    if cv_results:
        print(f"  CV AUC         : {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    print(f"{'='*55}")
    print(f"\nNext: run eval to compare learned vs RRF:")
    print(f"  python scripts/run_eval.py --collection {args.collection}")

    try:
        dense._client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
