from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger


QueryType = Literal["keyword", "semantic", "vague"]


@dataclass(frozen=True)
class EvalQuery:
    id: str
    text: str
    type: QueryType


@dataclass(frozen=True)
class EvalDataset:
    queries: list[EvalQuery]
    relevance: dict[str, list[str]]  # query_id -> list of relevant chunk_ids

    def get_relevant(self, query_id: str) -> set[str]:
        return set(self.relevance.get(query_id, []))

    def labeled_queries(self) -> list[EvalQuery]:
        """Return only queries that have at least one relevant chunk labeled."""
        return [q for q in self.queries if self.get_relevant(q.id)]

    def by_type(self, query_type: QueryType) -> list[EvalQuery]:
        return [q for q in self.labeled_queries() if q.type == query_type]

    @property
    def coverage(self) -> str:
        labeled = len(self.labeled_queries())
        total = len(self.queries)
        return f"{labeled}/{total} queries labeled"


def load_dataset(
    queries_path: Path | str = "evaluation/queries.json",
    relevance_path: Path | str = "evaluation/relevance.json",
) -> EvalDataset:
    queries_path = Path(queries_path)
    relevance_path = Path(relevance_path)

    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not relevance_path.exists():
        raise FileNotFoundError(f"Relevance file not found: {relevance_path}")

    with queries_path.open() as f:
        raw_queries = json.load(f)

    with relevance_path.open() as f:
        raw_relevance = json.load(f)

    queries = []
    for q in raw_queries:
        query_type = q.get("type", "semantic")
        if query_type not in ("keyword", "semantic", "vague"):
            logger.warning(f"Unknown query type '{query_type}' for {q['id']} — defaulting to 'semantic'")
            query_type = "semantic"
        queries.append(EvalQuery(id=q["id"], text=q["text"], type=query_type))

    # Strip internal note key if present
    relevance = {k: v for k, v in raw_relevance.items() if not k.startswith("_")}

    dataset = EvalDataset(queries=queries, relevance=relevance)
    logger.info(f"Loaded eval dataset — {dataset.coverage}")
    return dataset
