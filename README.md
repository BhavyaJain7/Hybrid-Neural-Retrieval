# Neural Search

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-local-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A hybrid document retrieval system combining BM25 sparse search, Qdrant dense vectors, and Tavily web augmentation — fused via weighted RRF, reranked with a cross-encoder, and summarized through confidence-gated LLM synthesis.

Every component ships only with quantitative evidence of improvement against a committed baseline.

---

## Table of Contents

- [Architecture](#architecture)
- [Stack](#stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Design Decisions](#design-decisions)
- [Out of Scope](#out-of-scope)
- [Roadmap](#roadmap)

---

## Architecture

```
Query
  │
  ├─► Query Expander (Groq, optional)
  │
  ├─► BM25s     (sparse)  ─┐
  ├─► Qdrant    (dense)    ├─► Weighted RRF ─► Cross-Encoder Reranker ─► Groq Synthesis
  └─► Tavily    (web)     ─┘   (3-way)           (ms-marco)              (confidence-gated)
```

**Query Expansion**: Groq generates 2 alternative phrasings of the query. All 3 are searched independently; results are unioned and deduped by `chunk_id` before fusion. Enabled for `vague` and `semantic` query types only.

**Web Deduplication**: Before RRF fusion, web snippets with cosine similarity > 0.85 to any local result are dropped — preventing web retrieval from simply amplifying what local search already found.

**Confidence-Gated Synthesis**: LLM synthesis only fires when the top RRF score exceeds `SYNTHESIS_THRESHOLD`. Below the threshold, the API returns raw results with an explicit warning rather than a confidently wrong answer.

---

## Stack

| Layer | Tool | Notes |
|---|---|---|
| Document parsing | `pymupdf`, `python-docx` | PDF and DOCX |
| Chunking | `langchain-text-splitters` | Configurable size + overlap |
| Token counting | `tiktoken` (`cl100k_base`) | Accurate — not word count |
| Sparse retrieval | `BM25s` + NLTK stopwords | Lexical baseline |
| Embedding model | `all-MiniLM-L6-v2` | Shared across dense + web |
| Dense retrieval | `Qdrant` (local mode) | No infra overhead |
| Web retrieval | `Tavily API` | Gated — not called on every query |
| Fusion | Weighted RRF (3-way) | BM25=1.0, Dense=1.0, Web=0.6 |
| Reranking | `ms-marco-MiniLM-L-6-v2` | Cross-encoder, retrieve-20/return-5 |
| Query expansion | `Groq` `llama-3.1-8b-instant` | 2 expansions per query |
| Synthesis | `Groq` `llama-3.1-8b-instant` | Confidence-gated |
| API | `FastAPI` | |
| UI | `Streamlit` | |
| Config | `pydantic-settings` + `.env` | |
| Logging | `loguru` | Per-component latency |

---

## Project Structure

```
neural_search/
  evaluation/
    queries.json              # 321 labeled queries (keyword / semantic / vague)
    relevance.json            # ground truth chunk_ids per query
    results/                  # committed benchmark outputs per phase
      phase2_baseline.json
      phase3.json
  scripts/
    run_eval.py               # evaluation runner — run at end of every phase
    build_eval_dataset.py     # semi-automated relevance labeling
    generate_queries.py       # Groq-assisted query generation
    BM25_benchmark.py         # latency benchmarking with SLA checks
    verify_index.py           # index sync checker
  src/
    neural_search/
      api/
        main.py               # FastAPI app + lifespan
        routes.py             # /search, /search/debug, /collections, /ingest
        schemas.py            # request/response models
      retrieval/
        sparse.py             # BM25s retriever
        dense.py              # Qdrant retriever
        web.py                # Tavily retriever + freshness weighting
        hybrid.py             # 3-way weighted RRF fusion
        reranker.py           # CrossEncoder reranker
        expander.py           # Query expansion via Groq
        deduplicator.py       # Web vs local cosine dedup
      evaluation/
        metrics.py            # P@K, Recall@K, MRR, nDCG
        dataset.py            # eval data loader
        runner.py             # per-method comparison runner
      ingestion/
        parser.py             # PDF + DOCX page extractor
        chunker.py            # tiktoken-accurate chunker
        pipeline.py           # ingestion orchestration + JSONL snapshot
      synthesis/
        groq_client.py        # Groq synthesizer with retry + backoff
        prompt.py             # prompt builder
      config.py               # pydantic-settings
    ui/
      app.py                  # Streamlit app
      components/             # sidebar, results, upload, collections
```

---

## Setup

### Prerequisites

- Python 3.10+
- Groq API key (required — used for synthesis and query expansion)
- Tavily API key (optional — enables web augmentation)

### Install

```bash
git clone https://github.com/Arynshr/Neural_search.git
cd Neural_search
pip install -e .
```

### Configure

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required |
| `TAVILY_API_KEY` | — | Optional |
| `TAVILY_ENABLED` | `false` | Enable web augmentation |
| `TAVILY_WEIGHT` | `0.6` | Web source RRF weight |
| `WEB_TRIGGER_THRESHOLD` | `0.3` | Min local confidence before Tavily fires |
| `SYNTHESIS_THRESHOLD` | `0.4` | Min RRF score to trigger LLM synthesis |
| `SYNTHESIS_ENABLED` | `true` | |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `CHUNK_SIZE` | `512` | Characters |
| `CHUNK_OVERLAP` | `64` | Characters |

### Run

```bash
# API
uvicorn src.neural_search.api.main:app --reload

# UI
streamlit run src/ui/app.py

# Ingest documents
python scripts/ingest_documents.py --source ./docs --collection default
```

---

## API Reference

### `POST /search`

**Request**

```json
{
  "query": "how do transformers handle long sequences",
  "collection": "default",
  "k": 10,
  "mode": "hybrid",
  "rerank": true,
  "rerank_top_k": 5,
  "expand": false,
  "web_search": false,
  "synthesize": true,
  "query_type": "semantic"
}
```

`mode`: `hybrid` | `sparse` | `dense` | `learned`  
`query_type`: `keyword` | `semantic` | `vague`

**Response**

```json
{
  "query": "...",
  "mode": "hybrid",
  "reranked": true,
  "retrieval_confidence": 0.74,
  "web_results_used": false,
  "synthesis_triggered": true,
  "expansion_queries": [],
  "results": [
    {
      "chunk_id": "doc_001::12",
      "source_file": "attention_paper.pdf",
      "page": 4,
      "text": "...",
      "score": 0.91,
      "rank": 1,
      "rerank_score": 8.34,
      "source": "local",
      "source_url": null
    }
  ],
  "synthesis": {
    "answer": "...",
    "sources_used": [],
    "model": "llama-3.1-8b-instant"
  },
  "latency": {
    "retrieval_ms": 312.4,
    "rerank_ms": 840.2,
    "synthesis_ms": 1204.1,
    "total_ms": 2356.7
  }
}
```

### `GET /search/debug`

```
GET /search/debug?query=attention+mechanism&collection=default&k=10
```

Returns ranked results from BM25, Dense, Hybrid RRF, and Web independently — for retrieval method comparison.

### `POST /collections/{slug}/ingest`

Multipart file upload. Accepts PDF and DOCX. Supports incremental ingestion — re-ingesting a file replaces only that file's chunks.

### `GET /health`

Returns collection count and Tavily enabled status.

---

## Evaluation

Run `scripts/run_eval.py` at the end of every phase. Results are committed to `evaluation/results/`.

```bash
python scripts/run_eval.py \
  --collection default \
  --eval-dir evaluation/ \
  --phase 3 \
  --k 5
```

### Phase 3 Results — 321 Labeled Queries

| Method | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 (sparse) | 0.296 | 0.546 | 0.546 | 0.572 |
| Dense (Qdrant) | 0.299 | 0.562 | 0.562 | 0.588 |
| Hybrid RRF | 0.310 | 0.609 | 0.613 | 0.638 |
| Hybrid RRF + Reranker | **0.314** | **0.636** | **0.675** | **0.692** |
| Learned Fusion | 0.286 | 0.572 | 0.596 | 0.617 |

Reranker lift over Hybrid RRF: **+8.5% nDCG@5**

### Per Query-Type — Hybrid RRF + Reranker

| Query Type | Queries | nDCG@5 | MRR |
|---|---|---|---|
| Keyword | 140 | 0.889 | 0.873 |
| Semantic | 84 | 0.881 | 0.864 |
| Vague | 97 | 0.244 | 0.226 |

Vague queries are the primary gap. Addressed in Phase 4 (query expansion) and Phase 5 (web augmentation).

### Performance Targets

| Metric | Target | Status |
|---|---|---|
| P@3 | > 0.80 | Pending Phase 4+ |
| nDCG@5 lift over BM25 | > 15% | 20.9% at Phase 3 |
| Search latency (no web) | < 2s | TBD |
| Search latency (with web) | < 5s | TBD |
| Tavily calls / day | < 30 | Gating in place |

---

## Design Decisions

**Why RRF over learned fusion?** Logistic regression requires 500+ labeled pairs minimum. At current dataset size, learned fusion underperforms RRF (0.617 vs 0.692 nDCG@5). Revisit when the dataset scales.

**Why adaptive web gating?** Tavily free tier is 1000 calls/month. Calling it on every query exhausts the budget in ~33 days at moderate usage. Web retrieval fires only when local confidence is low or explicitly requested.

**Why a cross-encoder reranker instead of a bi-encoder?** Cross-encoders attend to query and document jointly, producing more accurate relevance scores at the cost of latency. The retrieve-20/rerank/return-5 pattern keeps end-to-end latency acceptable.

**Why confidence-gated synthesis?** Unconditional synthesis on weak retrieval produces confident, wrong answers. The gate ensures the LLM only summarizes when retrieval quality justifies it.

**Why `tiktoken` for token counting?** Word splitting systematically underestimates token counts. Accurate counts are required for chunking strategy ablation results to be meaningful.

---

## Out of Scope

| Feature | Reason |
|---|---|
| Learned hybrid (LogReg / LightGBM) | Requires 500+ labeled pairs — dataset too small |
| Fine-tuning embedding model | Same constraint |
| ColBERT / SPLADE | Complexity without proven gain at this scale |
| Multi-modal (PDF images) | Separate problem domain |
| Vector quantization | Qdrant handles internally |

---

## License

MIT
