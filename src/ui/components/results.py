from __future__ import annotations

import streamlit as st

SOURCE_BADGE = {
    "sparse": "🟡 BM25",
    "dense": "🔵 Neural",
    "sparse+dense": "🟢 Hybrid",
    "dense+sparse": "🟢 Hybrid",
    "hybrid": "🟢 Hybrid",
    "web": "🌐 Web",
}


def render_answer(
    synthesis: dict | None,
    triggered: bool = True,
    confidence: float = 1.0,
) -> None:
    """
    Render synthesised answer.
    triggered=False means confidence gate blocked synthesis.
    """
    if not triggered:
        st.info(
            f"⚠️ Answer generation skipped — retrieval confidence "
            f"({confidence:.3f}) is below threshold. Showing results only."
        )
        return

    if not synthesis:
        return

    st.subheader("🤖 Generated Answer")
    st.markdown(synthesis.get("answer", ""))

    sources = synthesis.get("sources_used", [])
    if sources:
        with st.expander("Sources used"):
            for s in sources:
                url = s.get("source_url")
                if url:
                    st.markdown(f"- [{url}]({url})")
                else:
                    st.markdown(
                        f"- {s.get('source_file', 'unknown')} "
                        f"(page {s.get('page', '?')})"
                    )

    st.caption(f"Model: {synthesis.get('model', 'unknown')}")
    st.divider()


def render_results(
    results: list[dict],
    latency_ms: float,
    mode: str,
    web_results_used: bool = False,
    retrieval_confidence: float = 0.0,
    expansion_queries: list[str] | None = None,
) -> None:
    """Renders ranked result cards with metadata and score breakdown."""
    if expansion_queries:
        st.caption(f"🔁 Expanded: {' | '.join(expansion_queries)}")

    web_label = " + 🌐 Web" if web_results_used else ""
    st.caption(
        f"{len(results)} results · {latency_ms}ms · "
        f"mode: {mode}{web_label} · "
        f"confidence: {retrieval_confidence:.3f}"
    )

    for result in results:
        source = result.get("source", "")
        badge = SOURCE_BADGE.get(source, "⚪ Unknown")
        is_web = source == "web"
        border = "#2196F3" if is_web else "#4CAF50"

        # Best available score key
        if result.get("rerank_score") is not None:
            score_key, score_val = "rerank", result["rerank_score"]
        elif result.get("rrf_score") is not None:
            score_key, score_val = "rrf", result["rrf_score"]
        else:
            score_key, score_val = "score", result.get("score", 0.0)

        st.markdown(
            f'<div style="border-left:3px solid {border};padding-left:10px;'
            f'margin-bottom:6px"><b>{badge}</b> &nbsp; Rank {result["rank"]} '
            f'&nbsp;|&nbsp; <code>{score_key}: {round(score_val, 4)}</code></div>',
            unsafe_allow_html=True,
        )

        source_url = result.get("source_url")
        if source_url:
            st.markdown(f"**Source:** [{source_url}]({source_url})")
        else:
            st.caption(
                f"📄 {result.get('source_file', '?')} · "
                f"page {result.get('page', '?')} · "
                f"{result.get('token_count', '?')} tokens"
            )

        with st.expander("View chunk text"):
            st.write(result.get("text", ""))

        cols = st.columns(4)
        if result.get("rrf_score") is not None:
            cols[0].metric("RRF", round(result["rrf_score"], 4))
        if result.get("rerank_score") is not None:
            cols[1].metric("Rerank", round(result["rerank_score"], 4))
        if result.get("freshness_weight") is not None:
            cols[2].metric("Freshness", round(result["freshness_weight"], 3))

        st.divider()


def render_debug(debug: dict) -> None:
    st.subheader("🔍 Debug View")

    tab_labels = ["BM25 (Sparse)", "Dense (Neural)", "Hybrid RRF"]
    if debug.get("web"):
        tab_labels.append("Web (Tavily)")

    tabs = st.tabs(tab_labels)

    def _render_tab(tab, results: list[dict], label: str) -> None:
        with tab:
            if not results:
                st.info(f"No {label} results")
                return
            for r in results:
                score = r.get("rrf_score", r.get("score", 0))
                st.markdown(
                    f"**[{r.get('rank', '?')}]** `{r['chunk_id']}` — "
                    f"score: `{score:.4f}`"
                )
                st.caption(
                    f"{r.get('source_file', '?')} · page {r.get('page', '?')}"
                )
                st.write(r.get("text", "")[:200] + "...")
                st.divider()

    _render_tab(tabs[0], debug.get("sparse", []), "BM25")
    _render_tab(tabs[1], debug.get("dense", []), "Dense")
    _render_tab(tabs[2], debug.get("hybrid_rrf", []), "Hybrid RRF")
    if debug.get("web"):
        _render_tab(tabs[3], debug.get("web", []), "Web")
