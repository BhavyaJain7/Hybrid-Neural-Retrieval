import sys
from pathlib import Path

# #17: ensure ui/ is on sys.path so relative component imports work
# regardless of where Streamlit is launched from
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import requests
from ui.config import API_BASE                          # #23: single source
from ui.components.sidebar import render_sidebar
from ui.components.upload import render_upload_tab
from ui.components.collections import render_collections_tab
from ui.components.results import render_results, render_answer, render_debug

st.set_page_config(
    page_title="Neural Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "tavily_calls" not in st.session_state:
    st.session_state.tavily_calls = 0

options = render_sidebar()
active_collection = options["collection"]

st.title("🔍 Neural Search")
st.caption("Hybrid semantic search — BM25 + Dense Retrieval + Reciprocal Rank Fusion")

if active_collection:
    st.markdown(f"Searching in: **{active_collection}**")

st.markdown("---")

tab_search, tab_upload, tab_collections, tab_history = st.tabs([
    "🔍 Search", "📂 Upload", "🗂 Collections", "🕓 History"
])

with tab_search:
    if not active_collection:
        st.info("👈 Create a collection and upload documents to get started")
    else:
        query = st.text_input(
            "Ask a question or describe what you're looking for",
            placeholder="e.g. What are the payment terms for contract renewal?",
            label_visibility="collapsed",
        )

        col_btn, col_debug = st.columns([1, 6])
        search_clicked = col_btn.button("Search", type="primary", use_container_width=True)
        debug_mode = col_debug.toggle("Debug view (show BM25 vs Neural vs RRF breakdown)", value=False)

        if search_clicked and query.strip():
            with st.spinner("Searching..."):
                try:
                    if debug_mode:
                        resp = requests.get(
                            f"{API_BASE}/search/debug",
                            params={"query": query, "collection": active_collection, "k": options["top_k"]},
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            render_debug(resp.json())
                        else:
                            st.error(f"Debug failed: {resp.text}")
                    else:
                        resp = requests.post(
                            f"{API_BASE}/search",
                            json={
                                "query": query,
                                "collection": active_collection,
                                "k": options["top_k"],
                                "mode": options["mode"],
                                "synthesize": options["synthesize"],
                                "expand": options["expand"],
                                "web_search": options["web_search"],
                            },
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            synthesis_triggered = data.get("synthesis_triggered", False)
                            retrieval_confidence = data.get("retrieval_confidence", 0.0)
                            st.session_state.query_history.append({
                                "query": query,
                                "collection": active_collection,
                                "latency_ms": data["latency_ms"],
                                "mode": data["mode"],
                                "results": len(data["results"]),
                                "retrieval_confidence": retrieval_confidence,
                                "web_used": data.get("web_results_used", False),
                                "latency": data.get("latency", {}),
                            })
                            if data.get("web_results_used"):
                                st.session_state.tavily_calls += 1
                            if options["synthesize"]:
                                render_answer(
                                    synthesis=data.get("synthesis"),
                                    triggered=synthesis_triggered,
                                    confidence=retrieval_confidence,
                                )
                            render_results(
                                results=data["results"],
                                latency_ms=data["latency_ms"],
                                mode=data["mode"],
                                web_results_used=data.get("web_results_used", False),
                                retrieval_confidence=retrieval_confidence,
                                expansion_queries=data.get("expansion_queries", []),
                            )
                        else:
                            st.error(f"Search error: {resp.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API — run `./run.sh api`")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        elif search_clicked:
            st.warning("Please enter a query")

with tab_upload:
    render_upload_tab(active_collection)

with tab_collections:
    render_collections_tab()

with tab_history:
    st.subheader("🕓 Query History")
    history = st.session_state.query_history
    if not history:
        st.info("No queries yet")
    else:
        for entry in reversed(history[-30:]):
            with st.container(border=True):
                # Row 1: query + collection
                c1, c2 = st.columns([5, 2])
                web_icon = " 🌐" if entry.get("web_used") else ""
                c1.markdown(f"**{entry['query']}**{web_icon}")
                c2.caption(f"📁 {entry['collection']}")

                # Row 2: mode | confidence | total latency | result count
                c3, c4, c5, c6 = st.columns([2, 2, 2, 1])
                c3.caption(f"`{entry['mode']}`")
                conf = entry.get("retrieval_confidence", 0.0)
                c4.caption(f"conf: `{conf:.4f}`")
                c5.caption(f"⏱ {entry['latency_ms']} ms total")
                c6.caption(f"{entry.get('results', '?')} results")

                # Row 3: per-component latency breakdown (if available)
                lat = entry.get("latency") or {}
                if lat:
                    parts = []
                    if lat.get("retrieval_ms") is not None:
                        parts.append(f"retrieval: {lat['retrieval_ms']}ms")
                    if lat.get("rerank_ms") is not None:
                        parts.append(f"rerank: {lat['rerank_ms']}ms")
                    if lat.get("synthesis_ms") is not None:
                        parts.append(f"synthesis: {lat['synthesis_ms']}ms")
                    if parts:
                        st.caption("  " + "  |  ".join(parts))

        if st.button("Clear History"):
            st.session_state.tavily_calls = 0
            st.session_state.query_history = []
            st.rerun()
