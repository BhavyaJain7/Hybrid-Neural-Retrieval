from __future__ import annotations

import requests
import streamlit as st

from ui.config import API_BASE


def render_sidebar() -> dict:
    st.sidebar.title("🔍 Neural Search")

    # ── Collection picker ─────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{API_BASE}/collections", timeout=3)
        collections = resp.json() if resp.status_code == 200 else []
    except Exception:
        collections = []
        st.sidebar.warning("API unreachable — is the server running?")

    active_slug = None

    if collections:
        col_options = {c["name"]: c["slug"] for c in collections}
        col_labels = list(col_options.keys())
        default_idx = 0

        if "active_collection" in st.session_state:
            saved = st.session_state.active_collection
            matching = [i for i, s in enumerate(col_options.values()) if s == saved]
            if matching:
                default_idx = matching[0]

        selected_name = st.sidebar.radio("Collection", col_labels, index=default_idx)
        active_slug = col_options[selected_name]
        st.session_state.active_collection = active_slug

        active = next(c for c in collections if c["slug"] == active_slug)
        st.sidebar.caption(
            f"{active['total_chunks']} chunks · {active['total_tokens']} tokens"
        )
    else:
        st.sidebar.info("No collections yet — create one in the Collections tab.")

    st.sidebar.divider()

    # ── Search options ────────────────────────────────────────────────────────
    st.sidebar.subheader("Search Options")

    mode = st.sidebar.selectbox(
        "Retrieval mode",
        ["hybrid", "sparse", "dense"],
        index=0,
    )
    top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=20, value=5)

    st.sidebar.divider()

    # ── Feature toggles ───────────────────────────────────────────────────────
    st.sidebar.subheader("Features")

    synthesize = st.sidebar.toggle("🤖 Generate Answer (Groq)", value=False)
    expand = st.sidebar.toggle("🔁 Expand Query", value=False)
    web_search = st.sidebar.toggle("🌐 Augment with Web Search", value=False)

    if web_search:
        st.sidebar.caption("⚠️ Uses Tavily API credits.")
    if expand:
        st.sidebar.caption("Query rephrased ×2 before retrieval.")

    # ── Observability: Tavily call counter ───────────────────────────────────
    tavily_calls = st.session_state.get("tavily_calls", 0)
    if tavily_calls > 0 or web_search:
        st.sidebar.divider()
        color = "red" if tavily_calls >= 25 else ("orange" if tavily_calls >= 15 else "green")
        st.sidebar.markdown(
            f"**🌐 Tavily calls this session:** "
            f"<span style='color:{color};font-weight:bold'>{tavily_calls}</span>/30 daily budget",
            unsafe_allow_html=True,
        )

    return {
        "collection": active_slug,
        "mode": mode,
        "top_k": top_k,
        "synthesize": synthesize,
        "expand": expand,
        "web_search": web_search,
    }
