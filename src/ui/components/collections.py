import streamlit as st
import requests
from ui.config import API_BASE


def render_collections_tab():
    st.subheader("🗂 Manage Collections")

    # ── Create new collection ────────────────────────────────────────────────
    with st.expander("➕ Create New Collection", expanded=False):
        col_name = st.text_input("Collection name", placeholder="e.g. HR Policies")
        col_desc = st.text_area("Description (optional)", height=80)
        if st.button("Create Collection", type="primary"):
            if not col_name.strip():
                st.warning("Please enter a collection name")
            else:
                resp = requests.post(
                    f"{API_BASE}/collections",
                    json={"name": col_name.strip(), "description": col_desc.strip()},
                    timeout=5,
                )
                if resp.status_code == 201:
                    st.success(f"✓ Collection '{col_name}' created")
                    st.rerun()
                elif resp.status_code == 400:
                    st.error(resp.json().get("detail", "Error creating collection"))
                else:
                    st.error("Unexpected error")

    st.markdown("---")

    # ── Existing collections ─────────────────────────────────────────────────
    try:
        resp = requests.get(f"{API_BASE}/collections", timeout=3)
        collections = resp.json() if resp.status_code == 200 else []
    except Exception:
        st.error("Cannot reach API")
        return

    if not collections:
        st.info("No collections yet. Create one above.")
        return

    for col in collections:
        with st.container(border=True):
            header_col, del_col = st.columns([5, 1])
            with header_col:
                st.markdown(f"### {col['name']}")
                if col["description"]:
                    st.caption(col["description"])
            with del_col:
                if st.button("🗑️", key=f"del_{col['slug']}", help="Delete collection"):
                    st.session_state[f"confirm_delete_{col['slug']}"] = True

            # Confirm delete
            if st.session_state.get(f"confirm_delete_{col['slug']}"):
                st.warning(f"Delete **{col['name']}** and all its documents? This cannot be undone.")
                c1, c2 = st.columns(2)
                if c1.button("Yes, delete", key=f"yes_{col['slug']}", type="primary"):
                    r = requests.delete(f"{API_BASE}/collections/{col['slug']}", timeout=10)
                    if r.status_code == 204:
                        st.success("Deleted")
                        st.session_state.pop(f"confirm_delete_{col['slug']}", None)
                        if st.session_state.get("active_collection") == col["slug"]:
                            st.session_state.pop("active_collection", None)
                        st.rerun()
                    else:
                        st.error("Delete failed")
                if c2.button("Cancel", key=f"no_{col['slug']}"):
                    st.session_state.pop(f"confirm_delete_{col['slug']}", None)
                    st.rerun()

            # Stats row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Files", len(col["files"]))
            m2.metric("Chunks", col["total_chunks"])
            m3.metric("Tokens", f"{col['total_tokens']:,}")
            m4.metric("Created", col["created_at"][:10])

            # File table
            if col["files"]:
                with st.expander(f"📄 Files ({len(col['files'])})"):
                    for f in col["files"]:
                        fc1, fc2, fc3, fc4 = st.columns([3, 1, 1, 1])
                        fc1.markdown(f"**{f['filename']}**")
                        fc2.caption(f"{f['pages']} pages")
                        fc3.caption(f"{f['chunks']} chunks")
                        fc4.caption(f"{f['ingested_at'][:10]}")
