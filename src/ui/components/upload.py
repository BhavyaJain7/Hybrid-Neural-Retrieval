import streamlit as st
import requests
from ui.config import API_BASE


def render_upload_tab(active_collection: str | None):
    st.subheader("📂 Upload Documents")

    if not active_collection:
        st.warning("Select or create a collection first (use the sidebar or Collections tab)")
        return

    # Show active collection context
    try:
        resp = requests.get(f"{API_BASE}/collections/{active_collection}", timeout=3)
        col = resp.json() if resp.status_code == 200 else {}
    except Exception:
        col = {}

    st.info(f"Uploading into: **{col.get('name', active_collection)}**")

    uploaded_files = st.file_uploader(
        "Select PDF or DOCX files from your computer",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Hold Ctrl/Cmd to select multiple files",
    )

    if not uploaded_files:
        return

    # Pre-flight duplicate check
    existing_files = {f["filename"] for f in col.get("files", [])}
    duplicates = [f.name for f in uploaded_files if f.name in existing_files]
    new_files = [f for f in uploaded_files if f.name not in existing_files]

    if duplicates:
        st.warning(f"Already indexed: **{', '.join(duplicates)}**")
        force = st.checkbox("Re-ingest duplicates (overwrites existing)")
    else:
        force = False

    files_to_process = uploaded_files if force else new_files

    if not files_to_process:
        st.info("No new files to ingest")
        return

    st.markdown(f"**{len(files_to_process)} file(s) ready to ingest:**")
    for f in files_to_process:
        file_bytes = f.getvalue()  # read once; avoids double-consuming the stream
        size_kb = round(len(file_bytes) / 1024, 1)
        st.caption(f"📄 {f.name}  —  {size_kb} KB")

    if st.button("Ingest Documents", type="primary"):
        progress = st.progress(0, text="Starting...")
        results_placeholder = st.empty()
        summary = []

        for i, upload in enumerate(files_to_process):
            # Advance progress before processing so the bar reflects current work
            progress.progress(
                (i + 1) / len(files_to_process),
                text=f"Processing {upload.name} ({i + 1}/{len(files_to_process)})...",
            )
            file_bytes = upload.getvalue()  # read once
            try:
                resp = requests.post(
                    f"{API_BASE}/collections/{active_collection}/ingest",
                    files={"file": (upload.name, file_bytes, upload.type)},
                    params={"force": str(force).lower()},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    summary.append({
                        "file": upload.name,
                        "chunks": data["chunks_indexed"],
                        "status": "✓",
                        "warnings": data.get("warnings", []),
                    })
                elif resp.status_code == 409:
                    summary.append({"file": upload.name, "status": "⚠ skipped (duplicate)", "chunks": 0, "warnings": []})
                else:
                    summary.append({"file": upload.name, "status": "✗ failed", "chunks": 0, "warnings": [resp.text]})
            except Exception as e:
                summary.append({"file": upload.name, "status": "✗ error", "chunks": 0, "warnings": [str(e)]})

        progress.progress(1.0, text="Done!")

        # Results summary table
        st.markdown("### Ingestion Results")
        total_chunks = sum(s["chunks"] for s in summary)

        for s in summary:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"{s['status']} **{s['file']}**")
                c2.caption(f"{s['chunks']} chunks")
                for w in s["warnings"]:
                    st.caption(f"⚠ {w}")

        st.success(f"Complete — **{total_chunks} chunks** indexed")
        st.rerun()
