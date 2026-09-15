"""Phase 1 Streamlit UI. Run from the repository root: streamlit run app.py."""
import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from rag_insight.config import Settings
from rag_insight.embeddings import OllamaEmbedder
from rag_insight.inspection import CharacterBudgetTokenizer, format_inspection, inspect_documents


def location(record):
    if record["page"] is not None:
        return f"page {record['page']}"
    if record["line_start"] is not None:
        return f"lines {record['line_start']}-{record['line_end']}"
    return "location unavailable"


def inspect_uploads(uploads, settings, collection):
    """Persist session uploads, then parse and chunk without model calls."""
    names = [Path(upload.name).name for upload in uploads]
    if len(names) != len(set(names)):
        raise ValueError("Rename duplicate filenames before uploading them together.")
    root = Path("data/uploads") / collection
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for upload, name in zip(uploads, names, strict=True):
        path = root / name
        path.write_bytes(upload.getvalue())
        paths.append(path)
    return inspect_documents(paths, settings, CharacterBudgetTokenizer())


st.set_page_config(page_title="RAG Insight — Phase 1", page_icon="📚", layout="wide")
st.title("RAG Insight")
st.caption("Phase 1: verify document ingestion, provenance, and chunk boundaries.")
configured = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
configured["chunk_strategy"] = "structure"
base_settings = Settings(**configured)
st.info("Document inspection does not create an index or call an LLM. The embedding model below is tested "
        "separately; inspection uses a conservative character budget because Ollama does not expose its tokenizer.")

if "collection" not in st.session_state:
    st.session_state.collection = uuid4().hex

with st.sidebar:
    st.header("Active implementation")
    st.caption(f"Embedding backend: `{base_settings.embedding_backend}`")
    st.caption(f"Embedding model: `{base_settings.embedding_model}`")
    st.caption("Embedding client: Python standard-library HTTP → Ollama `/api/embed`")
    st.caption("Parsing: `pypdf` + UTF-8 decoder · UI: `streamlit`")
    if st.button("Test embedding model"):
        try:
            vector = OllamaEmbedder(base_settings.embedding_model, base_settings.ollama_url).encode(
                ["Phase 1 embedding smoke test"]
            )[0]
            st.success(f"Ollama responded with {len(vector)} dimensions.")
        except RuntimeError as error:
            st.error(str(error))
    st.header("Chunk settings")
    strategy = st.selectbox("Strategy", ["structure", "recursive"],
                            help="Structure preserves Markdown heading ancestry; recursive is the baseline.")
    chunk_tokens = st.number_input("Chunk budget", min_value=1, value=220, step=10,
                                   help="Character budget in this offline Phase 1 environment.")
    overlap_tokens = st.number_input("Overlap", min_value=0, max_value=chunk_tokens - 1,
                                     value=min(30, chunk_tokens - 1), step=1)
    st.caption("The configured embedding tokenizer is unavailable, so these are conservative character units.")

with st.expander("How to test Phase 1", expanded=False):
    st.markdown(
        "1. Upload `structured.md`, `unicode.txt`, and a small PDF.\n"
        "2. Click **Inspect documents** and confirm document/chunk counts are nonzero.\n"
        "3. Open chunks and verify heading paths, code fences, line/page locations, and source text.\n"
        "4. Confirm **Missing locations** is zero for Markdown/text/PDF samples.\n"
        "5. Upload an empty file, unsupported file, or image-only PDF and confirm a clear error.\n"
        "6. Click **Test embedding model** and confirm Ollama returns 768 dimensions.\n\n"
        "Automated coverage for replacement atomicity, rollback, deterministic IDs, and exact invariants "
        "is run with `python -m pytest`; the UI is for interactive provenance review."
    )

uploads = st.file_uploader("Upload Markdown, text, or text-based PDF documents", type=["md", "txt", "pdf"],
                           accept_multiple_files=True)
if st.button("Inspect documents", type="primary", disabled=not uploads):
    settings = Settings(
        embedding_backend=base_settings.embedding_backend,
        embedding_model=base_settings.embedding_model,
        ollama_url=base_settings.ollama_url,
        chunk_strategy=strategy,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    try:
        with st.spinner("Parsing documents and building inspectable chunks…"):
            st.session_state.inspection = inspect_uploads(uploads, settings, st.session_state.collection)
        st.success("Documents parsed and chunked. No embeddings or index were created.")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not inspect the upload: {error}")
        st.session_state.pop("inspection", None)

records = st.session_state.get("inspection", [])
if records:
    documents = sorted({record["filename"] for record in records})
    missing_locations = sum(record["page"] is None and record["line_start"] is None for record in records)
    max_tokens = max(record["token_count"] for record in records)
    one, two, three, four = st.columns(4)
    one.metric("Documents", len(documents))
    two.metric("Chunks", len(records))
    three.metric("Maximum budget units", max_tokens)
    four.metric("Missing locations", missing_locations)

    st.subheader("Chunk inventory")
    table = [{"filename": record["filename"], "heading_path": record["heading_path"] or "—",
              "location": location(record), "token_count": record["token_count"],
              "chunk_id": record["chunk_id"]} for record in records]
    st.dataframe(table, use_container_width=True, hide_index=True)

    labels = {
        f"{record['filename']} | {record['heading_path'] or 'no heading'} | {location(record)} | {record['chunk_id']}": record
        for record in records
    }
    selected = labels[st.selectbox("Inspect a chunk", labels)]
    st.caption(f"Tokenizer: {selected['tokenizer']} · Document version: {selected['document_version']}")
    st.code(selected["text"], language="markdown" if selected["filename"].endswith(".md") else None)

    st.download_button("Download inspection JSON", json.dumps(records, indent=2, ensure_ascii=False),
                       "chunk-inspection.json", "application/json")
    st.download_button("Download inspection Markdown", format_inspection(records, "markdown"),
                       "chunk-inspection.md", "text/markdown")
else:
    st.caption("Upload supported documents and select Inspect documents to view chunks and source locations.")
