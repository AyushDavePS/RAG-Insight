"""Local upload-to-cited-chat Streamlit application."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import streamlit as st

from rag_insight.bootstrap import build
from rag_insight.config import Settings
from rag_insight.embeddings import OllamaEmbedder
from rag_insight.inspection import CharacterBudgetTokenizer, format_inspection, inspect_documents


def location(value):
    if value.page is not None:
        return f"page {value.page}"
    if value.line_start is not None:
        return f"lines {value.line_start}-{value.line_end}"
    return "location unavailable"


def inspect_uploads(uploads, settings, collection):
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
    return inspect_documents(paths, settings, CharacterBudgetTokenizer()), paths


def render_sources(sources):
    for number, source in enumerate(sources, 1):
        with st.expander(f"[{number}] {source.filename} - {source.heading_path or location(source)}"):
            st.caption(f"{location(source)} | {source.source_uri or 'local upload'}")
            st.text(source.text)


def conversation_download(messages):
    records = []
    for message in messages:
        record = {"role": message["role"], "content": message["content"]}
        if "answer" in message:
            record["answer"] = asdict(message["answer"])
        records.append(record)
    return json.dumps(records, indent=2, ensure_ascii=False)


st.set_page_config(page_title="RAG Insight", page_icon="book", layout="wide")
st.title("RAG Insight")
st.caption("Upload documents, then ask grounded questions in a cited local chat.")

configured = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
base_settings = Settings(**configured)
if "collection" not in st.session_state:
    st.session_state.collection = uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Active configuration")
    st.caption(f"Vector backend: `{base_settings.vector_backend}`")
    st.caption(f"Collection: `{base_settings.chroma_collection}`")
    st.caption(f"Embedding model: `{base_settings.embedding_model}`")
    st.caption(f"Generation model: `{base_settings.llm_model}`")
    st.caption(f"Agentic retrieval: `{'enabled' if base_settings.agentic else 'disabled'}`")
    strategies = ["hybrid", "structure", "recursive"]
    strategy = st.selectbox("Chunk strategy", strategies, index=strategies.index(base_settings.chunk_strategy),
                            help="Hybrid retains structure-aware chunks and adds non-duplicate recursive fallbacks.")
    chunk_tokens = st.number_input("Chunk budget", min_value=1, value=base_settings.chunk_tokens, step=10)
    overlap_tokens = st.number_input("Overlap", min_value=0, max_value=chunk_tokens - 1,
                                     value=min(base_settings.overlap_tokens, chunk_tokens - 1), step=1)
    ocr_enabled = st.checkbox("OCR scanned PDFs with local PP-OCRv5 (CPU)", value=base_settings.ocr_enabled)
    ocr_model = st.selectbox("OCR profile", ["mobile", "server"], index=0,
                             help="Mobile is practical on CPU; server is slower and higher accuracy.")
    if st.button("Test embedding model"):
        try:
            vector = OllamaEmbedder(base_settings.embedding_model, base_settings.ollama_url).encode_batch(["smoke test"])[0]
            st.success(f"Ollama returned {len(vector)} dimensions.")
        except RuntimeError as error:
            st.error(str(error))

settings = replace(base_settings, chunk_strategy=strategy, chunk_tokens=chunk_tokens,
                   overlap_tokens=overlap_tokens, ocr_enabled=ocr_enabled, ocr_model=ocr_model)
uploads = st.file_uploader("Upload Markdown, text, or text-based PDF documents", type=["md", "txt", "pdf"],
                           accept_multiple_files=True)
if st.button("Inspect documents", type="primary", disabled=not uploads):
    try:
        records, paths = inspect_uploads(uploads, settings, st.session_state.collection)
        st.session_state.inspection, st.session_state.upload_paths = records, paths
        st.success("Documents parsed and chunked. Inspection does not call models or create an index.")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not inspect uploads: {error}")

records = st.session_state.get("inspection", [])
if records:
    with st.expander("Chunk inspection"):
        st.dataframe([
            {"filename": record["filename"], "heading": record["heading_path"] or "-",
             "location": f"page {record['page']}" if record["page"] else f"lines {record['line_start']}-{record['line_end']}",
             "tokens": record["token_count"], "chunk_id": record["chunk_id"]}
            for record in records
        ], use_container_width=True, hide_index=True)
        st.download_button("Download inspection JSON", json.dumps(records, indent=2, ensure_ascii=False),
                           "chunk-inspection.json", "application/json")
        st.download_button("Download inspection Markdown", format_inspection(records, "markdown"),
                           "chunk-inspection.md", "text/markdown")
    if st.button("Index documents", type="primary"):
        try:
            with st.spinner("Embedding and indexing documents..."):
                pipeline = build(settings, Path("data/indexes") / f"{st.session_state.collection}.sqlite")
                count = sum(pipeline.ingest(path) for path in st.session_state.upload_paths)
            st.session_state.pipeline, st.session_state.indexed = pipeline, True
            st.success(f"Indexed {count} chunks in this browser session's collection.")
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not index documents: {error}")

pipeline = st.session_state.get("pipeline")
if pipeline and st.session_state.get("indexed"):
    controls = st.columns(2)
    if controls[0].button("Reset chat"):
        st.session_state.messages = []
        st.rerun()
    controls[1].caption(f"Session collection ID: `{st.session_state.collection}`")
    confirmation = controls[1].text_input("Type the session ID above to clear its collection")
    if controls[1].button("Clear collection", type="secondary"):
        if confirmation != st.session_state.collection:
            st.error("Collection was not cleared: confirmation must match this session ID.")
        else:
            try:
                pipeline.clear_collection()
                st.session_state.messages, st.session_state.indexed = [], False
                st.success("This session's collection was cleared.")
            except RuntimeError as error:
                st.error(f"Could not clear the collection: {error}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "answer" in message:
                answer = message["answer"]
                st.caption(f"Evidence {'sufficient' if answer.sufficient else 'insufficient'}: {answer.reason}")
                render_sources(answer.sources)
                with st.expander("Tool and retrieval trace"):
                    st.json(answer.trace)
    question = st.chat_input("Ask a question about your indexed documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        history = [
            {
                "role": item["role"],
                "content": item["content"],
                **({"source_filenames": [source.filename for source in item["answer"].sources]}
                   if "answer" in item else {}),
            }
            for item in st.session_state.messages[:-1]
        ]
        with st.chat_message("assistant"), st.spinner("Retrieving grounded evidence..."):
            try:
                answer = pipeline.ask(question, history=history)
                st.write(answer.text)
                st.caption(f"Evidence {'sufficient' if answer.sufficient else 'insufficient'}: {answer.reason}")
                render_sources(answer.sources)
                with st.expander("Tool and retrieval trace"):
                    st.json(answer.trace)
                st.session_state.messages.append({"role": "assistant", "content": answer.text, "answer": answer})
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"Could not answer the question: {error}")
    if st.session_state.messages:
        st.download_button("Download conversation", conversation_download(st.session_state.messages),
                           "rag-conversation.json", "application/json")
else:
    st.info("Inspect and index at least one document to begin a cited chat.")
