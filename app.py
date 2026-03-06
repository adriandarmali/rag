# app.py
import streamlit as st
from pathlib import Path
import os
import time

# Import your code file as a module.
# If your file is named rag.py, use: from rag import Config, RAGEngine
from rag import Config, RAGEngine

st.set_page_config(page_title="Local RAG Chat", page_icon="📚", layout="wide")

def save_uploads_to_data_dir(uploaded_files):
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in uploaded_files:
        out_path = Config.DATA_DIR / f.name
        with open(out_path, "wb") as out:
            out.write(f.getbuffer())
        saved += 1
    return saved

def stream_ollama_answer(prompt: str):
    import ollama
    stream = ollama.chat(
        model=Config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token

@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role","content","sources"}]

if "index_ready" not in st.session_state:
    st.session_state.index_ready = engine.load_index()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📚 RAG Controls")

    st.caption("Runs fully local via Ollama.")
    st.write(f"Embed model: `{Config.EMBED_MODEL}`")
    st.write(f"LLM model: `{Config.LLM_MODEL}`")

    st.divider()
    st.subheader("Documents")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Ingest / Reindex", use_container_width=True, disabled=not uploaded):
            # Save files then rebuild index
            saved = save_uploads_to_data_dir(uploaded)
            if Config.INDEX_FILE.exists():
                os.remove(Config.INDEX_FILE)

            with st.spinner(f"Indexing {saved} file(s)…"):
                engine.build_index()

            st.session_state.index_ready = engine.load_index()
            if st.session_state.index_ready:
                st.success("Index ready.")
            else:
                st.error("Index failed. Check terminal logs.")

    with col2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []

    st.divider()
    top_k = st.slider("Final contexts (FINAL_K)", 1, 10, int(Config.FINAL_K))
    rerank = st.toggle("Rerank (slow)", value=bool(Config.RERANK_ENABLED))

    # Apply runtime overrides (safe for demo)
    Config.FINAL_K = int(top_k)
    Config.RERANK_ENABLED = bool(rerank)

# ---------------- Main ----------------
st.title("Local RAG Chat")
st.write("Upload docs, ingest, then ask questions. Answers use only retrieved context.")

if not st.session_state.index_ready:
    st.warning("No index loaded. Upload docs in the sidebar and click **Ingest / Reindex**.")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. Retrieved chunk**")
                    st.write(s)

prompt = st.chat_input("Ask a question…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.index_ready:
        with st.chat_message("assistant"):
            st.markdown("Upload docs and ingest first.")
        st.session_state.messages.append({"role": "assistant", "content": "Upload docs and ingest first."})
    else:
        with st.chat_message("assistant"):
            t0 = time.time()
            with st.spinner("Retrieving…"):
                contexts = engine.retrieve(prompt)

            context_text = "\n\n---\n\n".join(contexts)
            rag_prompt = (
                "Answer the question using ONLY the context below.\n"
                "If the context does not contain the answer, say you don't know.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION:\n{prompt}\n\n"
                "ANSWER:"
            )

            # Stream output to UI
            answer = st.write_stream(stream_ollama_answer(rag_prompt))

            # Save message with sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer if isinstance(answer, str) else str(answer),
                "sources": contexts,
                "meta": {"latency_ms": int((time.time() - t0) * 1000)},
            })