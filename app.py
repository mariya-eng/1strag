"""
app.py
------
Streamlit UI for the RAG assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore
from src.document_loader import load_and_chunk

st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="wide")

# Cache the pipeline so it isn't rebuilt on every interaction/rerun
@st.cache_resource
def get_pipeline():
    return RAGPipeline(enable_logging=True)


pipeline = get_pipeline()

st.title("🔎 RAG Assistant")
st.caption("Chroma (vectors) · Gemini (embeddings + LLM) · PostgreSQL (logging)")

# ---------------- Sidebar: document upload / ingestion ----------------
with st.sidebar:
    st.header("📄 Document Index")

    store = VectorStore()
    st.metric("Chunks indexed", store.count())

    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files to add to the index",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest uploaded files"):
        import os
        os.makedirs("data", exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join("data", f.name), "wb") as out:
                out.write(f.getbuffer())

        with st.spinner("Embedding and indexing..."):
            chunks = load_and_chunk("data")
            store.add_chunks(chunks)
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
        st.rerun()

    if st.button("⚠️ Reset entire index"):
        store.reset()
        st.warning("Index cleared.")
        st.rerun()

# ---------------- Main: chat interface ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (chunk {s['chunk_id']}, distance {s['distance']:.3f})")

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.ask(query)
        st.markdown(result["answer"])
        st.caption(f"⏱ {result['latency_seconds']}s")
        if result["sources"]:
            with st.expander("Sources used"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}** (chunk {s['chunk_id']}, distance {s['distance']:.3f})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
