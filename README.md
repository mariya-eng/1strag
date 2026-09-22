# RAG Assistant

A Retrieval-Augmented Generation app built with:
- **Vector store:** ChromaDB (local, persistent)
- **UI / app framework:** Python + Streamlit
- **Embeddings + LLM:** Google Gemini API
- **Logging:** PostgreSQL (stores every question, answer, sources, latency)

## Project structure

```
rag_project/
├── app.py                  # Streamlit UI (chat interface + upload)
├── ingest.py                # CLI script to index documents from data/
├── config.py                 # All settings, read from .env
├── requirements.txt
├── .env.example              # Copy to .env and fill in your keys
├── data/                     # Put your .txt / .pdf files here
├── chroma_db/                 # Chroma's persistent storage (auto-created)
└── src/
    ├── document_loader.py    # Reads + chunks .txt / .pdf files
    ├── embeddings.py          # Gemini embedding calls
    ├── vector_store.py        # Chroma add/search wrapper
    ├── llm.py                 # Gemini answer generation
    ├── db.py                  # PostgreSQL logging
    └── rag_pipeline.py        # Ties retrieval + generation + logging together
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add:
   - Your `GEMINI_API_KEY` (get one from https://aistudio.google.com/apikey)
   - Your PostgreSQL connection details (`POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`)

3. **Create the PostgreSQL database** (if it doesn't exist yet)
   ```bash
   createdb rag_logs
   ```
   The `query_logs` table is created automatically the first time the app runs.

4. **Add your documents**
   Drop `.txt` or `.pdf` files into the `data/` folder.

5. **Build the vector index**
   ```bash
   python ingest.py
   ```
   Re-run this any time you add new documents. Use `--reset` to rebuild from scratch:
   ```bash
   python ingest.py --reset
   ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`. You can also upload files directly from the sidebar instead of using `ingest.py`.

## How it works

1. **Chunking** — documents are split into ~800-word overlapping chunks (`document_loader.py`) so retrieval stays precise.
2. **Embedding** — each chunk is embedded with Gemini's `text-embedding-004` model and stored in Chroma with cosine similarity (`embeddings.py`, `vector_store.py`).
3. **Retrieval** — a user's question is embedded the same way and the top-K most similar chunks are pulled from Chroma.
4. **Generation** — those chunks are placed into a prompt and sent to Gemini (`gemini-2.0-flash` by default) which is instructed to answer only from the given context (`llm.py`).
5. **Logging** — every question, answer, source chunks, and response time are saved to PostgreSQL (`db.py`) for review later.

## Notes on efficiency

- Embeddings are batched (50 chunks per API call) during ingestion to avoid excessive round trips.
- Chroma is used in persistent mode, so you don't need to re-embed documents every time you restart the app.
- The Streamlit pipeline object is cached with `@st.cache_resource` so the Chroma client isn't recreated on every rerun.
- Postgres logging failures are caught and logged as warnings rather than crashing the app — the chat still works even if the DB is temporarily unreachable.

## Customizing

- Change `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` in `.env` to tune retrieval quality vs. speed.
- Swap `GEMINI_LLM_MODEL` to a different Gemini model (e.g. `gemini-2.0-pro`) for higher-quality but slower answers.
- Add support for more file types by extending `document_loader.py`.
