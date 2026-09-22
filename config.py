"""
config.py
---------
Central place for all settings. Values are read from environment
variables (or a local .env file) so secrets never get hard-coded
into the source code.
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file (if present) into the environment
load_dotenv()

# ---- Gemini (LLM + embeddings) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

# ---- Chroma (vector store) ----
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")

# ---- Chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))

# ---- PostgreSQL (query/answer logging) ----
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_logs")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
