"""
rag_pipeline.py
----------------
Orchestrates the full RAG flow: embed query -> retrieve chunks from
Chroma -> generate an answer with Gemini -> log to PostgreSQL.
"""

import time
from typing import Dict

from config import TOP_K
from src.vector_store import VectorStore
from src.llm import generate_answer
from src import db


class RAGPipeline:
    def __init__(self, enable_logging: bool = True):
        self.store = VectorStore()
        self.enable_logging = enable_logging
        if self.enable_logging:
            try:
                db.init_db()
            except Exception as e:
                # Don't crash the whole app if Postgres isn't reachable —
                # just disable logging and keep answering questions.
                print(f"[warning] Postgres logging disabled: {e}")
                self.enable_logging = False

    def ask(self, query: str, top_k: int = TOP_K) -> Dict:
        start = time.time()

        contexts = self.store.search(query, top_k=top_k)
        answer = generate_answer(query, contexts)

        latency = round(time.time() - start, 2)

        if self.enable_logging:
            try:
                db.log_query(query, answer, contexts, latency)
            except Exception as e:
                print(f"[warning] failed to log query: {e}")

        return {
            "answer": answer,
            "sources": contexts,
            "latency_seconds": latency,
        }
