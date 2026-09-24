"""
vector_store.py
----------------
Wraps ChromaDB so the app can add and search document chunks without
touching the Chroma API directly. Chroma persists to disk so the
index survives between runs.
"""

from typing import List, Dict
import chromadb

from config import CHROMA_DIR, CHROMA_COLLECTION
from src.embeddings import embed_texts, embed_query


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict]):
        """chunks: [{"source": ..., "chunk_id": ..., "text": ...}, ...]"""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        ids = [f"{c['source']}-{c['chunk_id']}" for c in chunks]
        metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

        # Embed in manageable batches so one huge document doesn't
        # blow past API payload limits.
        batch_size = 50
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_ids = ids[start:start + batch_size]
            batch_meta = metadatas[start:start + batch_size]
            batch_vectors = embed_texts(batch_texts)

            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_vectors,
                documents=batch_texts,
                metadatas=batch_meta,
            )

    def search(self, query: str, top_k: int) -> List[Dict]:
        query_vector = embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": text,
                "source": meta.get("source"),
                "chunk_id": meta.get("chunk_id"),
                "distance": dist,
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> List[Dict]:
        """Return a summary of every unique source document currently indexed,
        with how many chunks each one contributed. Used to show an upload
        history in the UI."""
        if self.count() == 0:
            return []

        results = self.collection.get()
        metadatas = results.get("metadatas", [])

        counts: Dict[str, int] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1

        return [
            {"source": source, "chunks": chunk_count}
            for source, chunk_count in sorted(counts.items())
        ]

    def reset(self):
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
