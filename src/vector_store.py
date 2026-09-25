"""
vector_store.py
----------------
Wraps Qdrant so the app can add and search document chunks without
touching the Qdrant API directly. Qdrant Cloud stores data remotely,
so it survives app restarts/redeploys — unlike a local Chroma folder
on a host with no persistent disk.
"""

import uuid
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from src.embeddings import embed_texts, embed_query


def _point_id(source: str, chunk_id: int) -> str:
    """Deterministic ID so re-ingesting the same chunk overwrites it
    instead of creating a duplicate point."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}-{chunk_id}"))


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION

    def _collection_exists(self) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return self.collection_name in existing

    def _ensure_collection(self, vector_size: int):
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_chunks(self, chunks: List[Dict]):
        """chunks: [{"source": ..., "chunk_id": ..., "text": ...}, ...]"""
        if not chunks:
            return

        batch_size = 50
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c["text"] for c in batch]
            vectors = embed_texts(texts)

            self._ensure_collection(vector_size=len(vectors[0]))

            points = [
                PointStruct(
                    id=_point_id(c["source"], c["chunk_id"]),
                    vector=vector,
                    payload={
                        "source": c["source"],
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                    },
                )
                for c, vector in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int) -> List[Dict]:
        if not self._collection_exists():
            return []

        query_vector = embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        hits = []
        for r in results:
            hits.append({
                "text": r.payload.get("text"),
                "source": r.payload.get("source"),
                "chunk_id": r.payload.get("chunk_id"),
                "distance": round(1 - r.score, 4),
            })
        return hits

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        return self.client.count(collection_name=self.collection_name, exact=True).count

    def list_documents(self) -> List[Dict]:
        if not self._collection_exists() or self.count() == 0:
            return []

        counts: Dict[str, int] = {}
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=200,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                source = p.payload.get("source", "unknown")
                counts[source] = counts.get(source, 0) + 1
            if next_offset is None:
                break

        return [
            {"source": source, "chunks": chunk_count}
            for source, chunk_count in sorted(counts.items())
        ]

    def reset(self):
        if self._collection_exists():
            self.client.delete_collection(self.collection_name)
