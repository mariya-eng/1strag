"""
embeddings.py
-------------
Thin wrapper around the Gemini embedding API so the rest of the app
never has to think about the underlying SDK calls.
"""

from typing import List
import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_EMBED_MODEL

genai.configure(api_key=GEMINI_API_KEY)


def embed_texts(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Embed a batch of texts. Used when writing chunks into Chroma."""
    vectors = []
    for text in texts:
        result = genai.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=text,
            task_type=task_type,
        )
        vectors.append(result["embedding"])
    return vectors


def embed_query(query: str) -> List[float]:
    """Embed a single user query. Uses a different task_type than
    document embedding — Gemini optimizes the vector space slightly
    differently for queries vs. documents, which improves retrieval.
    """
    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]
