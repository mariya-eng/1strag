"""
document_loader.py
-------------------
Reads files (txt / pdf) from a folder and splits them into overlapping
chunks that are small enough to embed and retrieve accurately.
"""

import os
from typing import List, Dict
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(folder_path: str) -> List[Dict]:
    """Load every .txt / .pdf file in a folder into memory.

    Returns a list of {"source": filename, "text": full_text}.
    """
    docs = []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        if not os.path.isfile(full_path):
            continue

        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "txt":
            text = _read_txt(full_path)
        elif ext == "pdf":
            text = _read_pdf(full_path)
        else:
            continue  # skip unsupported file types

        if text.strip():
            docs.append({"source": filename, "text": text})

    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-based chunks.

    Word-based (not character-based) chunking keeps sentences more
    intact, which gives the LLM cleaner context at answer time.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)  # avoid infinite loop if overlap >= chunk_size

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


def load_and_chunk(folder_path: str) -> List[Dict]:
    """Convenience wrapper: load all docs in a folder and chunk them.

    Returns a list of {"source": filename, "chunk_id": int, "text": chunk}.
    """
    all_chunks = []
    for doc in load_documents(folder_path):
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
            })
    return all_chunks
