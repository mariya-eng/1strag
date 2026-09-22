"""
ingest.py
---------
Run this once (or whenever you add new documents) to load files from
the data/ folder, chunk them, embed them, and store them in Chroma.

Usage:
    python ingest.py
    python ingest.py --folder data --reset
"""

import argparse
from src.document_loader import load_and_chunk
from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--folder", default="data", help="Folder containing .txt / .pdf files")
    parser.add_argument("--reset", action="store_true", help="Clear the collection before ingesting")
    args = parser.parse_args()

    print(f"Loading and chunking documents from '{args.folder}'...")
    chunks = load_and_chunk(args.folder)
    print(f"Created {len(chunks)} chunks from the source documents.")

    if not chunks:
        print("No chunks to index. Add .txt or .pdf files to the data/ folder first.")
        return

    store = VectorStore()
    if args.reset:
        print("Resetting existing collection...")
        store.reset()

    print("Embedding and storing chunks in Chroma (this calls the Gemini API)...")
    store.add_chunks(chunks)

    print(f"Done. Collection now has {store.count()} chunks.")


if __name__ == "__main__":
    main()
