"""
view_data.py
------------
Quick script to see everything currently stored in the Chroma vector
database — useful for debugging / checking what got indexed.

Run with:
    python view_data.py
"""

from src.vector_store import VectorStore

store = VectorStore()
count = store.count()
print(f"Total chunks stored: {count}\n")

if count == 0:
    print("No data indexed yet. Upload a document through the app first.")
else:
    # Pull everything directly from the underlying Chroma collection
    results = store.collection.get()

    for i, (doc_id, text, meta) in enumerate(zip(
        results["ids"], results["documents"], results["metadatas"]
    )):
        print(f"--- Chunk {i+1} ---")
        print(f"ID: {doc_id}")
        print(f"Source: {meta.get('source')} (chunk {meta.get('chunk_id')})")
        print(f"Text preview: {text[:200]}...")
        print()