# vector_db.py
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Where we save our FAISS index + chunk list
INDEX_PATH = "faiss_index.pkl"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def init_vector_table():
    """
    Ensure an index file exists (empty) on disk.
    """
    if not os.path.exists(INDEX_PATH):
        # Create an empty index with dummy dimension
        # We'll overwrite once we have real embeddings
        empty_index = faiss.IndexFlatIP(384)
        pickle.dump(([], empty_index), open(INDEX_PATH, "wb"))

def upsert_vectors(chunks, embeddings):
    """
    Build a new FAISS index from scratch each time (simple but effective for small corpora).
    """
    # Convert embeddings to numpy float32
    embs = np.vstack([
        emb if isinstance(emb, np.ndarray) else emb.cpu().numpy()
        for emb in embeddings
    ]).astype("float32")

    # Build a new IndexFlatIP (inner product) for dot-product similarity
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # Persist chunks + index
    with open(INDEX_PATH, "wb") as f:
        pickle.dump((chunks, index), f)

def query_vectors(query: str, top_k: int = 5):
    """
    Embed the query and retrieve top_k similar chunks.
    """
    # Load existing chunks + index
    if not os.path.exists(INDEX_PATH):
        return []

    with open(INDEX_PATH, "rb") as f:
        chunks, index = pickle.load(f)

    # Embed the query
    q_emb = EMBED_MODEL.encode([query]).astype("float32")
    # Search in FAISS
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for idx in idxs[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results
