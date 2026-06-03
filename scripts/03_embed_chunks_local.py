# scripts/03_embed_chunks_local.py
"""
Generate embeddings for chunked documents using a local SentenceTransformer model.

Input:
- output/chunks.jsonl (from scripts/02_chunk_manifest.py)

Output:
- output/embeddings.jsonl (one row per chunk_id with embedding vector)

Why:
- This is a one-time preprocessing step for the PoC
- After this, embeddings are ingested into Qdrant for retrieval
"""

import os
import sys
import json
from typing import Dict, Any, List

# Ensure project root is on import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import config

from sentence_transformers import SentenceTransformer

OUT_DIR = "output"
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
EMBEDDINGS_PATH = os.path.join(OUT_DIR, "embeddings.jsonl")

LOCAL_MODEL_MAP = {
    "local-minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chunks_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run scripts/02_chunk_manifest.py first.")

    model_key = (config.EMBEDDING_MODEL or "").strip()
    if model_key not in LOCAL_MODEL_MAP:
        raise RuntimeError(
            f"Unsupported local model key '{model_key}'. "
            f"Set EMBEDDING_MODEL to one of: {list(LOCAL_MODEL_MAP.keys())}"
        )

    expected_dim = int(config.EMBEDDING_DIMENSION)
    model_name = LOCAL_MODEL_MAP[model_key]

    print(f"Loading local embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Reading chunks: {CHUNKS_PATH}")
    chunks = read_jsonl(CHUNKS_PATH)
    print(f"Chunks loaded: {len(chunks)}")

    # Start fresh each run
    if os.path.exists(EMBEDDINGS_PATH):
        os.remove(EMBEDDINGS_PATH)

    BATCH_SIZE = 64  # safe for CPU; increase if you want

    total_written = 0
    for batch in chunks_list(chunks, BATCH_SIZE):
        texts = [c["chunk_text"] for c in batch]
        chunk_ids = [c["chunk_id"] for c in batch]

        vectors = model.encode(texts, normalize_embeddings=True)

        # Sanity check embedding size
        if len(vectors[0]) != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {len(vectors[0])}, expected {expected_dim}. "
                f"Check EMBEDDING_DIMENSION and local model."
            )

        out_rows = []
        for cid, vec in zip(chunk_ids, vectors):
            out_rows.append({
                "kb_version": config.KB_VERSION,
                "chunk_id": cid,
                "embedding_model": model_key,
                "embedding": vec.tolist(),
            })

        append_jsonl(EMBEDDINGS_PATH, out_rows)
        total_written += len(out_rows)
        print(f"Embedded: {total_written}/{len(chunks)}")

    print(f"✅ Done. Wrote embeddings: {EMBEDDINGS_PATH}")
    print(f"Total embeddings: {total_written}")

if __name__ == "__main__":
    main()
