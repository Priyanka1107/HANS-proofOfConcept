# scripts/04_ingest_chunks_qdrant.py
"""
Ingest chunk embeddings into Qdrant.

Why this file exists:
- chunks.jsonl contains metadata + chunk_text for each chunk
- embeddings.jsonl contains vectors for each chunk_id
- Qdrant stores vectors + payload (metadata) for retrieval

IMPORTANT:
Qdrant point IDs must be INT or UUID.
Our chunk_id is a long string, so we convert chunk_id -> UUID (stable).
"""

import os
import sys
import json
import uuid
from typing import Dict, Any, List

# Ensure project root is on import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.config import config

OUT_DIR = "output"
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
EMBEDDINGS_PATH = os.path.join(OUT_DIR, "embeddings.jsonl")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Convert arbitrary chunk_id string to a *stable* UUID.
    - Stable means: same chunk_id => same UUID every run
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def main():
    print("=" * 80)
    print("HANS POC - Ingest Embeddings to Qdrant")
    print("=" * 80)
    
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"❌ Missing {CHUNKS_PATH}. Run scripts/02_chunk_manifest.py first.")

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"❌ Missing {EMBEDDINGS_PATH}. Run scripts/03_embed_chunks.py first.")

    print(f"\n📖 Loading chunks from {CHUNKS_PATH}...")
    chunks = read_jsonl(CHUNKS_PATH)
    print(f"✅ Loaded {len(chunks)} chunks")

    print(f"\n📦 Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings_rows = read_jsonl(EMBEDDINGS_PATH)
    print(f"✅ Loaded {len(embeddings_rows)} embeddings")

    # Build a lookup: chunk_id -> embedding vector
    emb_by_chunk_id: Dict[str, List[float]] = {}
    for r in embeddings_rows:
        emb_by_chunk_id[r["chunk_id"]] = r["embedding"]

    # Connect Qdrant
    print(f"\n🔗 Connecting to Qdrant...")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    print(f"✅ Connected to Qdrant")

    points: List[PointStruct] = []

    print(f"\n🔧 Preparing points for ingestion...")
    for c in chunks:
        chunk_id = c["chunk_id"]

        # Skip if embedding missing (should not happen if embed finished)
        if chunk_id not in emb_by_chunk_id:
            continue

        vector = emb_by_chunk_id[chunk_id]

        # Qdrant point ID must be UUID or int
        point_id = chunk_id_to_uuid(chunk_id)

        # Payload = metadata stored alongside vector for retrieval and reranking
        # Build normalized payload with all expected keys
        payload = {
            # IDs and version
            "kb_version": c.get("kb_version", config.KB_VERSION),
            "chunk_id": chunk_id,
            "object_id": c.get("object_id", ""),
            
            # Metadata
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "object_type": c.get("object_type", "general_info"),
            "lang": c.get("lang", "en"),
            
            # Chunk info
            "chunk_index": c.get("chunk_index", 0),
            "char_start": c.get("char_start", 0),
            "char_end": c.get("char_end", 0),
            
            # Content (at retrieval time, this becomes "content" key for normalization)
            "chunk_text": c.get("chunk_text", ""),
            
            # Timestamps (used for conflict resolution and freshness)
            "last_updated": c.get("last_updated", ""),
            "last_processed": c.get("last_processed", ""),
            "last_scraped": c.get("last_scraped", ""),
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    print(f"✅ Prepared {len(points)} points")
    
    print(f"\n📤 Uploading to Qdrant collection: {config.QDRANT_COLLECTION}...")
    try:
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        print(f"✅ Ingested {len(points)} points into Qdrant")
    except Exception as e:
        print(f"❌ Failed to ingest: {e}")
        raise

    print("\n" + "=" * 80)
    print("✅ Qdrant ingestion complete!")
    print("=" * 80)
    print(f"\nNext step: Run 'uvicorn app.main:app --reload' to start the API")


if __name__ == "__main__":
    main()
