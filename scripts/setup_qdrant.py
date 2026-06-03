# scripts/setup_qdrant.py
"""
Setup Qdrant collection for the HANS POC.

This script:
- Connects to Qdrant Cloud using QDRANT_URL and QDRANT_API_KEY
- Creates a collection with the correct vector dimension
- Validates dimension matches the embedding provider

Why this script exists:
- Qdrant needs to know the vector dimension in advance
- We create/verify a clean collection for the POC embeddings
- Different embedding providers use different dimensions:
  * Cohere embed-multilingual-v3.0: 1024 dimensions
  * SentenceTransformer MiniLM-L6-v2: 384 dimensions

Run:
  python scripts/setup_qdrant.py
"""

import os
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Ensure project root is on import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import config

load_dotenv()


def main():
    print("=" * 80)
    print("HANS POC - Qdrant Collection Setup")
    print("=" * 80)
    
    # Get configuration
    qdrant_url = config.QDRANT_URL
    api_key = config.QDRANT_API_KEY
    collection_name = config.QDRANT_COLLECTION
    embedding_dimension = config.EMBEDDING_DIMENSION
    embedding_provider = config.EMBEDDING_PROVIDER
    
    print(f"\n📋 Configuration:")
    print(f"   Provider: {embedding_provider}")
    print(f"   Collection: {collection_name}")
    print(f"   Dimension: {embedding_dimension}")
    print(f"   URL: {qdrant_url}")
    
    # Validate required config
    if not qdrant_url:
        raise RuntimeError("❌ QDRANT_URL missing in .env")
    if not api_key:
        raise RuntimeError("❌ QDRANT_API_KEY missing in .env")
    if not collection_name:
        raise RuntimeError("❌ QDRANT_COLLECTION missing in .env")
    
    # Parse URL for REST client
    u = urlparse(qdrant_url)
    host = u.hostname
    https = (u.scheme == "https")
    port = u.port or 6333  # Qdrant REST default
    
    print(f"\n🔗 Connecting to Qdrant Cloud...")
    print(f"   Host: {host}:{port}")
    print(f"   HTTPS: {https}")
    
    # Create client (prefer_grpc=False for cloud URL)
    try:
        client = QdrantClient(
            host=host,
            port=port,
            https=https,
            api_key=api_key,
            prefer_grpc=False
        )
        print(f"✅ Connected to Qdrant")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        raise
    
    # Check if collection exists
    print(f"\n🔍 Checking for existing collection...")
    try:
        existing_collections = [c.name for c in client.get_collections().collections]
        collection_exists = collection_name in existing_collections
    except Exception as e:
        print(f"⚠️  Failed to list collections: {e}")
        collection_exists = False
    
    if collection_exists:
        print(f"✅ Collection already exists: {collection_name}")
        
        # Get collection info to check dimension
        try:
            collection_info = client.get_collection(collection_name)
            stored_dim = collection_info.config.params.vectors.size
            
            if stored_dim != embedding_dimension:
                print(f"\n⚠️  DIMENSION MISMATCH!")
                print(f"   Stored dimension: {stored_dim}")
                print(f"   Expected dimension: {embedding_dimension}")
                print(f"   This collection cannot be used with the current EMBEDDING_PROVIDER")
                print(f"\n   To continue, either:")
                print(f"   1) Create a new collection with a different name")
                print(f"   2) Change EMBEDDING_PROVIDER/EMBEDDING_DIMENSION to match stored collection")
                print(f"\n❌ Exiting without making changes")
                sys.exit(1)
            
            print(f"✅ Collection dimension verified: {stored_dim}")
        except Exception as e:
            print(f"⚠️  Could not verify collection dimension: {e}")
    else:
        print(f"ℹ️  Collection does not exist, creating...")
        
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
            )
            print(f"✅ Created collection: {collection_name}")
            print(f"   Dimension: {embedding_dimension}")
            print(f"   Distance: COSINE")
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            raise
    
    print("\n" + "=" * 80)
    print("✅ Qdrant setup complete!")
    print("=" * 80)
    print(f"\nNext step: Run 'python scripts/03_embed_chunks.py'")


if __name__ == "__main__":
    main()

