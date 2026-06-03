# app/retrieval.py
"""
Retrieval layer for the PoC:
- Generate query embedding using provider (Cohere or local)
- Vector search in Qdrant
- Optional reranking using Cohere
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import config

logger = logging.getLogger(__name__)

# -------------------------
# Local embedding model
# -------------------------
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_local_model: Optional[SentenceTransformer] = None


def _get_local_model() -> SentenceTransformer:
    global _local_model
    if _local_model is None:
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
    return _local_model


# -------------------------
# Cohere client helpers
# -------------------------
_cohere_client = None


def _get_cohere_client():
    """
    Cohere Python SDK v5 supports ClientV2.
    We try ClientV2 first; if unavailable, fall back to Client.
    """
    global _cohere_client
    if _cohere_client is not None:
        return _cohere_client

    if not config.COHERE_API_KEY:
        return None

    try:
        import cohere  # type: ignore

        # Prefer v2 client
        if hasattr(cohere, "ClientV2"):
            _cohere_client = cohere.ClientV2(api_key=config.COHERE_API_KEY)
        else:
            _cohere_client = cohere.Client(api_key=config.COHERE_API_KEY)

        return _cohere_client
    except Exception as e:
        logger.warning(f"Failed to initialize Cohere client: {e}")
        _cohere_client = None
        return None


def _extract_cohere_embedding(resp) -> List[float]:
    """
    Cohere SDK responses differ by client/version.

    We support:
    - Older: resp.embeddings is a list-of-lists -> resp.embeddings[0]
    - Newer v2: resp.embeddings is an object with .float/.int8/.binary arrays
               -> resp.embeddings.float[0]
    """
    if resp is None:
        raise RuntimeError("Cohere embed returned empty response")

    emb = getattr(resp, "embeddings", None)
    if emb is None:
        raise RuntimeError("Cohere embed response missing 'embeddings'")

    # v2 style: emb.float exists
    if hasattr(emb, "float") and emb.float is not None:
        return list(emb.float[0])

    # classic style: emb is indexable list
    try:
        return list(emb[0])
    except Exception as e:
        raise RuntimeError(f"Unsupported Cohere embeddings format: {type(emb)} ({e})")


# -------------------------
# Qdrant client (reused)
# -------------------------
qdrant_client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY
)


def _normalize(vec: List[float]) -> List[float]:
    """Normalize vector to unit length for cosine similarity."""
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return vec
    return (v / n).tolist()


def generate_embedding(text: str) -> List[float]:
    """
    Create an embedding vector for the user query.

    Routes by EMBEDDING_PROVIDER:
    - "cohere": Cohere embed API (typically 1024 dims for embed-multilingual-v3.0)
    - "local":  SentenceTransformer (384 dims typical)

    NOTE:
    - input_type="search_query" for runtime queries
    - input_type="search_document" for offline chunk embeddings
    """
    provider = (config.EMBEDDING_PROVIDER or "local").strip().lower()
    expected_dim = int(getattr(config, "EMBEDDING_DIMENSION", 0) or 0)

    if provider == "cohere":
        client = _get_cohere_client()
        if not client:
            raise RuntimeError(
                "COHERE_API_KEY not set but EMBEDDING_PROVIDER='cohere'. "
                "Add COHERE_API_KEY to .env or change EMBEDDING_PROVIDER to 'local'."
            )

        try:
            resp = client.embed(
                texts=[text],
                model=config.COHERE_EMBED_MODEL,
                input_type="search_query",
                truncate="END",
            )
            vec = _extract_cohere_embedding(resp)
            vec = _normalize(vec)
        except Exception as e:
            logger.error(f"Cohere embed failed: {e}")
            raise

        if expected_dim and len(vec) != expected_dim:
            raise RuntimeError(
                f"Cohere embedding dimension mismatch: got {len(vec)}, "
                f"expected {expected_dim}. Check EMBEDDING_DIMENSION in .env."
            )
        return vec

    # local
    model = _get_local_model()
    vecs = model.encode([text], normalize_embeddings=True)
    vec = vecs[0].tolist()

    if expected_dim and len(vec) != expected_dim:
        raise RuntimeError(
            f"Local embedding dimension mismatch: got {len(vec)}, "
            f"expected {expected_dim}. Check EMBEDDING_DIMENSION in .env."
        )
    return vec


def vector_search(query_embedding: List[float], top_k: int = 20) -> List[Dict[str, Any]]:
    """Dense vector search in Qdrant."""
    hits = qdrant_client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=top_k,
    )

    results: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        results.append({
            "id": str(h.id),
            "score": float(h.score),
            **payload,
        })

    return _normalize_payload(results)


def _normalize_payload(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize Qdrant payload to expected keys:
    - title
    - source_url (alias for url)
    - content (alias for chunk_text)
    - last_updated (fallback: last_processed/last_scraped/"")
    """
    normalized: List[Dict[str, Any]] = []
    for r in results:
        title = r.get("title", "") or ""
        source_url = r.get("url") or r.get("source_url", "") or ""
        content = r.get("chunk_text") or r.get("content", "") or ""
        last_updated = (
            r.get("last_updated")
            or r.get("last_processed")
            or r.get("last_scraped")
            or ""
        )

        normalized.append({
            **r,
            "title": title,
            "source_url": source_url,
            "content": content,
            "last_updated": last_updated,
        })
    return normalized


def keyword_search(query: str, language: str = "en", top_k: int = 20) -> List[Dict[str, Any]]:
    """Optional keyword search stub (not used in this PoC)."""
    return []


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """RRF merge (works even if keyword_results is empty)."""
    scores: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["id"])
        scores.setdefault(doc_id, {"doc": doc, "rrf_score": 0.0})
        scores[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, doc in enumerate(keyword_results):
        doc_id = str(doc["id"])
        scores.setdefault(doc_id, {"doc": doc, "rrf_score": 0.0})
        scores[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)

    fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    out: List[Dict[str, Any]] = []
    for item in fused:
        d = dict(item["doc"])
        d["rrf_score"] = item["rrf_score"]
        out.append(d)
    return out


def rerank_documents(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Cohere rerank (optional).
    If Cohere is not configured or fails, fall back to original order.
    """
    if not documents:
        return []

    client = _get_cohere_client()
    if not client:
        logger.debug("Reranking skipped: no Cohere API key")
        return documents[:top_k]

    try:
        inputs: List[str] = []
        for d in documents:
            title = d.get("title", "") or ""
            content = d.get("content") or d.get("chunk_text", "") or ""
            inputs.append(f"{title}\n\n{content[:1200]}")

        resp = client.rerank(
            model=config.RERANK_MODEL,
            query=query,
            documents=inputs,
            top_n=min(top_k, len(documents)),
            return_documents=False,
        )

        reranked: List[Dict[str, Any]] = []
        for r in resp.results:
            doc = dict(documents[r.index])
            doc["rerank_score"] = float(r.relevance_score)
            reranked.append(doc)

        return _normalize_payload(reranked)

    except Exception as e:
        logger.warning(f"Rerank failed, using fallback. Reason: {e}")
        return documents[:top_k]