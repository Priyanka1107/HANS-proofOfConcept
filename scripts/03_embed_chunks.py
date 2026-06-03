# scripts/03_embed_chunks.py
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Set

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import config

OUT_DIR = "output"
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
EMBEDDINGS_PATH = os.path.join(OUT_DIR, "embeddings.jsonl")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _normalize(vec: List[float]) -> List[float]:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return vec
    return (v / n).tolist()


def _extract_cohere_embeddings(resp) -> List[List[float]]:
    """
    Cohere v5 compatibility shim.

    v5 returns:
        resp.embeddings.float

    Older SDKs returned:
        resp.embeddings -> List[List[float]]
    """
    emb = getattr(resp, "embeddings", None)
    if emb is None:
        raise RuntimeError("Cohere embed response missing 'embeddings' field")

    # Cohere v5 (most common)
    if hasattr(emb, "float") and emb.float is not None:
        return emb.float

    # Sometimes dict-like
    if isinstance(emb, dict):
        if "float" in emb:
            return emb["float"]
        return next(iter(emb.values()))

    # Older SDK: already list
    if isinstance(emb, list):
        return emb

    raise RuntimeError(f"Unknown Cohere embeddings format: {type(emb)}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_existing_embeddings(path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                cid = obj.get("chunk_id")
                if cid:
                    done.add(str(cid))
            except Exception:
                continue
    return done


def _is_rate_limit_error(msg: str) -> bool:
    m = (msg or "").lower()
    return "rate limit" in m or "429" in m or "too many requests" in m


def _is_transient_error(msg: str) -> bool:
    m = (msg or "").lower()
    return any(x in m for x in ["timeout", "connection", "502", "503", "504"])


# --------------------------------------------------
# Cohere embedding
# --------------------------------------------------

def embed_with_cohere_batch(
    cohere_client,
    texts: List[str],
    batch_size: int,
    max_retries: int = 10,
    min_wait_seconds: int = 15,
    max_wait_seconds: int = 120,
) -> List[List[float]]:

    all_embeddings: List[List[float]] = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = texts[batch_start:batch_end]

        attempt = 0
        while True:
            attempt += 1
            try:
                print(f"  Embedding batch {batch_start // batch_size + 1} ({batch_start}-{batch_end})...")

                resp = cohere_client.embed(
                    texts=batch,
                    model=config.COHERE_EMBED_MODEL,
                    input_type="search_document",
                    truncate="END",
                )

                raw_embeddings = _extract_cohere_embeddings(resp)
                embeddings = [_normalize(vec) for vec in raw_embeddings]
                all_embeddings.extend(embeddings)
                break

            except Exception as e:
                msg = str(e)

                if _is_rate_limit_error(msg) or _is_transient_error(msg):
                    if attempt >= max_retries:
                        raise

                    wait = min_wait_seconds * (2 ** (attempt - 1))
                    wait = int(min(wait, max_wait_seconds))

                    print(f"⚠️ Retry {attempt}/{max_retries} after error: {e}")
                    print(f"⏳ Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                raise

    return all_embeddings


# --------------------------------------------------
# Main
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    p.add_argument("--batch-size", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("HANS POC - Unified Embedding Generation")
    print("=" * 80)

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}")

    provider = config.EMBEDDING_PROVIDER.strip().lower()
    expected_dim = int(config.EMBEDDING_DIMENSION)

    chunks = read_jsonl(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    if args.reset and os.path.exists(EMBEDDINGS_PATH):
        os.remove(EMBEDDINGS_PATH)
        print("Reset: deleted existing embeddings")

    done_chunk_ids = load_existing_embeddings(EMBEDDINGS_PATH)

    work_chunk_ids: List[str] = []
    work_texts: List[str] = []

    for c in chunks:
        cid = str(c.get("chunk_id"))
        txt = (c.get("chunk_text") or "").strip()

        if not cid or not txt:
            continue
        if cid in done_chunk_ids:
            continue

        work_chunk_ids.append(cid)
        work_texts.append(txt)

    if not work_texts:
        print("Nothing to embed.")
        return

    print(f"Embedding {len(work_texts)} chunks using {provider}")

    # -----------------------------
    # Cohere
    # -----------------------------
    if provider == "cohere":
        if not config.COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY missing")

        import cohere
        co = cohere.ClientV2(api_key=config.COHERE_API_KEY)

        embeddings = embed_with_cohere_batch(
            cohere_client=co,
            texts=work_texts,
            batch_size=args.batch_size,
        )

        embedding_model = f"cohere::{config.COHERE_EMBED_MODEL}"

    # -----------------------------
    # Local
    # -----------------------------
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(work_texts, normalize_embeddings=True)
        embeddings = [v.tolist() for v in vectors]
        embedding_model = "local::sentence-transformers/all-MiniLM-L6-v2"

    if len(embeddings[0]) != expected_dim:
        raise RuntimeError(
            f"Dimension mismatch: got {len(embeddings[0])}, expected {expected_dim}"
        )

    rows = []
    for cid, vec in zip(work_chunk_ids, embeddings):
        rows.append({
            "kb_version": config.KB_VERSION,
            "chunk_id": cid,
            "embedding_model": embedding_model,
            "embedding": vec,
        })

    append_jsonl(EMBEDDINGS_PATH, rows)

    print(f"Appended {len(rows)} embeddings")
    print("Done.")


if __name__ == "__main__":
    main()