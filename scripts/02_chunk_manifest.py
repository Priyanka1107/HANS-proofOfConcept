# 02_chunk_manifest.py

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Tuple

# --- add project root to import path (so "app" can be imported) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import config


OUT_DIR = "output"
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")

# Sliding window settings (characters)
CHUNK_SIZE = 1000
OVERLAP = 150
STRIDE = CHUNK_SIZE - OVERLAP

os.makedirs(OUT_DIR, exist_ok=True)


def sliding_window_chunks(text: str, chunk_size: int, stride: int) -> List[Tuple[int, int, str]]:
    """
    Returns list of (char_start, char_end, chunk_text) using sliding window.
    Ensures last part of the text is included.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append((start, end, chunk))

        # Stop if we've reached the end
        if end == n:
            break

        start += stride

    return chunks


def stable_chunk_id(object_id: str, chunk_index: int, chunk_text: str) -> str:
    """
    Generates a stable chunk id. If you re-run chunking with same rules and same text,
    you get the same chunk_id.
    """
    h = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:12]
    return f"{object_id}::chunk_{chunk_index}::{h}"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    manifest_path = config.MANIFEST_PATH
    kb_version = config.KB_VERSION

    print(f"Reading manifest: {manifest_path}")
    objects = read_jsonl(manifest_path)
    print(f"Objects loaded: {len(objects)}")

    total_chunks = 0
    skipped_empty = 0

    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for obj in objects:
            object_id = obj.get("object_id")
            object_type = obj.get("object_type")
            url = obj.get("url")
            lang = obj.get("lang", "en")
            full_text = (obj.get("full_text") or "").strip()

            # Skip objects with no usable text
            if not full_text:
                skipped_empty += 1
                continue

            # Create overlapping chunks
            chunks = sliding_window_chunks(full_text, CHUNK_SIZE, STRIDE)

            for i, (char_start, char_end, chunk_text) in enumerate(chunks):
                chunk_id = stable_chunk_id(object_id, i, chunk_text)

                chunk_row = {
                    "kb_version": kb_version,
                    "chunk_id": chunk_id,
                    "object_id": object_id,
                    "object_type": object_type,
                    "url": url,
                    "lang": lang,
                    "chunk_index": i,
                    "char_start": char_start,
                    "char_end": char_end,
                    "chunk_size": CHUNK_SIZE,
                    "overlap": OVERLAP,
                    "stride": STRIDE,
                    "chunk_text": chunk_text,
                }

                out.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Wrote chunks: {CHUNKS_PATH}")
    print(f"Total chunks: {total_chunks}")
    print(f"Skipped objects with empty full_text: {skipped_empty}")


if __name__ == "__main__":
    main()
