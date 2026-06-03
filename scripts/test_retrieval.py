"""
Test retrieval + save output in a readable format.

What this proves:
- Query -> local embedding (384 dims)
- Qdrant returns top-k chunks with similarity scores
- Payload fields (object_type, url, chunk_id, chunk_text) come back correctly

Output:
- output/retrieval_test_results.txt
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# --- Ensure project root is importable (fixes: No module named 'app') ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.retrieval import generate_embedding, vector_search

OUT_FILE = "output/retrieval_test_results.txt"

TEST_QUERIES = [
    "What is the application deadline for HTW Berlin?",
    "How do I apply via uni-assist?",
    "Do I need German language proof?",
]

TOP_K = 3
SNIPPET_CHARS = 350


def _safe(v: Any) -> str:
    """Convert None/empty to 'N/A' for cleaner logs."""
    if v is None:
        return "N/A"
    if isinstance(v, str) and not v.strip():
        return "N/A"
    return str(v)


def _snippet(text: str, n: int = SNIPPET_CHARS) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def format_run(query: str, vec: List[float], results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"Query: {query}")
    lines.append(f"Embedding dimension: {len(vec)}")
    lines.append(f"Results count: {len(results)}\n")

    for i, r in enumerate(results, 1):
        lines.append(f"--- Result {i} ---")
        lines.append(f"score: {_safe(r.get('score'))}")

        # payload fields (from Qdrant)
        lines.append(f"object_type: {_safe(r.get('object_type'))}")
        lines.append(f"object_id: {_safe(r.get('object_id'))}")
        lines.append(f"url: {_safe(r.get('url'))}")
        lines.append(f"chunk_id: {_safe(r.get('chunk_id'))}")

        lines.append("chunk_text: " + _snippet(r.get("chunk_text", "")))
        lines.append("")  # blank line between results

    return "\n".join(lines)


def main():
    os.makedirs("output", exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Retrieval Test Run: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        for q in TEST_QUERIES:
            vec = generate_embedding(q)
            results = vector_search(vec, top_k=TOP_K)

            block = format_run(q, vec, results)

            # print to console (like your screenshot)
            print(block)
            print("=" * 80)

            # write to file
            f.write(block + "\n")
            f.write("=" * 80 + "\n\n")

    print(f"\n✅ Saved retrieval output to: {OUT_FILE}")


if __name__ == "__main__":
    main()
