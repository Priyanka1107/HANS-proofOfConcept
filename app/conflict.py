# app/conflict.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Tuple, Any

from app.config import config

def _parse_dt(s: str) -> datetime:
    # Accept ISO strings; fall back to "very old" if missing/bad
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1)

def detect_and_resolve_conflicts(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Input: reranked docs (list of dicts with keys like:
        id, title, content, source_url, category, object_type, last_updated,
        is_canonical (bool), priority (int)
    Output:
        final_docs (same structure, may add conflict_alternatives)
        conflicts (list of conflict descriptions)
    """

    # Group by category (or fallback group if missing)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for d in documents:
        cat = d.get("category") or "uncategorized"
        by_cat.setdefault(cat, []).append(d)

    conflicts: List[Dict[str, Any]] = []
    chosen: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for cat, docs in by_cat.items():
        if len(docs) == 1:
            leftovers.append(docs[0])
            continue

        # Detect potential "freshness conflict"
        dts = [_parse_dt(x.get("last_updated", "")) for x in docs]
        gap_days = (max(dts) - min(dts)).days if dts else 0

        # Only call it a conflict if the gap is large (default 180 days used in your doc examples)
        if gap_days > 180:
            conflicts.append({
                "type": "timestamp_conflict",
                "category": cat,
                "doc_ids": [x.get("id") for x in docs],
                "gap_days": gap_days,
                "oldest": min(dts).isoformat(),
                "newest": max(dts).isoformat(),
                "message": f"Multiple sources in '{cat}' differ in recency by {gap_days} days; using canonical/newest rule."
            })

            # Resolve: pick best doc
            def score(x: Dict[str, Any]):
                return (
                    bool(x.get("is_canonical", False)),
                    _parse_dt(x.get("last_updated", "")),
                    config.OBJECT_TYPE_PRIORITY.get(x.get("object_type", ""), 0),
                    int(x.get("priority", 0))
                )

            best = max(docs, key=score)
            best = dict(best)  # copy
            best["conflict_alternatives"] = [
                {
                    "id": x.get("id"),
                    "title": x.get("title"),
                    "last_updated": x.get("last_updated"),
                    "object_type": x.get("object_type"),
                    "reason": "lower_priority"
                }
                for x in docs if x.get("id") != best.get("id")
            ]
            chosen.append(best)
        else:
            # No meaningful conflict → keep docs as-is
            leftovers.extend(docs)

    # Return chosen conflict-resolved docs first, then others (already reranked)
    final_docs = chosen + leftovers
    return final_docs, conflicts
