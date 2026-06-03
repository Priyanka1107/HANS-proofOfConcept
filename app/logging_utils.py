# app/logging_utils.py
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json

from app.config import config

RESULTS_DIR = os.path.join(os.getcwd(), "results")
LOG_PATH = os.path.join(RESULTS_DIR, "query_logs.jsonl")
REGRESSION_PATH = os.path.join(RESULTS_DIR, "regression_set.jsonl")

def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def _try_pg_connect():
    if not getattr(config, "POSTGRES_URL", None):
        return None
    try:
        return psycopg2.connect(config.POSTGRES_URL)
    except Exception:
        return None

def init_db_schema():
    """Optional: create the table if Postgres is available."""
    conn = _try_pg_connect()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id SERIAL PRIMARY KEY,
        ts TIMESTAMP NOT NULL,
        query_text TEXT NOT NULL,
        language TEXT,
        retrieval_data JSONB,
        conflicts JSONB,
        answer TEXT,
        validation JSONB,
        timing JSONB,
        failure_type TEXT
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

def log_complete_query(
    query_text: str,
    language: str,
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    fused_results: List[Dict[str, Any]],
    final_docs: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    answer_data: Dict[str, Any],
    validation: Dict[str, Any],
    timing: Dict[str, float],
    original_query: str | None = None,
    standalone_query: str | None = None,
    enhanced_query: str | None = None,
    detected_intent: str | None = None,
    mode: str | None = None,
    used_memory: bool = False,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> Dict[str, Any]:
    _ensure_dirs()

    log_entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "query": {
            "text": query_text,
            "original_text": original_query or query_text,
            "standalone_text": standalone_query or query_text,
            "enhanced_text": enhanced_query or query_text,
            "language": language,
            "detected_intent": detected_intent,
            "mode": mode,
            "used_memory": used_memory,
        },
        "retrieval": {
            "vector_results": [{"id": r.get("id"), "score": r.get("score")} for r in vector_results],
            "keyword_results": [{"id": r.get("id"), "score": r.get("score")} for r in keyword_results],
            "fused_top": [{"id": r.get("id"), "rrf_score": r.get("rrf_score")} for r in fused_results[:20]],
            "final_doc_ids": [d.get("id") for d in final_docs],
        },
        "fallback": {
            "used": fallback_used,
            "reason": fallback_reason,
        },
        "conflicts": {"detected": bool(conflicts), "details": conflicts},
        "generation": {
            "answer": answer_data.get("answer"),
            "usage": answer_data.get("usage", {}),
        },
        "validation": validation,
        "timing": timing,
        "failure_type": validation.get("failure_type"),
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    conn = _try_pg_connect()
    if conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_logs (ts, query_text, language, retrieval_data, conflicts, answer, validation, timing, failure_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                datetime.fromisoformat(log_entry["timestamp"]),
                query_text,
                language,
                Json(log_entry["retrieval"]),
                Json(log_entry["conflicts"]),
                log_entry["generation"]["answer"],
                Json(log_entry["validation"]),
                Json(log_entry["timing"]),
                log_entry["failure_type"],
            ),
        )
        conn.commit()
        cur.close()
        conn.close()

    return log_entry

    # 1) Always log to JSONL
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # 2) Try to also log to Postgres (optional)
    conn = _try_pg_connect()
    if conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_logs (ts, query_text, language, retrieval_data, conflicts, answer, validation, timing, failure_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                datetime.fromisoformat(log_entry["timestamp"]),
                query_text,
                language,
                Json(log_entry["retrieval"]),
                Json(log_entry["conflicts"]),
                log_entry["generation"]["answer"],
                Json(log_entry["validation"]),
                Json(log_entry["timing"]),
                log_entry["failure_type"],
            ),
        )
        conn.commit()
        cur.close()
        conn.close()

    return log_entry

def add_to_regression_set(log_entry: Dict[str, Any]) -> None:
    """Store failing cases so you can replay them later."""
    _ensure_dirs()
    with open(REGRESSION_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
