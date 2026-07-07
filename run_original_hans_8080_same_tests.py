"""
Run the final paired evaluation inputs against the ORIGINAL HANS backend.

Purpose:
- Reuse the same final evaluation input sets used for the final v4 paired evaluation.
- Send them to an older/original HANS backend, usually through /ask.
- Save outputs for manual comparison against the final HANS PoC results.

How to run from the v4 project root:
    $env:ORIGINAL_HANS_URL="http://127.0.0.1:8080"
    $env:ORIGINAL_HANS_ENDPOINT="/ask"
    python run_original_hans_8080_same_tests.py

Optional:
    $env:ORIGINAL_HANS_ENDPOINT="/query"
    $env:ORIGINAL_HANS_MODE="conversational"   # only if original backend accepts mode
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: requests is not installed. Run: pip install requests")
    sys.exit(1)

ROOT = Path.cwd()
EVAL_DIR = ROOT / "evaluation"
OUT_CSV = EVAL_DIR / "original_hans_same_tests_results.csv"
OUT_JSON = EVAL_DIR / "original_hans_same_tests_results.json"

ORIGINAL_HANS_URL = os.getenv("ORIGINAL_HANS_URL", "http://127.0.0.1:8080").rstrip("/")
ORIGINAL_HANS_ENDPOINT = os.getenv("ORIGINAL_HANS_ENDPOINT", "/ask")
ORIGINAL_HANS_MODE = os.getenv("ORIGINAL_HANS_MODE", "").strip()
TIMEOUT_SECONDS = int(os.getenv("ORIGINAL_HANS_TIMEOUT", "120"))

TOPIC_KEYWORDS = {
    "application_before_graduation": ["before graduation", "final transcript", "final certificate", "provisional", "vorläufig", "abschlusszeugnis"],
    "final_certificate_submission": ["final certificate", "final transcript", "submit", "nachreichen", "abschlusszeugnis"],
    "english_language_requirements": ["english", "toefl", "ielts", "toeic", "cambridge", "b2", "c1"],
    "german_language_requirements": ["german", "deutsch", "telc", "goethe", "testdaf", "b2", "c1"],
    "language_requirements": ["language", "english", "german", "sprache", "sprach"],
    "language_of_instruction": ["taught", "language of instruction", "english", "german", "unterricht", "sprache"],
    "application_fee": ["application fee", "fee", "uni-assist", "handling fee", "gebühr"],
    "tuition_fees": ["tuition", "study fees", "fees", "semester fee", "semester contribution", "studiengebühren"],
    "semester_contribution": ["semester contribution", "semester fee", "public transport", "semesterticket", "semesterbeitrag"],
    "application_route": ["apply", "application portal", "uni-assist", "hochschulstart", "route", "bewerbung"],
    "admission_requirements": ["admission", "requirements", "entrance qualification", "zulassung", "voraussetzung"],
    "qualification_recognition": ["recognition", "anabin", "certificate", "qualification", "vpd", "zeugnis"],
    "motivation_letter": ["motivation", "letter", "motivationsschreiben"],
    "work_experience": ["work experience", "professional experience", "qualified professional", "berufserfahrung"],
    "required_documents": ["documents", "certificate", "transcript", "upload", "unterlagen", "dokumente"],
    "study_format": ["on-campus", "full-time", "part-time", "online", "format", "campus", "vollzeit"],
    "start_semester": ["summer semester", "winter semester", "start", "begin", "semesterstart"],
    "application_deadline": ["deadline", "application period", "apply by", "frist", "bewerbungszeitraum"],
    "credit_recognition": ["credit", "recognition", "transfer", "ects", "anerkennung"],
    "certified_translations": ["translation", "certified", "beglaub", "übersetzung"],
    "grade_conversion": ["grade", "conversion", "gpa", "notenumrechnung"],
    "accommodation": ["accommodation", "housing", "student residence", "wohnheim"],
    "fees": ["fee", "tuition", "semester contribution", "semester fee", "gebühr"],
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def extract_text(resp_json: Any) -> str:
    """Try common response shapes from old/new HANS backends."""
    if isinstance(resp_json, str):
        return resp_json
    if not isinstance(resp_json, dict):
        return json.dumps(resp_json, ensure_ascii=False)
    keys = [
        "answer",
        "response",
        "draft",
        "output",
        "text",
        "message",
        "result",
        "generated_answer",
        "final_answer",
    ]
    for k in keys:
        v = resp_json.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Nested common shapes
    for k in ["data", "payload"]:
        v = resp_json.get(k)
        if isinstance(v, dict):
            t = extract_text(v)
            if t:
                return t
    return json.dumps(resp_json, ensure_ascii=False)


def extract_sources(resp_json: Any) -> Any:
    if isinstance(resp_json, dict):
        for k in ["sources", "citations", "references", "docs", "source_documents"]:
            if k in resp_json:
                return resp_json[k]
    return None


def count_citations(text: str, sources: Any) -> int:
    doc_refs = set(re.findall(r"\[Doc\s*\d+\]", text or "", flags=re.IGNORECASE))
    if doc_refs:
        return len(doc_refs)
    if isinstance(sources, list):
        return len(sources)
    if isinstance(sources, dict):
        return len(sources)
    return 0


def topic_keyword_coverage(output_text: str, expected_topics: List[str]) -> Tuple[List[str], float]:
    text = (output_text or "").lower()
    missing: List[str] = []
    for topic in expected_topics or []:
        kws = TOPIC_KEYWORDS.get(topic, [])
        if kws and not any(kw.lower() in text for kw in kws):
            missing.append(topic)
        elif not kws:
            # Unknown topic: do not mark missing by keyword heuristic
            pass
    denom = len(expected_topics or [])
    if denom == 0:
        return [], 100.0
    return missing, round(100.0 * (denom - len(missing)) / denom, 1)


def build_payloads(query: str, session_id: str, case_id: str, turn_index: int) -> List[Dict[str, Any]]:
    """Try several common payload schemas because older HANS versions may differ."""
    base_variants: List[Dict[str, Any]] = [
        # Original baseline_copy UI documents POST /ask with {"q": "..."}.
        # Put the strict original shape first; if the backend accepts extra fields,
        # the following variants also test session-like payloads.
        {"q": query},
        {"q": query, "session_id": session_id},
        {"query": query, "session_id": session_id},
        {"question": query, "session_id": session_id},
        {"message": query, "session_id": session_id},
        {"input": query, "session_id": session_id},
        {"text": query, "session_id": session_id},
    ]
    if ORIGINAL_HANS_MODE:
        with_mode = []
        for p in base_variants:
            q = dict(p)
            q["mode"] = ORIGINAL_HANS_MODE
            with_mode.append(q)
        return with_mode + base_variants
    return base_variants


def call_original(query: str, session_id: str, case_id: str, turn_index: int) -> Dict[str, Any]:
    url = ORIGINAL_HANS_URL + ORIGINAL_HANS_ENDPOINT
    last_error = ""
    start = time.time()
    for payload in build_payloads(query, session_id, case_id, turn_index):
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
            elapsed = round(time.time() - start, 2)
            if r.status_code >= 400:
                last_error = f"HTTP {r.status_code}: {r.text[:500]}"
                continue
            try:
                data = r.json()
            except Exception:
                data = {"answer": r.text}
            out = extract_text(data)
            sources = extract_sources(data)
            return {
                "http_ok": True,
                "http_status": r.status_code,
                "payload_used": json.dumps(payload, ensure_ascii=False),
                "response_time_seconds": elapsed,
                "raw_response": data,
                "output_text": out,
                "source_count": len(sources) if isinstance(sources, (list, dict)) else "",
                "citation_count": count_citations(out, sources),
                "error": "",
            }
        except Exception as e:
            last_error = repr(e)
            continue
    return {
        "http_ok": False,
        "http_status": "",
        "payload_used": "",
        "response_time_seconds": round(time.time() - start, 2),
        "raw_response": {},
        "output_text": "",
        "source_count": "",
        "citation_count": 0,
        "error": last_error or "Unknown request failure",
    }


def first_turn_text(case: Dict[str, Any]) -> str:
    turns = case.get("turns") or []
    if turns and isinstance(turns[0], str):
        return turns[0]
    if turns and isinstance(turns[0], dict):
        return turns[0].get("email_text") or turns[0].get("query") or turns[0].get("question") or ""
    return case.get("email_text") or case.get("question") or ""


def run_case(row_base: Dict[str, Any], input_text: str, session_id: str, turn_index: int, expected_topics: List[str]) -> Dict[str, Any]:
    result = call_original(input_text, session_id=session_id, case_id=row_base.get("case_id", ""), turn_index=turn_index)
    missing, topic_cov = topic_keyword_coverage(result["output_text"], expected_topics)
    auto_attention_reasons = []
    if not result["http_ok"]:
        auto_attention_reasons.append("http_error")
    if missing:
        auto_attention_reasons.append("keyword_missing_topics=" + ", ".join(missing))
    if result["citation_count"] == 0:
        auto_attention_reasons.append("no_detected_citations")

    return {
        **row_base,
        "turn_index": turn_index,
        "input_text": input_text,
        "expected_topics": ", ".join(expected_topics or []),
        "keyword_missing_topics": ", ".join(missing),
        "keyword_topic_coverage_percent": topic_cov,
        "citation_count": result["citation_count"],
        "source_count": result["source_count"],
        "response_time_seconds": result["response_time_seconds"],
        "http_ok": "Yes" if result["http_ok"] else "No",
        "http_status": result["http_status"],
        "needs_manual_attention": "Yes" if auto_attention_reasons else "No",
        "attention_reasons": " | ".join(auto_attention_reasons),
        "payload_used": result["payload_used"],
        "error": result["error"],
        "output_text": result["output_text"],
        "raw_response_json": json.dumps(result["raw_response"], ensure_ascii=False),
    }


def main() -> None:
    EVAL_DIR.mkdir(exist_ok=True)
    print("Original HANS same-input evaluation")
    print(f"Backend: {ORIGINAL_HANS_URL}{ORIGINAL_HANS_ENDPOINT}")
    print(f"Mode parameter: {ORIGINAL_HANS_MODE or '(not sent)'}")

    email_cases = load_json(EVAL_DIR / "final_email_provider_cases.json", [])
    thread_cases = load_json(EVAL_DIR / "final_email_thread_cases.json", [])
    qa_cases = load_json(EVAL_DIR / "final_qa_memory_cases.json", [])
    qa_ref_selected = load_json(EVAL_DIR / "hans_qa_reference_selected.json", [])

    rows: List[Dict[str, Any]] = []

    # 1) Final email provider cases as plain original HANS queries
    for case in email_cases:
        text = first_turn_text(case)
        base = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "comparison_group_id": f"original_email_as_query::{case.get('case_id')}",
            "source_case_id": case.get("source_case_id", case.get("case_id")),
            "case_id": case.get("case_id"),
            "case_type": "original_email_as_query",
            "mode": "original_hans_query",
            "expected_programme": case.get("expected_programme"),
            "title": case.get("title", ""),
            "reference_answer": "",
            "staff_comment": "",
        }
        rows.append(run_case(base, text, f"orig_email_{case.get('case_id')}", 1, case.get("expected_topics", [])))

    # 2) Email thread cases as session turns, if original supports session_id
    for case in thread_cases:
        turns = case.get("turns") or []
        session_id = f"orig_thread_{case.get('case_id')}"
        for idx, turn in enumerate(turns, start=1):
            if isinstance(turn, dict):
                text = turn.get("email_text") or turn.get("query") or turn.get("question") or ""
                exp_topics = turn.get("expected_topics") or case.get("expected_topics") or []
                exp_followup = turn.get("expected_followup_type", "")
            else:
                text = str(turn)
                exp_topics = case.get("expected_topics") or []
                exp_followup = ""
            base = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "comparison_group_id": f"original_email_thread::{case.get('case_id')}",
                "source_case_id": case.get("case_id"),
                "case_id": case.get("case_id"),
                "case_type": "original_email_thread_as_query",
                "mode": "original_hans_query_session",
                "expected_programme": case.get("expected_programme"),
                "expected_followup_type": exp_followup,
                "title": case.get("title", ""),
                "reference_answer": "",
                "staff_comment": "",
            }
            rows.append(run_case(base, text, session_id, idx, exp_topics))

    # 3) QA memory cases as session turns
    for case in qa_cases:
        turns = case.get("turns") or []
        session_id = f"orig_qa_{case.get('case_id')}"
        for idx, turn in enumerate(turns, start=1):
            if isinstance(turn, dict):
                text = turn.get("query") or turn.get("question") or turn.get("email_text") or ""
                exp_topics = turn.get("expected_topics") or []
                ref = turn.get("reference_answer") or ""
            else:
                text = str(turn)
                exp_topics = []
                ref = ""
            base = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "comparison_group_id": f"original_qa_memory::{case.get('case_id')}",
                "source_case_id": case.get("case_id"),
                "case_id": case.get("case_id"),
                "case_type": "original_qa_memory_like",
                "mode": "original_hans_query_session",
                "expected_programme": case.get("expected_programme", ""),
                "title": case.get("title", ""),
                "reference_answer": ref,
                "staff_comment": "",
            }
            rows.append(run_case(base, text, session_id, idx, exp_topics))

    # 4) Selected HANS Q&A reference questions, one-shot
    for item in qa_ref_selected:
        q = item.get("question") or ""
        if not q.strip():
            continue
        case_id = f"hans_ref_{item.get('source_row', item.get('number', 'x'))}"
        base = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "comparison_group_id": f"original_hans_reference::{case_id}",
            "source_case_id": case_id,
            "case_id": case_id,
            "case_type": "original_hans_reference_question",
            "mode": "original_hans_query",
            "expected_programme": "",
            "title": f"HANS reference question {item.get('number', '')}",
            "reference_answer": item.get("reference_answer", ""),
            "staff_comment": item.get("staff_comment", ""),
        }
        rows.append(run_case(base, q, f"orig_ref_{case_id}", 1, []))

    fieldnames = [
        "run_timestamp",
        "comparison_group_id",
        "source_case_id",
        "case_id",
        "case_type",
        "mode",
        "turn_index",
        "title",
        "expected_programme",
        "expected_followup_type",
        "input_text",
        "expected_topics",
        "keyword_missing_topics",
        "keyword_topic_coverage_percent",
        "citation_count",
        "source_count",
        "response_time_seconds",
        "http_ok",
        "http_status",
        "needs_manual_attention",
        "attention_reasons",
        "reference_answer",
        "staff_comment",
        "payload_used",
        "error",
        "output_text",
        "raw_response_json",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(rows)
    http_ok = sum(1 for r in rows if r["http_ok"] == "Yes")
    needs_attention = sum(1 for r in rows if r["needs_manual_attention"] == "Yes")
    print("\nDone.")
    print(f"Rows: {total}")
    print(f"HTTP OK: {http_ok}/{total}")
    print(f"Rows needing manual attention by heuristic: {needs_attention}/{total}")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
