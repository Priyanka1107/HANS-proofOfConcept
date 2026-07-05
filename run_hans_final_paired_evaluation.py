"""
Final paired evaluation runner for HANS PoC.

Purpose
-------
This script is intended for thesis evaluation after the final implementation freeze.

It uses a paired design:
1. Same Email Assistant questions are run with:
   - email_claude
   - email_mistral

2. Same multi-turn Email Assistant thread cases are run with:
   - email_claude
   - email_mistral

3. Same QA question pairs are run with:
   - baseline_qa
   - conversational_qa

This makes the comparison fairer than a broad mixed regression test, because the same inputs
are used for comparable modes.

How to run
----------
1. Start backend:
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

2. Put these JSON files in the evaluation/ folder:
   evaluation/final_email_provider_cases.json
   evaluation/final_email_thread_cases.json
   evaluation/final_qa_memory_cases.json

3. Run:
   python run_hans_final_paired_evaluation.py

Outputs
-------
evaluation/final_paired_evaluation_results.csv
evaluation/final_paired_evaluation_results.json
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
EVAL_DIR = os.path.join(os.getcwd(), "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

EMAIL_CASES_PATH = os.path.join(EVAL_DIR, "final_email_provider_cases.json")
THREAD_CASES_PATH = os.path.join(EVAL_DIR, "final_email_thread_cases.json")
QA_CASES_PATH = os.path.join(EVAL_DIR, "final_qa_memory_cases.json")

CSV_PATH = os.path.join(EVAL_DIR, "final_paired_evaluation_results.csv")
JSON_PATH = os.path.join(EVAL_DIR, "final_paired_evaluation_results.json")

EMAIL_MODES = ["email_claude", "email_mistral"]
QA_MODES = ["baseline_qa", "conversational_qa"]

HEADERS = [
    "run_timestamp",
    "comparison_group_id",
    "source_case_id",
    "case_id",
    "case_type",
    "turn_index",
    "mode",
    "backend_endpoint",
    "generation_provider",
    "generation_model",
    "language",
    "student_email",
    "subject",
    "thread_id",
    "session_id_sent",
    "session_id_returned",
    "input_text",
    "expected_programme",
    "matched_programme",
    "expected_followup_type",
    "actual_followup_type",
    "expected_topics",
    "detected_topics",
    "missing_topics",
    "extra_topics",
    "topic_coverage_percent",
    "expected_review_required",
    "review_required",
    "review_reason",
    "expected_answer_points",
    "reference_answer",
    "staff_comment",
    "auto_checks",
    "auto_pass",
    "quality_label",
    "quality_score",
    "is_grounded",
    "grounding_confidence",
    "citation_count",
    "source_count",
    "used_memory",
    "standalone_query",
    "response_time_seconds",
    "http_status",
    "error",
    "output_text",
]

TOPIC_EQUIVALENTS = {
    "application_route": {"application_route", "application_process"},
    "application_process": {"application_route", "application_process"},
    "application_before_graduation": {"application_before_graduation", "conditional_enrolment", "final_certificate_submission"},
    "final_certificate_submission": {"application_before_graduation", "conditional_enrolment", "final_certificate_submission", "application_deadline"},
    "conditional_enrolment": {"application_before_graduation", "conditional_enrolment", "final_certificate_submission"},
    "english_language_requirements": {"english_language_requirements", "language_requirements"},
    "german_language_requirements": {"german_language_requirements", "language_requirements"},
    "language_requirements": {"english_language_requirements", "german_language_requirements", "language_requirements", "language_of_instruction"},
    "language_of_instruction": {"language_of_instruction", "language_requirements"},
    "application_fee": {"application_fee", "fees", "tuition_fees", "semester_contribution"},
    "fees": {"application_fee", "fees", "tuition_fees", "semester_contribution"},
    "tuition_fees": {"tuition_fees", "fees", "application_fee", "semester_contribution"},
    "semester_contribution": {"semester_contribution", "fees", "tuition_fees"},
    "qualification_recognition": {"qualification_recognition", "admission_requirements"},
    "admission_requirements": {"admission_requirements", "qualification_recognition", "grade_conversion"},
    "motivation_letter": {"motivation_letter", "required_documents"},
    "application_deadline": {"application_deadline"},
    "start_semester": {"start_semester", "application_deadline"},
    "study_format": {"study_format"},
    "work_experience": {"work_experience", "admission_requirements"},
    "required_documents": {"required_documents", "document_uploads", "hard_copy_documents", "certified_translations", "motivation_letter"},
    "document_uploads": {"document_uploads", "required_documents", "hard_copy_documents"},
    "hard_copy_documents": {"hard_copy_documents", "document_uploads", "required_documents"},
    "certified_translations": {"certified_translations", "required_documents"},
    "credit_recognition": {"credit_recognition", "required_documents"},
    "grade_conversion": {"grade_conversion", "admission_requirements"},
    "accommodation": {"accommodation"},
}


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def topic_metrics(expected: List[str], detected: List[str]) -> Dict[str, Any]:
    expected_set = set(expected or [])
    detected_set = set(detected or [])
    if not expected_set:
        return {"missing": [], "extra": [], "coverage": "not_applicable"}

    covered = []
    for exp in sorted(expected_set):
        equivalents = TOPIC_EQUIVALENTS.get(exp, {exp})
        if equivalents & detected_set:
            covered.append(exp)

    missing = [t for t in sorted(expected_set) if t not in covered]
    extra = []
    for det in sorted(detected_set):
        matched = False
        for exp in expected_set:
            if det in TOPIC_EQUIVALENTS.get(exp, {exp}):
                matched = True
                break
        if not matched and det not in expected_set:
            extra.append(det)

    coverage = round((len(covered) / len(expected_set)) * 100, 2)
    return {"missing": missing, "extra": extra, "coverage": coverage}


def detect_email_topics(data: Dict[str, Any]) -> List[str]:
    topics = data.get("detected_topics", []) or []
    out = []
    for t in topics:
        if isinstance(t, dict) and t.get("topic_id"):
            out.append(str(t.get("topic_id")))
        elif isinstance(t, str):
            out.append(t)
    return out


def get_review_required(data: Dict[str, Any]) -> bool:
    quality = data.get("quality", {}) or {}
    return bool(quality.get("review_required", False) or data.get("flagged_for_human", False))


def get_citation_count(data: Dict[str, Any]) -> int:
    quality = data.get("quality", {}) or {}
    citations = data.get("citations", []) or quality.get("citations", []) or []
    try:
        return int(quality.get("citation_count", len(citations)))
    except Exception:
        return len(citations)


def text_contains(text: str, needle: Optional[str]) -> bool:
    if not needle:
        return True
    return needle.lower() in (text or "").lower()


def programme_match(expected: Optional[str], data: Dict[str, Any]) -> bool:
    if not expected:
        return True
    context = data.get("email_context", {}) or {}
    haystack = " ".join([
        str(data.get("matched_programme", "")),
        str(context.get("matched_programme", "")),
        str(context.get("target_program", "")),
        str(context.get("target_programme", "")),
        str(context.get("programme_url", "")),
    ]).lower()
    return expected.lower() in haystack


def call_email(email_text: str, test: Dict[str, Any], mode: str, thread_id: str) -> Dict[str, Any]:
    payload = {
        "email_text": email_text,
        "student_email": test.get("student_email"),
        "subject": test.get("subject"),
        "thread_id": thread_id,
        "session_id": None,
        "language": test.get("language"),
        "top_k": 3,
        "mode": mode,
    }
    start = time.time()
    resp = requests.post(f"{API_URL}/email", json=payload, timeout=240)
    elapsed = time.time() - start
    if resp.ok:
        data = resp.json()
        error = ""
    else:
        data = {}
        error = resp.text
    return {"resp": resp, "data": data, "elapsed": elapsed, "error": error, "payload": payload}


def call_query(query: str, test: Dict[str, Any], mode: str, session_id: Optional[str]) -> Dict[str, Any]:
    payload = {
        "query": query,
        "language": test.get("language"),
        "top_k": 5,
        "session_id": session_id,
        "mode": mode,
    }
    start = time.time()
    resp = requests.post(f"{API_URL}/query", json=payload, timeout=180)
    elapsed = time.time() - start
    if resp.ok:
        data = resp.json()
        error = ""
    else:
        data = {}
        error = resp.text
    return {"resp": resp, "data": data, "elapsed": elapsed, "error": error, "payload": payload, "sent_session_id": session_id}


def row_for_email(
    case: Dict[str, Any],
    mode: str,
    comparison_group_id: str,
    case_type: str,
    turn_index: int,
    email_text: str,
    expected_followup_type: str,
    expected_topics: List[str],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    data = result["data"]
    resp = result["resp"]
    output = data.get("staff_draft", "") if data else ""
    detected = detect_email_topics(data)
    metrics = topic_metrics(expected_topics, detected)
    quality = data.get("quality", {}) or {}
    validation = data.get("validation", {}) or {}
    sources = data.get("sources", []) or []
    context = data.get("email_context", {}) or {}

    expected_review = case.get("expected_review_required", "")
    if isinstance(expected_review, bool):
        expected_review = "Yes" if expected_review else "No"

    review_match = True
    if expected_review in {"Yes", "No"}:
        review_match = ("Yes" if get_review_required(data) else "No") == expected_review

    citation_count = get_citation_count(data)
    programme_ok = programme_match(case.get("expected_programme"), data)
    followup_ok = True
    if expected_followup_type:
        followup_ok = expected_followup_type == data.get("followup_type", "")

    # This is intentionally a lightweight automated pass.
    # Manual review remains required for source relevance and staff usefulness.
    auto_checks = {
        "http_ok": bool(resp.ok),
        "programme_match": programme_ok,
        "followup_match": followup_ok,
        "no_missing_topics": not metrics["missing"],
        "review_match": review_match,
        "has_citation_or_review": citation_count > 0 or get_review_required(data),
    }
    auto_pass = all(auto_checks.values())

    return {
        "run_timestamp": datetime.now().isoformat(),
        "comparison_group_id": comparison_group_id,
        "source_case_id": case.get("source_case_id", ""),
        "case_id": case.get("case_id", ""),
        "case_type": case_type,
        "turn_index": turn_index,
        "mode": mode,
        "backend_endpoint": "/email",
        "generation_provider": data.get("generation_provider", ""),
        "generation_model": data.get("generation_model", ""),
        "language": case.get("language", ""),
        "student_email": case.get("student_email", ""),
        "subject": case.get("subject", ""),
        "thread_id": data.get("thread_id") or result["payload"].get("thread_id", ""),
        "session_id_sent": "",
        "session_id_returned": data.get("session_id", ""),
        "input_text": email_text,
        "expected_programme": case.get("expected_programme", ""),
        "matched_programme": data.get("matched_programme", "") or context.get("matched_programme", "") or context.get("target_program", ""),
        "expected_followup_type": expected_followup_type,
        "actual_followup_type": data.get("followup_type", ""),
        "expected_topics": ", ".join(expected_topics or []),
        "detected_topics": ", ".join(detected),
        "missing_topics": ", ".join(metrics["missing"]),
        "extra_topics": ", ".join(metrics["extra"]),
        "topic_coverage_percent": metrics["coverage"],
        "expected_review_required": expected_review,
        "review_required": "Yes" if get_review_required(data) else "No",
        "review_reason": quality.get("review_reason", ""),
        "expected_answer_points": json.dumps(case.get("expected_answer_points", []), ensure_ascii=False),
        "reference_answer": case.get("reference_answer", ""),
        "staff_comment": case.get("staff_comment", ""),
        "auto_checks": json.dumps(auto_checks, ensure_ascii=False),
        "auto_pass": "Yes" if auto_pass else "No",
        "quality_label": quality.get("quality_label", ""),
        "quality_score": quality.get("quality_score", ""),
        "is_grounded": validation.get("is_grounded", ""),
        "grounding_confidence": validation.get("confidence", ""),
        "citation_count": citation_count,
        "source_count": len(sources),
        "used_memory": "",
        "standalone_query": "",
        "response_time_seconds": round(result["elapsed"], 2),
        "http_status": resp.status_code,
        "error": result.get("error", ""),
        "output_text": output,
    }


def row_for_query(
    case: Dict[str, Any],
    mode: str,
    comparison_group_id: str,
    turn: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    data = result["data"]
    resp = result["resp"]
    output = data.get("answer", "") if data else ""
    validation = data.get("validation", {}) or {}
    citations = data.get("citations", []) or []
    sources = data.get("sources", []) or []
    turn_index = int(turn.get("turn_index", 1))
    expected_programme = turn.get("expected_programme")

    # For conversational follow-ups, memory should be used and the standalone query should
    # include the programme context when one is expected.
    memory_ok = True
    standalone_ok = True
    if mode == "conversational_qa" and turn_index > 1:
        memory_ok = bool(data.get("used_memory"))
        standalone_ok = text_contains(data.get("standalone_query", ""), expected_programme)
    if mode == "baseline_qa":
        memory_ok = not bool(data.get("used_memory"))

    auto_checks = {
        "http_ok": bool(resp.ok),
        "has_citation": len(citations) > 0,
        "memory_expectation": memory_ok,
        "standalone_query_context": standalone_ok,
    }
    auto_pass = all(auto_checks.values())

    return {
        "run_timestamp": datetime.now().isoformat(),
        "comparison_group_id": comparison_group_id,
        "source_case_id": case.get("source", ""),
        "case_id": case.get("case_id", ""),
        "case_type": "qa_memory_pair",
        "turn_index": turn_index,
        "mode": mode,
        "backend_endpoint": "/query",
        "generation_provider": "",
        "generation_model": "",
        "language": case.get("language", ""),
        "student_email": "",
        "subject": "",
        "thread_id": "",
        "session_id_sent": result.get("sent_session_id") or "",
        "session_id_returned": data.get("session_id", ""),
        "input_text": turn.get("query", ""),
        "expected_programme": expected_programme or "",
        "matched_programme": "",
        "expected_followup_type": "",
        "actual_followup_type": "",
        "expected_topics": ", ".join(turn.get("expected_topics", [])),
        "detected_topics": "",
        "missing_topics": "",
        "extra_topics": "",
        "topic_coverage_percent": "manual_QA",
        "expected_review_required": "",
        "review_required": "not_applicable",
        "review_reason": "QA mode has no email review logic",
        "expected_answer_points": json.dumps(turn.get("reference_answer_points", []), ensure_ascii=False),
        "reference_answer": "",
        "staff_comment": "",
        "auto_checks": json.dumps(auto_checks, ensure_ascii=False),
        "auto_pass": "Yes" if auto_pass else "No",
        "quality_label": "",
        "quality_score": "",
        "is_grounded": validation.get("is_grounded", ""),
        "grounding_confidence": validation.get("confidence", ""),
        "citation_count": len(citations),
        "source_count": len(sources),
        "used_memory": data.get("used_memory", ""),
        "standalone_query": data.get("standalone_query", ""),
        "response_time_seconds": round(result["elapsed"], 2),
        "http_status": resp.status_code,
        "error": result.get("error", ""),
        "output_text": output,
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    auto_pass = sum(1 for r in rows if r.get("auto_pass") == "Yes")
    print(f"\nOverall automated checks: {auto_pass}/{total} rows passed")

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["case_type"], r["mode"])
        groups.setdefault(key, []).append(r)

    print("\nBy case type and mode:")
    for key, group_rows in sorted(groups.items()):
        passed = sum(1 for r in group_rows if r.get("auto_pass") == "Yes")
        print(f"- {key[0]} / {key[1]}: {passed}/{len(group_rows)}")

    print("\nRows needing manual attention:")
    for r in rows:
        if r.get("auto_pass") != "Yes":
            print(f"- {r['case_id']} | {r['case_type']} | {r['mode']} | missing_topics={r['missing_topics']} | review={r['review_required']} | citations={r['citation_count']}")


def main() -> None:
    print(f"Using backend: {API_URL}")

    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Health check: {health_resp.status_code}")
    except Exception as exc:
        print("Could not reach backend. Start uvicorn first.")
        print(exc)
        return

    email_cases = load_json(EMAIL_CASES_PATH)
    thread_cases = load_json(THREAD_CASES_PATH)
    qa_cases = load_json(QA_CASES_PATH)

    rows: List[Dict[str, Any]] = []

    print(f"\nRunning paired Email Assistant provider comparison: {len(email_cases)} cases × {len(EMAIL_MODES)} modes")
    for case in email_cases:
        email_text = case["turns"][0] if case.get("turns") else ""
        for mode in EMAIL_MODES:
            comparison_group_id = f"email_provider::{case['case_id']}"
            thread_id = f"{case.get('thread_id', case['case_id'])}_{mode}"
            result = call_email(email_text, case, mode, thread_id)
            row = row_for_email(
                case=case,
                mode=mode,
                comparison_group_id=comparison_group_id,
                case_type="email_provider_pair",
                turn_index=1,
                email_text=email_text,
                expected_followup_type=case.get("expected_followup_type", "new_enquiry"),
                expected_topics=case.get("expected_topics", []),
                result=result,
            )
            rows.append(row)
            print(f"  {case['case_id']} / {mode}: {row['auto_pass']} ({row['response_time_seconds']}s)")

    print(f"\nRunning paired Email Assistant thread comparison: {len(thread_cases)} threads × {len(EMAIL_MODES)} modes")
    for case in thread_cases:
        for mode in EMAIL_MODES:
            comparison_group_id = f"email_thread::{case['case_id']}"
            thread_id = f"{case.get('thread_id', case['case_id'])}_{mode}"
            for turn in case.get("turns", []):
                email_text = turn.get("email_text", "")
                result = call_email(email_text, case, mode, thread_id)
                case_for_row = dict(case)
                case_for_row["expected_programme"] = case.get("expected_programme")
                case_for_row["expected_review_required"] = case.get("expected_review_required", "")
                row = row_for_email(
                    case=case_for_row,
                    mode=mode,
                    comparison_group_id=comparison_group_id,
                    case_type="email_thread_pair",
                    turn_index=int(turn.get("turn_index", 1)),
                    email_text=email_text,
                    expected_followup_type=turn.get("expected_followup_type", ""),
                    expected_topics=turn.get("expected_topics", []),
                    result=result,
                )
                rows.append(row)
                print(f"  {case['case_id']} turn {turn.get('turn_index')} / {mode}: {row['auto_pass']} ({row['response_time_seconds']}s)")

    print(f"\nRunning paired QA memory comparison: {len(qa_cases)} scenarios × {len(QA_MODES)} modes × turns")
    for case in qa_cases:
        for mode in QA_MODES:
            comparison_group_id = f"qa_memory::{case['case_id']}"
            session_id: Optional[str] = None
            for turn in case.get("turns", []):
                result = call_query(turn.get("query", ""), case, mode, session_id)
                data = result["data"]
                if data.get("session_id"):
                    session_id = data.get("session_id")
                row = row_for_query(case, mode, comparison_group_id, turn, result)
                rows.append(row)
                print(f"  {case['case_id']} turn {turn.get('turn_index')} / {mode}: {row['auto_pass']} ({row['response_time_seconds']}s)")

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print_summary(rows)
    print(f"\nSaved CSV: {CSV_PATH}")
    print(f"Saved JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
