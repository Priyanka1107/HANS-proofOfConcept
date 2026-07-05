"""
Scripted smoke tests for HANS PoC four-mode backend.

How to use:
1) Start the backend first:
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

2) From the project root, run:
   python run_hans_script_smoke_tests.py

3) Results are written to:
   evaluation/script_smoke_test_results.csv
   evaluation/script_smoke_test_results.json

The script does not use Streamlit. It calls the backend API directly.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
EVAL_DIR = os.path.join(os.getcwd(), "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

CSV_PATH = os.path.join(EVAL_DIR, "script_smoke_test_results.csv")
JSON_PATH = os.path.join(EVAL_DIR, "script_smoke_test_results.json")

HEADERS = [
    "run_timestamp",
    "case_id",
    "case_type",
    "backend_endpoint",
    "backend_mode",
    "generation_provider",
    "generation_model",
    "thread_id",
    "session_id_sent",
    "session_id_returned",
    "student_email",
    "subject",
    "input_text",
    "expected_followup_type",
    "actual_followup_type",
    "expected_topics",
    "detected_topics",
    "missing_topics",
    "extra_topics",
    "topic_coverage_percent",
    "expected_checks",
    "check_results",
    "passed_basic_checks",
    "quality_label",
    "quality_score",
    "review_required",
    "review_reason",
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
    "application_before_graduation": {"application_before_graduation", "conditional_enrolment", "final_certificate_submission"},
    "english_language_requirements": {"english_language_requirements", "language_requirements"},
    "application_fee": {"application_fee", "fees", "tuition_fees", "semester_contribution"},
    "qualification_recognition": {"qualification_recognition", "admission_requirements"},
    "admission_requirements": {"admission_requirements", "qualification_recognition"},
    "motivation_letter": {"motivation_letter"},
    "application_deadline": {"application_deadline"},
    "language_of_instruction": {"language_of_instruction", "language_requirements"},
    "study_format": {"study_format"},
}


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
    for det in detected_set:
        matched = False
        for exp in expected_set:
            if det in TOPIC_EQUIVALENTS.get(exp, {exp}):
                matched = True
                break
        if not matched and det not in expected_set:
            extra.append(det)

    coverage = round((len(covered) / len(expected_set)) * 100, 2)
    return {"missing": missing, "extra": sorted(extra), "coverage": coverage}


def detect_email_topics(data: Dict[str, Any]) -> List[str]:
    return [str(t.get("topic_id", "")) for t in data.get("detected_topics", []) if t.get("topic_id")]


def contains_all(text: str, phrases: List[str]) -> bool:
    lower = (text or "").lower()
    return all(p.lower() in lower for p in phrases)


def contains_none(text: str, phrases: List[str]) -> bool:
    lower = (text or "").lower()
    return all(p.lower() not in lower for p in phrases)


def run_check(output: str, check: Dict[str, Any], data: Dict[str, Any]) -> bool:
    kind = check.get("type")
    if kind == "contains_all":
        return contains_all(output, check.get("phrases", []))
    if kind == "contains_none":
        return contains_none(output, check.get("phrases", []))
    if kind == "quality_min":
        try:
            return float((data.get("quality") or {}).get("quality_score", 0)) >= float(check.get("value", 0))
        except Exception:
            return False
    if kind == "used_memory_is_true":
        return bool(data.get("used_memory")) is True
    if kind == "standalone_contains_all":
        return contains_all(data.get("standalone_query", ""), check.get("phrases", []))
    return True


EMAIL_TESTS = [
    {
        "case_id": "email_01_ib_en_new",
        "mode": "email_claude",
        "thread_id": "script_thread_ib_final01",
        "student_email": "ananya@example.com",
        "subject": "Application for International Business Master",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nMy name is Ananya and I am completing my Bachelor's degree in Business Administration in India. I would like to apply for the Master's in International Business for the winter semester. Since I will receive my final transcript only in July, can I still apply before graduation? Do I need English language proof and are there application fees?""",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_before_graduation", "english_language_requirements", "application_fee"],
        "checks": [
            {"name": "greets Ananya", "type": "contains_all", "phrases": ["Dear Ananya"]},
            {"name": "keeps Master context", "type": "contains_all", "phrases": ["Master", "International Business"]},
            {"name": "fee verification link", "type": "contains_all", "phrases": ["Official uni-assist handling fees"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "email_02_ib_en_followup_motivation_letter",
        "mode": "email_claude",
        "thread_id": "script_thread_ib_final01",
        "student_email": "ananya@example.com",
        "subject": "Application for International Business Master",
        "language": None,
        "email_text": "Thank you for your reply. Is a motivation letter also required?",
        "expected_followup_type": "followup_new_topic",
        "expected_topics": ["motivation_letter"],
        "checks": [
            {"name": "greets Ananya", "type": "contains_all", "phrases": ["Dear Ananya"]},
            {"name": "has motivation letter answer", "type": "contains_all", "phrases": ["motivation letter"]},
            {"name": "does not switch to Bachelor", "type": "contains_none", "phrases": ["Bachelor's programme", "Bachelor programme"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
            {"name": "score at least 70", "type": "quality_min", "value": 70},
        ],
    },
    {
        "case_id": "email_03_mpmd_de_new",
        "mode": "email_claude",
        "thread_id": "script_thread_mpmd_final01",
        "student_email": "arjun@example.com",
        "subject": "Bewerbung Project Management and Data Science",
        "language": None,
        "email_text": """Sehr geehrtes Zulassungsteam,\n\nich habe meinen Bachelorabschluss in Electronics Engineering in Indien abgeschlossen und arbeite derzeit als Softwareentwickler. Ich möchte mich für den Masterstudiengang Project Management and Data Science bewerben. Können Sie mir bitte mitteilen, wann die Bewerbungsfrist ist, ob der Studiengang vollständig auf Englisch unterrichtet wird und ob es sich um ein Präsenzstudium handelt?\n\nMit freundlichen Grüßen\nArjun""",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_deadline", "language_of_instruction", "study_format"],
        "checks": [
            {"name": "German reply", "type": "contains_all", "phrases": ["Mit freundlichen Grüßen"]},
            {"name": "deadline/language/format", "type": "contains_all", "phrases": ["Bewerbungsfrist", "englisch", "Präsenz"]},
            {"name": "no APS profile advice", "type": "contains_none", "phrases": ["APS", "Berufserfahrung"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "email_04_ib_de_new",
        "mode": "email_claude",
        "thread_id": "script_thread_ib_de_final01",
        "student_email": "ananya@example.com",
        "subject": "Bewerbung International Business Master",
        "language": None,
        "email_text": """Sehr geehrtes Zulassungsteam,\n\nmein Name ist Ananya und ich schließe derzeit meinen Bachelor in Betriebswirtschaftslehre in Indien ab. Ich möchte mich für den Masterstudiengang International Business zum Wintersemester bewerben. Da ich mein endgültiges Zeugnis erst im Juli erhalte, kann ich mich trotzdem vor dem Abschluss bewerben? Brauche ich einen Nachweis über Englischkenntnisse und fallen Bewerbungsgebühren an?\n\nMit freundlichen Grüßen\nAnanya""",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_before_graduation", "english_language_requirements", "application_fee"],
        "checks": [
            {"name": "greets Ananya", "type": "contains_all", "phrases": ["Ananya"]},
            {"name": "German reply", "type": "contains_all", "phrases": ["Mit freundlichen Grüßen"]},
            {"name": "answers fee", "type": "contains_all", "phrases": ["Bewerbungsgeb"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "email_05_csb_de_new",
        "mode": "email_claude",
        "thread_id": "script_thread_csb_de_final01",
        "student_email": "student@example.com",
        "subject": "Bewerbung Cyber Security and Business",
        "language": None,
        "email_text": """Sehr geehrtes Zulassungsteam,\n\nich bin EU-Bürger und lebe in Portugal. Ich mache derzeit mein International Baccalaureate Diploma und möchte mich für den Bachelorstudiengang Cyber Security and Business bewerben. Muss ich mich über uni-assist bewerben, erfüllt mein IB-Diplom die Zulassungsvoraussetzungen und brauche ich ein Motivationsschreiben?\n\nMit freundlichen Grüßen""",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "qualification_recognition", "admission_requirements", "motivation_letter"],
        "checks": [
            {"name": "German reply", "type": "contains_all", "phrases": ["Mit freundlichen Grüßen"]},
            {"name": "EU route", "type": "contains_all", "phrases": ["EU", "Hochschulstart"]},
            {"name": "not uni-assist main route", "type": "contains_none", "phrases": ["müssen Sie sich über uni-assist bewerben", "apply through uni-assist"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "email_06_dual_citizenship_en",
        "mode": "email_claude",
        "thread_id": "script_thread_dual_citizenship_final01",
        "student_email": "student@example.com",
        "subject": "Application route with dual citizenship",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI have dual French and Moroccan citizenship and I currently live in Morocco. I am finishing my school-leaving certificate outside Germany and would like to apply for the Cyber Security and Business Bachelor's programme. Should I apply as an EU applicant through the HTW Berlin application portal, or do I need to apply through uni-assist because I live in Morocco and my school certificate is foreign?\n\nKind regards""",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "qualification_recognition"],
        "checks": [
            {"name": "recognises EU/French citizenship", "type": "contains_all", "phrases": ["French", "EU"]},
            {"name": "separates Morocco/residence", "type": "contains_all", "phrases": ["Morocco"]},
            {"name": "does not make uni-assist main route", "type": "contains_none", "phrases": ["you need to apply through uni-assist", "must apply through uni-assist"]},
            {"name": "score at least 75", "type": "quality_min", "value": 75},
        ],
    },
]

QA_TESTS = [
    {
        "case_id": "qa_01_mpmd_de_first",
        "mode": "conversational_qa",
        "session_key": "qa_mpmd",
        "language": None,
        "query": "Ich interessiere mich für den Masterstudiengang Project Management and Data Science an der HTW Berlin. Wo finde ich die Zulassungsvoraussetzungen?",
        "expected_topics": ["admission_requirements"],
        "checks": [
            {"name": "German answer", "type": "contains_all", "phrases": ["Zulassung"]},
            {"name": "mentions MPMD", "type": "contains_all", "phrases": ["Project Management and Data Science"]},
        ],
    },
    {
        "case_id": "qa_02_mpmd_de_followup",
        "mode": "conversational_qa",
        "session_key": "qa_mpmd",
        "language": None,
        "query": "Wird er auf Englisch oder Deutsch unterrichtet?",
        "expected_topics": ["language_of_instruction"],
        "checks": [
            {"name": "uses memory", "type": "used_memory_is_true"},
            {"name": "standalone mentions programme", "type": "standalone_contains_all", "phrases": ["Project Management and Data Science"]},
            {"name": "answers English", "type": "contains_all", "phrases": ["Englisch"]},
        ],
    },
]


def call_email(test: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "email_text": test["email_text"],
        "student_email": test.get("student_email"),
        "subject": test.get("subject"),
        "thread_id": test.get("thread_id"),
        "session_id": None,
        "language": test.get("language"),
        "top_k": 3,
        "mode": test["mode"],
    }
    start = time.time()
    resp = requests.post(f"{API_URL}/email", json=payload, timeout=180)
    elapsed = time.time() - start
    data: Dict[str, Any]
    error = ""
    if resp.ok:
        data = resp.json()
    else:
        data = {}
        error = resp.text
    return {"resp": resp, "data": data, "elapsed": elapsed, "error": error, "payload": payload}


def call_query(test: Dict[str, Any], sessions: Dict[str, Optional[str]]) -> Dict[str, Any]:
    session_key = test.get("session_key", "default")
    sent_session_id = sessions.get(session_key)
    payload = {
        "query": test["query"],
        "language": test.get("language"),
        "top_k": 5,
        "session_id": sent_session_id,
        "mode": test["mode"],
    }
    start = time.time()
    resp = requests.post(f"{API_URL}/query", json=payload, timeout=120)
    elapsed = time.time() - start
    error = ""
    if resp.ok:
        data = resp.json()
        if data.get("session_id"):
            sessions[session_key] = data.get("session_id")
    else:
        data = {}
        error = resp.text
    return {"resp": resp, "data": data, "elapsed": elapsed, "error": error, "payload": payload, "sent_session_id": sent_session_id}


def row_for_email(test: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    data = result["data"]
    resp = result["resp"]
    output = data.get("staff_draft", "") if data else ""
    detected = detect_email_topics(data)
    metrics = topic_metrics(test.get("expected_topics", []), detected)
    quality = data.get("quality", {}) or {}
    validation = data.get("validation", {}) or {}
    citations = data.get("citations", []) or quality.get("citations", []) or []
    sources = data.get("sources", []) or []

    check_results = []
    for check in test.get("checks", []):
        passed = run_check(output, check, data)
        check_results.append({"name": check.get("name", "check"), "passed": passed})

    passed_basic = (
        resp.ok
        and test.get("expected_followup_type", "") == (data.get("followup_type", "") if data else "")
        and not metrics["missing"]
        and all(item["passed"] for item in check_results)
    )

    return {
        "run_timestamp": datetime.now().isoformat(),
        "case_id": test["case_id"],
        "case_type": "email",
        "backend_endpoint": "/email",
        "backend_mode": test["mode"],
        "generation_provider": data.get("generation_provider", ""),
        "generation_model": data.get("generation_model", ""),
        "thread_id": data.get("thread_id") or test.get("thread_id", ""),
        "session_id_sent": "",
        "session_id_returned": data.get("session_id", ""),
        "student_email": test.get("student_email", ""),
        "subject": test.get("subject", ""),
        "input_text": test.get("email_text", ""),
        "expected_followup_type": test.get("expected_followup_type", ""),
        "actual_followup_type": data.get("followup_type", ""),
        "expected_topics": ", ".join(test.get("expected_topics", [])),
        "detected_topics": ", ".join(detected),
        "missing_topics": ", ".join(metrics["missing"]),
        "extra_topics": ", ".join(metrics["extra"]),
        "topic_coverage_percent": metrics["coverage"],
        "expected_checks": json.dumps(test.get("checks", []), ensure_ascii=False),
        "check_results": json.dumps(check_results, ensure_ascii=False),
        "passed_basic_checks": "Yes" if passed_basic else "No",
        "quality_label": quality.get("quality_label", ""),
        "quality_score": quality.get("quality_score", ""),
        "review_required": "Yes" if quality.get("review_required") else "No",
        "review_reason": quality.get("review_reason", ""),
        "is_grounded": validation.get("is_grounded", ""),
        "grounding_confidence": validation.get("confidence", ""),
        "citation_count": quality.get("citation_count", len(citations)),
        "source_count": len(sources),
        "used_memory": "",
        "standalone_query": "",
        "response_time_seconds": round(result["elapsed"], 2),
        "http_status": resp.status_code,
        "error": result.get("error", ""),
        "output_text": output,
    }


def row_for_query(test: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    data = result["data"]
    resp = result["resp"]
    output = data.get("answer", "") if data else ""
    validation = data.get("validation", {}) or {}
    citations = data.get("citations", []) or []
    sources = data.get("sources", []) or []

    check_results = []
    for check in test.get("checks", []):
        passed = run_check(output, check, data)
        check_results.append({"name": check.get("name", "check"), "passed": passed})

    passed_basic = resp.ok and all(item["passed"] for item in check_results)

    return {
        "run_timestamp": datetime.now().isoformat(),
        "case_id": test["case_id"],
        "case_type": "qa",
        "backend_endpoint": "/query",
        "backend_mode": test["mode"],
        "generation_provider": "",
        "generation_model": "",
        "thread_id": "",
        "session_id_sent": result.get("sent_session_id") or "",
        "session_id_returned": data.get("session_id", ""),
        "student_email": "",
        "subject": "",
        "input_text": test.get("query", ""),
        "expected_followup_type": "",
        "actual_followup_type": "",
        "expected_topics": ", ".join(test.get("expected_topics", [])),
        "detected_topics": "",
        "missing_topics": "",
        "extra_topics": "",
        "topic_coverage_percent": "manual_QA",
        "expected_checks": json.dumps(test.get("checks", []), ensure_ascii=False),
        "check_results": json.dumps(check_results, ensure_ascii=False),
        "passed_basic_checks": "Yes" if passed_basic else "No",
        "quality_label": "",
        "quality_score": "",
        "review_required": "not_applicable",
        "review_reason": "QA mode has no email review logic",
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


def main() -> None:
    print(f"Using backend: {API_URL}")
    rows: List[Dict[str, Any]] = []

    # Quick health check.
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Health check: {health_resp.status_code}")
    except Exception as exc:
        print("Could not reach backend. Start uvicorn first.")
        print(exc)
        return

    print("\nRunning Email Assistant smoke tests...")
    for test in EMAIL_TESTS:
        print(f"- {test['case_id']}")
        try:
            result = call_email(test)
            row = row_for_email(test, result)
        except Exception as exc:
            row = {h: "" for h in HEADERS}
            row.update({
                "run_timestamp": datetime.now().isoformat(),
                "case_id": test["case_id"],
                "case_type": "email",
                "backend_endpoint": "/email",
                "backend_mode": test["mode"],
                "thread_id": test.get("thread_id", ""),
                "input_text": test.get("email_text", ""),
                "http_status": 0,
                "error": str(exc),
                "passed_basic_checks": "No",
            })
        rows.append(row)
        print(f"  passed_basic_checks={row['passed_basic_checks']} | score={row.get('quality_score')} | topics={row.get('detected_topics')}")

    print("\nRunning Conversational QA smoke tests...")
    sessions: Dict[str, Optional[str]] = {}
    for test in QA_TESTS:
        print(f"- {test['case_id']}")
        try:
            result = call_query(test, sessions)
            row = row_for_query(test, result)
        except Exception as exc:
            row = {h: "" for h in HEADERS}
            row.update({
                "run_timestamp": datetime.now().isoformat(),
                "case_id": test["case_id"],
                "case_type": "qa",
                "backend_endpoint": "/query",
                "backend_mode": test["mode"],
                "input_text": test.get("query", ""),
                "http_status": 0,
                "error": str(exc),
                "passed_basic_checks": "No",
            })
        rows.append(row)
        print(f"  passed_basic_checks={row['passed_basic_checks']} | used_memory={row.get('used_memory')} | grounded={row.get('is_grounded')}")

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed_basic_checks") == "Yes")
    print("\nDone.")
    print(f"Passed basic checks: {passed}/{total}")
    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")

    if passed != total:
        print("\nSome checks failed. Open the CSV and inspect check_results + output_text.")


if __name__ == "__main__":
    main()
