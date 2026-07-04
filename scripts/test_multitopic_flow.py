# scripts/test_multitopic_flow.py
"""
Run V4.6 Email Assistant tests.

Important difference from older V4 tests:
- The backend now generates ONE final staff-ready draft per email.
- The CSV therefore has ONE row per email, not one row per sub-topic.

Usage:
    python scripts/test_multitopic_flow.py --version v4_6_email_assistant --comment "evidence per topic, one final draft"
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Set
import re
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
CASES_PATH = os.path.join(EVAL_DIR, "multitopic_test_cases.json")
RESULTS_CSV = os.path.join(EVAL_DIR, "email_assistant_v5_results.csv")
RESULTS_JSONL = os.path.join(EVAL_DIR, "email_assistant_v5_results.jsonl")


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "run_timestamp",
        "version_label",
        "comment",
        "case_id",
        "title",
        "full_email",
        "student_email",
        "subject",
        "thread_id",
        "followup_type",
        "expected_topic_count",
        "expected_topics",
        "detected_topic_count",
        "detected_topics",
        "missing_topics",
        "extra_topics",
        "topic_coverage_percent",
        "response_time_seconds",
        "quality_score",
        "quality_label",
        "review_required",
        "review_reason",
        "bad_draft_phrase",
        "is_grounded",
        "grounding_confidence",
        "citation_count",
        "citations",
        "source_count",
        "staff_draft",
        "http_status",
        "error",
    ]

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Topic equivalence mapping for fair evaluation.
# Example: if the expected topic is "fees", the system should get credit when
# it detects tuition_fees, application_fee, or semester_contribution.
TOPIC_EQUIVALENTS: Dict[str, Set[str]] = {
    "language_requirements": {
        "language_requirements",
        "english_language_requirements",
        "german_language_requirements",
    },
    "required_documents": {
        "required_documents",
        "document_uploads",
        "certified_translations",
        "official_transcripts",
        "hard_copy_documents",
        "final_certificate_submission",
    },
    "fees": {
        "fees",
        "tuition_fees",
        "application_fee",
        "semester_contribution",
    },
    "final_certificate_submission": {
        "final_certificate_submission",
        "application_before_graduation",
        "conditional_enrolment",
        "document_uploads",
        "required_documents",
    },
    "application_route": {
        "application_route",
        "application_process",
    },
    "admission_requirements": {
        "admission_requirements",
        "qualification_recognition",
    },
}


def _is_expected_topic_covered(expected_topic: str, detected_set: Set[str]) -> bool:
    equivalent_topics = TOPIC_EQUIVALENTS.get(expected_topic, {expected_topic})
    return bool(equivalent_topics & detected_set)


def _is_extra_topic(detected_topic: str, expected_set: Set[str]) -> bool:
    # A detected topic is not extra if it satisfies any expected topic through the
    # equivalence mapping.
    for expected_topic in expected_set:
        equivalent_topics = TOPIC_EQUIVALENTS.get(expected_topic, {expected_topic})
        if detected_topic in equivalent_topics:
            return False
    return detected_topic not in expected_set


def topic_metrics(expected: List[str], detected: List[str]) -> Dict[str, Any]:
    expected_set: Set[str] = set(expected or [])
    detected_set: Set[str] = set(detected or [])

    if not expected_set:
        return {
            "missing_topics": "",
            "extra_topics": ", ".join(sorted(detected_set)),
            "topic_coverage_percent": "",
        }

    covered = [e for e in sorted(expected_set) if _is_expected_topic_covered(e, detected_set)]
    missing = [e for e in sorted(expected_set) if e not in covered]
    extra = sorted([d for d in detected_set if _is_extra_topic(d, expected_set)])

    coverage = round((len(covered) / len(expected_set)) * 100, 2)

    return {
        "missing_topics": ", ".join(missing),
        "extra_topics": ", ".join(extra),
        "topic_coverage_percent": coverage,
    }

DISCLAIMER_MARKERS = [
    "This draft was generated with AI support",
    "This mail was generated with AI support",
    "This email was generated with AI support",
]


def strip_disclaimer_for_metrics(text: str) -> str:
    """
    Remove the configurable AI disclaimer before evaluation phrase checks.

    The full draft is still stored in the CSV, but the quality/review metrics
    should only evaluate the generated answer body.
    """
    cleaned = text or ""
    for marker in DISCLAIMER_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
            break
    return cleaned

def run_case(api_url: str, case: Dict[str, Any], version: str, comment: str, timeout: int = 120) -> Dict[str, Any]:
    run_timestamp = datetime.now().isoformat()
    email_text = case["turns"][0]
    expected_topics = case.get("expected_topics", [])
    expected_topic_count = case.get("expected_topic_count", len(expected_topics))

    case_id = case.get("case_id", "case")
    title = case.get("title", "Student enquiry")

    # These fields are useful for comparing script results with UI results.
    # If the test case JSON does not provide them, create stable demo values.
    student_email = case.get("student_email") or f"{case_id}@example.com"
    subject = case.get("subject") or title
    thread_id = case.get("thread_id") or f"thread_{case_id}"

    payload = {
        "email_text": email_text,
        "student_email": student_email,
        "subject": subject,
        "thread_id": thread_id,
        "top_k": 3,
    }

    started = time.time()
    http_status = None

    try:
        resp = requests.post(f"{api_url}/email", json=payload, timeout=timeout)
        elapsed = round(time.time() - started, 2)
        http_status = resp.status_code

        if not resp.ok:
            return {
                "run_timestamp": run_timestamp,
                "version_label": version,
                "comment": comment,
                "case_id": case_id,
                "title": title,
                "full_email": email_text,
                "student_email": student_email,
                "subject": subject,
                "thread_id": thread_id,
                "followup_type": "",
                "expected_topic_count": expected_topic_count,
                "expected_topics": ", ".join(expected_topics),
                "detected_topic_count": 0,
                "detected_topics": "",
                "missing_topics": ", ".join(expected_topics),
                "extra_topics": "",
                "topic_coverage_percent": 0,
                "response_time_seconds": elapsed,
                "quality_score": "",
                "quality_label": "error",
                "review_required": "Yes",
                "review_reason": "Backend error",
                "bad_draft_phrase": "",
                "is_grounded": "",
                "grounding_confidence": "",
                "citation_count": 0,
                "citations": "",
                "source_count": 0,
                "staff_draft": "",
                "http_status": http_status,
                "error": resp.text[:500],
            }

        data = resp.json()
        detected_topics = [t.get("topic_id", "") for t in data.get("detected_topics", []) if t.get("topic_id")]
        metrics = topic_metrics(expected_topics, detected_topics)

        quality = data.get("quality", {})
        validation = data.get("validation", {})
        citations = data.get("citations", [])
        sources = data.get("sources", [])
        staff_draft = data.get("staff_draft", "")
        
        draft_citations = []
        for match in re.finditer(r"\[(?:Doc\s*)?(\d+)\]", staff_draft or "", flags=re.IGNORECASE):
            citation = f"[Doc {match.group(1)}]"
            if citation not in draft_citations:
                draft_citations.append(citation)

        # Additional test-level review: if expected topics are missing, mark review.
        review_required = bool(quality.get("review_required", False))
        review_reason = quality.get("review_reason", "") or ""

        if metrics["missing_topics"]:
            review_required = True
            if review_reason:
                review_reason += "; "
            review_reason += "Missing expected topic(s): " + metrics["missing_topics"]

        if metrics["extra_topics"]:
            # Extra topics are not always wrong, but they are useful to inspect.
            if review_reason:
                review_reason += "; "
            review_reason += "Extra detected topic(s): " + metrics["extra_topics"]

        # Draft-level safety checks for evaluation.
        draft_body_for_metrics = strip_disclaimer_for_metrics(staff_draft)
        lower_draft = (draft_body_for_metrics or "").lower()
        uncertainty_phrases = [
            "evidence documents do not contain",
            "available information does not specify",
            "not available in our current documentation",
            "should be verified by our staff",
            "please let us know which degree programme",
            "could you please specify",
            "which programme",
            "does not contain specific information",
            "do not contain specific information",
            "not specified in the available",
            "not specified in our",
            "cannot confirm",
            "could not confirm",
            "not confirm",
            "should be checked",
            "should be verified",
            "verify before sending",
            "check before sending",
            "contact student services directly",
            "contact the admissions office",
        ]
        uncertainty_found = any(p in lower_draft for p in uncertainty_phrases)
        
        if uncertainty_found:
            review_required = True
            if review_reason:
                review_reason += "; "
            review_reason += "Draft contains uncertainty or needs staff check"

        # Make the quality label consistent with the review decision.
        coverage = metrics["topic_coverage_percent"]
        try:
            coverage_value = float(coverage)
        except Exception:
            coverage_value = 0.0

        base_score = quality.get("quality_score", 100)
        try:
            base_score = float(base_score)
        except Exception:
            base_score = 100.0

        adjusted_score = base_score
        if metrics["missing_topics"]:
            adjusted_score -= 20
        if uncertainty_found:
            adjusted_score -= 15
        if validation and validation.get("is_grounded") is False:
            adjusted_score -= 20
        if not draft_citations:
            adjusted_score -= 20
        adjusted_score = max(0, min(100, round(adjusted_score, 2)))

        if review_required:
            if adjusted_score >= 75:
                adjusted_label = "partial"
            else:
                adjusted_label = "review"
        else:
            adjusted_label = "good" if adjusted_score >= 80 else "partial"

        return {
            "run_timestamp": run_timestamp,
            "version_label": version,
            "comment": comment,
            "case_id": case.get("case_id", ""),
            "title": case.get("title", ""),
            "full_email": email_text,
            "student_email": case.get("student_email", ""),
            "subject": case.get("subject", ""),
            "thread_id": data.get("thread_id") or thread_id,
            "followup_type": data.get("followup_type", "new_enquiry"),
            "expected_topic_count": expected_topic_count,
            "expected_topics": ", ".join(expected_topics),
            "detected_topic_count": len(detected_topics),
            "detected_topics": ", ".join(detected_topics),
            "missing_topics": metrics["missing_topics"],
            "extra_topics": metrics["extra_topics"],
            "topic_coverage_percent": metrics["topic_coverage_percent"],
            "response_time_seconds": elapsed,
            "quality_score": adjusted_score,
            "quality_label": adjusted_label,
            "review_required": "Yes" if review_required else "No",
            "review_reason": review_reason,
            "bad_draft_phrase": quality.get("bad_draft_phrase", ""),
            "is_grounded": validation.get("is_grounded", ""),
            "grounding_confidence": validation.get("confidence", ""),
            "citation_count": len(draft_citations),
            "citations": ", ".join(draft_citations),
            "source_count": len(sources),
            "staff_draft": staff_draft,
            "http_status": http_status,
            "error": "",
        }

    except Exception as e:
        elapsed = round(time.time() - started, 2)
        return {
            "run_timestamp": run_timestamp,
            "version_label": version,
            "comment": comment,
            "case_id": case.get("case_id", ""),
            "title": case.get("title", ""),
            "full_email": email_text,
            "student_email": case.get("student_email", ""),
            "subject": case.get("subject", ""),
            "thread_id": case.get("thread_id", ""),
            "followup_type": "",
            "expected_topic_count": expected_topic_count,
            "expected_topics": ", ".join(expected_topics),
            "detected_topic_count": 0,
            "detected_topics": "",
            "missing_topics": ", ".join(expected_topics),
            "extra_topics": "",
            "topic_coverage_percent": 0,
            "response_time_seconds": elapsed,
            "quality_score": "",
            "quality_label": "error",
            "review_required": "Yes",
            "review_reason": "API connection or runtime error",
            "bad_draft_phrase": "",
            "is_grounded": "",
            "grounding_confidence": "",
            "citation_count": 0,
            "citations": "",
            "source_count": 0,
            "staff_draft": "",
            "http_status": http_status,
            "error": str(e)[:500],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-url", default="http://127.0.0.1:8001")
    p.add_argument("--version", required=True)
    p.add_argument("--comment", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(EVAL_DIR, exist_ok=True)

    cases = load_cases(CASES_PATH)
    rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("HANS V5 Email Assistant Test Runner")
    print(f"Cases: {len(cases)} | Version: {args.version} | API: {args.api_url}")
    print("=" * 80)

    for case in cases:
        row = run_case(args.api_url, case, args.version, args.comment)
        rows.append(row)

        print(f"\n[{row['case_id']}] {row['title']}")
        print(f"  Expected: {row['expected_topics']}")
        print(f"  Detected: {row['detected_topics']}")
        print(f"  Coverage: {row['topic_coverage_percent']}%")
        print(f"  Quality: {row['quality_label']} ({row['quality_score']})")
        print(f"  Quality warning: {row['review_required']} - {row['review_reason']}")
        print(f"  Time: {row['response_time_seconds']} sec")

    write_csv(RESULTS_CSV, rows)
    write_jsonl(RESULTS_JSONL, rows)

    good = sum(1 for r in rows if r["quality_label"] == "good" and r["review_required"] == "No")
    review = sum(1 for r in rows if r["review_required"] == "Yes")
    avg_time = round(sum(float(r["response_time_seconds"]) for r in rows) / max(1, len(rows)), 2)
    avg_coverage_vals = [
        float(r["topic_coverage_percent"])
        for r in rows
        if r["topic_coverage_percent"] not in ("", None)
    ]
    avg_coverage = round(sum(avg_coverage_vals) / max(1, len(avg_coverage_vals)), 2)

    print("\n" + "=" * 80)
    print("Saved results:")
    print(f"  CSV:   {RESULTS_CSV}")
    print(f"  JSONL: {RESULTS_JSONL}")
    print("\nSummary:")
    print(f"  Good without review: {good}/{len(rows)}")
    print(f"  Quality warnings:    {review}/{len(rows)}")
    print(f"  Avg. time:           {avg_time} sec")
    print(f"  Avg. topic coverage: {avg_coverage}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
