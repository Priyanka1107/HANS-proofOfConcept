from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Set

import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

CASES_PATH = os.path.join(EVAL_DIR, "final_4mode_test_cases.json")
RESULTS_CSV = os.path.join(EVAL_DIR, "final_4mode_raw_results.csv")
RESULTS_JSONL = os.path.join(EVAL_DIR, "final_4mode_raw_results.jsonl")


MODES = [
    "baseline_qa",
    "conversational_qa",
    "email_claude",
    "email_mistral",
]


TOPIC_EQUIVALENTS: Dict[str, Set[str]] = {
    "language_requirements": {
        "language_requirements",
        "english_language_requirements",
        "german_language_requirements",
        "language_of_instruction",
    },
    "english_language_requirements": {
        "english_language_requirements",
        "language_requirements",
        "language_of_instruction",
    },
    "german_language_requirements": {
        "german_language_requirements",
        "language_requirements",
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


def load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalise_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return ""
    return str(value)


def topic_metrics(expected: List[str], detected: List[str]) -> Dict[str, Any]:
    expected_set = set(expected or [])
    detected_set = set(detected or [])

    if not expected_set:
        return {
            "missing_topics": "",
            "extra_topics": ", ".join(sorted(detected_set)),
            "topic_coverage_percent": "",
        }

    covered = []
    for expected_topic in sorted(expected_set):
        equivalents = TOPIC_EQUIVALENTS.get(expected_topic, {expected_topic})
        if equivalents & detected_set:
            covered.append(expected_topic)

    missing = [t for t in sorted(expected_set) if t not in covered]

    extra = []
    for detected_topic in detected_set:
        matched = False
        for expected_topic in expected_set:
            equivalents = TOPIC_EQUIVALENTS.get(expected_topic, {expected_topic})
            if detected_topic in equivalents:
                matched = True
                break
        if not matched:
            extra.append(detected_topic)

    return {
        "missing_topics": ", ".join(missing),
        "extra_topics": ", ".join(sorted(extra)),
        "topic_coverage_percent": round((len(covered) / len(expected_set)) * 100, 2),
    }


def extract_email_response_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    detected_topics = [
        t.get("topic_id", "")
        for t in data.get("detected_topics", [])
        if t.get("topic_id")
    ]

    quality = data.get("quality", {}) or {}
    validation = data.get("validation", {}) or {}
    email_context = data.get("email_context", {}) or {}
    sources = data.get("sources", []) or []

    source_urls = []
    for source in sources:
        url = source.get("url", "")
        if url:
            source_urls.append(url)

    return {
        "actual_programme": (
            email_context.get("matched_programme")
            or email_context.get("target_programme")
            or email_context.get("target_program")
            or ""
        ),
        "detected_topics": detected_topics,
        "actual_review_required": bool(quality.get("review_required", False)),
        "review_reason": quality.get("review_reason", "") or "",
        "quality_label": quality.get("quality_label", "") or "",
        "quality_score": quality.get("quality_score", ""),
        "citation_count": len(data.get("citations", []) or []),
        "source_count": len(sources),
        "source_urls": source_urls,
        "generated_output": data.get("staff_draft", "") or "",
        "is_grounded": validation.get("is_grounded", ""),
        "grounding_confidence": validation.get("confidence", ""),
    }


def extract_query_response_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    validation = data.get("validation", {}) or {}
    sources = data.get("sources", []) or []

    source_urls = []
    for source in sources:
        url = source.get("url", "")
        if url:
            source_urls.append(url)

    return {
        "actual_programme": "",
        "detected_topics": [],
        "actual_review_required": False,
        "review_reason": "baseline mode has no email review logic",
        "quality_label": "",
        "quality_score": "",
        "citation_count": len(data.get("citations", []) or []),
        "source_count": len(sources),
        "source_urls": source_urls,
        "generated_output": data.get("answer", "") or "",
        "is_grounded": validation.get("is_grounded", ""),
        "grounding_confidence": validation.get("confidence", ""),
    }


def run_one(api_url: str, case: Dict[str, Any], mode: str, evaluation_run_id: str) -> Dict[str, Any]:
    run_timestamp = datetime.now().isoformat()
    input_text = "\n\n".join(case.get("turns", []))
    started = time.time()

    expected_topics = case.get("expected_topics", []) or []
    expected_programme = case.get("expected_programme")
    expected_review_required = bool(case.get("expected_review_required", False))

    try:
        if mode in {"baseline_qa", "conversational_qa"}:
            payload = {
                "query": input_text,
                "language": case.get("language"),
                "top_k": 5,
                "mode": mode,
                "session_id": f"{evaluation_run_id}_{case.get('case_id')}_{mode}",
            }
            resp = requests.post(f"{api_url}/query", json=payload, timeout=180)
        else:
            payload = {
                "email_text": input_text,
                "student_email": case.get("student_email"),
                "subject": case.get("subject"),
                "thread_id": f"{evaluation_run_id}_{case.get('thread_id')}_{mode}",
                "language": case.get("language"),
                "top_k": 3,
                "mode": mode,
                "test_id": case.get("case_id"),
                "evaluation_run_id": evaluation_run_id,
            }
            resp = requests.post(f"{api_url}/email", json=payload, timeout=180)

        elapsed = round(time.time() - started, 2)

        if not resp.ok:
            return {
                "evaluation_run_id": evaluation_run_id,
                "run_timestamp": run_timestamp,
                "case_id": case.get("case_id", ""),
                "dataset_group": case.get("dataset_group", ""),
                "mode": mode,
                "input_text": input_text,
                "expected_programme": expected_programme or "",
                "actual_programme": "",
                "programme_match_status": "error",
                "expected_topics": ", ".join(expected_topics),
                "detected_topics": "",
                "missing_topics": ", ".join(expected_topics),
                "extra_topics": "",
                "topic_coverage_percent": 0,
                "expected_review_required": normalise_bool(expected_review_required),
                "actual_review_required": "",
                "review_match_status": "error",
                "review_reason": "",
                "answerable": case.get("answerable", ""),
                "citation_count": "",
                "source_count": "",
                "source_urls": "",
                "response_time_seconds": elapsed,
                "quality_label": "error",
                "quality_score": "",
                "is_grounded": "",
                "grounding_confidence": "",
                "generated_output": "",
                "http_status": resp.status_code,
                "error": resp.text[:1000],
            }

        data = resp.json()
        fields = (
            extract_query_response_fields(data)
            if mode in {"baseline_qa", "conversational_qa"}
            else extract_email_response_fields(data)
        )

        detected_topics = fields["detected_topics"]
        metrics = topic_metrics(expected_topics, detected_topics)

        actual_programme = fields["actual_programme"]
        if mode in {"baseline_qa", "conversational_qa"}:
            programme_match_status = "not_applicable"
        elif expected_programme:
            programme_match_status = "match" if actual_programme == expected_programme else "mismatch"
        else:
            programme_match_status = "not_required"

        actual_review_required = bool(fields["actual_review_required"])
        if mode in {"baseline_qa", "conversational_qa"}:
            review_match_status = "not_applicable"
        else:
            review_match_status = "match" if actual_review_required == expected_review_required else "mismatch"

        return {
            "evaluation_run_id": evaluation_run_id,
            "run_timestamp": run_timestamp,
            "case_id": case.get("case_id", ""),
            "dataset_group": case.get("dataset_group", ""),
            "mode": mode,
            "input_text": input_text,
            "expected_programme": expected_programme or "",
            "actual_programme": actual_programme,
            "programme_match_status": programme_match_status,
            "expected_topics": ", ".join(expected_topics),
            "detected_topics": ", ".join(detected_topics),
            "missing_topics": (
                "not_applicable"
                if mode in {"baseline_qa", "conversational_qa"}
                else metrics["missing_topics"]
            ),
            "extra_topics": (
                "not_applicable"
                if mode in {"baseline_qa", "conversational_qa"}
                else metrics["extra_topics"]
            ),
            "topic_coverage_percent": (
                "not_applicable"
                if mode in {"baseline_qa", "conversational_qa"}
                else metrics["topic_coverage_percent"]
            ),
            "expected_review_required": normalise_bool(expected_review_required),
            "actual_review_required": normalise_bool(actual_review_required),
            "review_match_status": review_match_status,
            "review_reason": fields["review_reason"],
            "answerable": case.get("answerable", ""),
            "citation_count": fields["citation_count"],
            "source_count": fields["source_count"],
            "source_urls": " | ".join(fields["source_urls"]),
            "response_time_seconds": elapsed,
            "quality_label": fields["quality_label"],
            "quality_score": fields["quality_score"],
            "is_grounded": fields["is_grounded"],
            "grounding_confidence": fields["grounding_confidence"],
            "generated_output": fields["generated_output"],
            "http_status": resp.status_code,
            "error": "",
        }

    except Exception as e:
        elapsed = round(time.time() - started, 2)
        return {
            "evaluation_run_id": evaluation_run_id,
            "run_timestamp": run_timestamp,
            "case_id": case.get("case_id", ""),
            "dataset_group": case.get("dataset_group", ""),
            "mode": mode,
            "input_text": input_text,
            "expected_programme": expected_programme or "",
            "actual_programme": "",
            "programme_match_status": "error",
            "expected_topics": ", ".join(expected_topics),
            "detected_topics": "",
            "missing_topics": ", ".join(expected_topics),
            "extra_topics": "",
            "topic_coverage_percent": 0,
            "expected_review_required": normalise_bool(expected_review_required),
            "actual_review_required": "",
            "review_match_status": "error",
            "review_reason": "",
            "answerable": case.get("answerable", ""),
            "citation_count": "",
            "source_count": "",
            "source_urls": "",
            "response_time_seconds": elapsed,
            "quality_label": "error",
            "quality_score": "",
            "is_grounded": "",
            "grounding_confidence": "",
            "generated_output": "",
            "http_status": "",
            "error": str(e)[:1000],
        }


def append_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases()

    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    evaluation_run_id = args.run_id or datetime.now().strftime("eval_%Y%m%d_%H%M%S")

    rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("HANS four-mode evaluation runner")
    print(f"Cases: {len(cases)}")
    print(f"Run ID: {evaluation_run_id}")
    print("=" * 80)

    for case in cases:
        print(f"\nCase: {case.get('case_id')} - {case.get('title')}")
        for mode in MODES:
            row = run_one(args.api_url, case, mode, evaluation_run_id)
            rows.append(row)
            print(
                f"  {mode}: status={row['http_status']} | "
                f"topics={row['topic_coverage_percent']} | "
                f"programme={row['programme_match_status']} | "
                f"review={row['review_match_status']} | "
                f"time={row['response_time_seconds']}s"
            )

    append_csv(RESULTS_CSV, rows)
    append_jsonl(RESULTS_JSONL, rows)

    print("\nSaved:")
    print(f"CSV:   {RESULTS_CSV}")
    print(f"JSONL: {RESULTS_JSONL}")


if __name__ == "__main__":
    main()