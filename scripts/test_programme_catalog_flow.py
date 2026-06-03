"""
Programme catalogue test runner for HANS V5.1.

Purpose:
- Test programme-aware retrieval with the programme catalogue layer.
- Supports flat single-email cases and thread-style cases.
- Extracts nested backend fields safely.
- Computes topic coverage using topic equivalence mapping.
- Saves clean CSV and JSONL results.

Run from project root:
    python scripts/test_programme_catalog_flow.py --version v5_1_programme_catalog --comment "V5.1 programme catalogue and programme-aware retrieval"
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
CASES_PATH = EVALUATION_DIR / "programme_catalog_test_cases_v5_1.json"

CSV_PATH = EVALUATION_DIR / "programme_catalog_v51_results.csv"
JSONL_PATH = EVALUATION_DIR / "programme_catalog_v51_results.jsonl"
CATALOGUE_PATH = PROJECT_ROOT / "data" / "programme_catalog.json"


CSV_HEADERS = [
    "timestamp",
    "version",
    "comment",
    "case_id",
    "title",
    "turn_index",
    "student_email",
    "thread_id",
    "subject",
    "email_text",
    "expected_programme",
    "actual_programme",
    "programme_match_status",
    "expected_topic_count",
    "actual_topic_count",
    "expected_topics",
    "detected_topics",
    "topic_coverage_percent",
    "expected_review_required",
    "review_required",
    "evaluation_review_required",
    "quality",
    "score",
    "grounded",
    "confidence",
    "followup_type",
    "response_time_sec",
    "citation_count",
    "num_sources",
    "http_status",
    "draft_response",
    "review_reasons",
    "source_urls",
]


TOPIC_EQUIVALENTS = {
    "application_route": {
        "application_route",
        "application_process",
        "application_via_uni_assist",
        "application_via_htw_portal",
        "application_portal",
        "uni_assist",
        "hochschulstart",
    },
    "application_fee": {
        "application_fee",
        "fees",
        "fee",
        "processing_fee",
        "uni_assist_fee",
        "application_cost",
    },
    "tuition_fees": {
        "tuition_fees",
        "fees",
        "tuition",
        "study_fees",
    },
    "semester_contribution": {
        "semester_contribution",
        "semester_fee",
        "fees",
        "contribution",
    },
    "english_language_requirements": {
        "english_language_requirements",
        "language_requirements",
        "language_requirement",
        "language_of_instruction",
        "english_proof",
        "english_language_proof",
    },
    "german_language_requirements": {
        "german_language_requirements",
        "language_requirements",
        "german_proof",
        "german_language_proof",
    },
    "language_requirements": {
        "language_requirements",
        "english_language_requirements",
        "german_language_requirements",
        "language_of_instruction",
        "english_proof",
        "german_proof",
    },
    "language_of_instruction": {
        "language_of_instruction",
        "language_requirements",
        "english_language_requirements",
        "study_language",
        "taught_in_english",
        "english_taught",
    },
    "study_format": {
        "study_format",
        "on_campus",
        "on-campus",
        "distance_learning",
        "online",
        "attendance",
    },
    "required_documents": {
        "required_documents",
        "documents",
        "document_uploads",
        "certified_translations",
        "official_transcripts",
        "school_certificates",
        "aps",
        "aps_certificate",
    },
    "admission_requirements": {
        "admission_requirements",
        "admission_requirement",
        "eligibility",
        "qualification_recognition",
        "work_experience",
        "required_documents",
    },
    "qualification_recognition": {
        "qualification_recognition",
        "admission_requirements",
        "ib",
        "international_baccalaureate",
        "anabin",
        "daad",
        "higher_education_entrance_qualification",
    },
    "motivation_letter": {
        "motivation_letter",
        "letter_of_motivation",
        "required_documents",
        "documents",
    },
    "work_experience": {
        "work_experience",
        "professional_experience",
        "qualified_professional_experience",
        "admission_requirements",
    },
    "application_deadline": {
        "application_deadline",
        "deadline",
        "application_period",
        "application_periods",
    },
    "application_before_graduation": {
        "application_before_graduation",
        "apply_before_graduation",
        "provisional_application",
        "final_certificate_submission",
        "preliminary_average_grade",
    },
    "final_certificate_submission": {
        "final_certificate_submission",
        "final_certificate",
        "degree_completion_proof",
        "application_before_graduation",
    },
}


UNCERTAIN_PHRASES = [
    "please check the specific programme website",
    "please check the programme website",
    "please check directly",
    "contact the admissions team",
    "contact student services",
    "not specified in the documents",
    "not available in the documents",
    "i do not have this information",
    "the documents do not contain",
    "cannot be determined",
    "not clearly stated",
]


def load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = load_json_file(path)

    if not isinstance(cases, list):
        raise ValueError("Test case file must contain a JSON list.")

    return cases


def load_programme_catalogue() -> List[Dict[str, Any]]:
    if not CATALOGUE_PATH.exists():
        return []

    try:
        data = load_json_file(CATALOGUE_PATH)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def normalise_to_turns(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(case.get("turns"), list):
        return case["turns"]

    return [
        {
            "student_email": case.get("student_email", ""),
            "thread_id": case.get("thread_id", case.get("case_id", "")),
            "subject": case.get("subject", ""),
            "email_text": case.get("email_text", ""),
            "expected_topics": case.get("expected_topics", []),
            "expected_topic_count": case.get("expected_topic_count", ""),
            "expected_programme": case.get("expected_programme", ""),
            "expected_review_required": case.get("expected_review_required", ""),
            "expected_followup_type": case.get("expected_followup_type", "new_enquiry"),
        }
    ]


def as_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        if not value.strip():
            return []
        return [v.strip() for v in value.split(",") if v.strip()]

    return [str(value).strip()]


def parse_maybe_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            try:
                return ast.literal_eval(value)
            except Exception:
                try:
                    return json.loads(value)
                except Exception:
                    return value

    return value


def get_nested(data: Dict[str, Any], paths: List[List[str]], default: Any = "") -> Any:
    for path in paths:
        current: Any = data

        for key in path:
            current = parse_maybe_dict(current)

            if not isinstance(current, dict):
                current = None
                break

            current = current.get(key)

        if current not in [None, ""]:
            return current

    return default


def quality_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    quality = parse_maybe_dict(data.get("quality", {}))

    if isinstance(quality, dict):
        return quality

    return {}


def extract_draft(data: Dict[str, Any]) -> str:
    candidates = [
        data.get("draft_response"),
        data.get("draft"),
        data.get("answer"),
        data.get("response"),
        data.get("message"),
        data.get("final_answer"),
        get_nested(data, [["email", "draft_response"]]),
        get_nested(data, [["result", "draft_response"]]),
        get_nested(data, [["result", "answer"]]),
        get_nested(data, [["quality", "draft_response"]]),
    ]

    for value in candidates:
        if value:
            return str(value)

    return ""


def extract_quality_label(data: Dict[str, Any]) -> str:
    q = quality_dict(data)

    candidates = [
        data.get("quality_label"),
        q.get("quality_label"),
        q.get("label"),
        q.get("quality"),
        data.get("quality") if isinstance(data.get("quality"), str) else "",
    ]

    for value in candidates:
        if value:
            return str(value)

    return ""


def extract_score(data: Dict[str, Any]) -> Any:
    q = quality_dict(data)
    return data.get("score", q.get("score", ""))


def extract_grounded(data: Dict[str, Any]) -> Any:
    q = quality_dict(data)
    return data.get("grounded", q.get("grounded", q.get("is_grounded", "")))


def extract_confidence(data: Dict[str, Any]) -> Any:
    q = quality_dict(data)
    return data.get("confidence", q.get("confidence", q.get("grounding_confidence", "")))


def extract_review_required(data: Dict[str, Any]) -> str:
    q = quality_dict(data)

    candidates = [
        data.get("review_required"),
        data.get("requires_review"),
        q.get("review_required"),
        q.get("requires_review"),
        get_nested(data, [["analysis", "review_required"]]),
        get_nested(data, [["analysis", "requires_review"]]),
    ]

    for value in candidates:
        if value in [None, ""]:
            continue

        if isinstance(value, bool):
            return "Yes" if value else "No"

        value_str = str(value).strip().lower()

        if value_str in {"true", "yes", "1"}:
            return "Yes"

        if value_str in {"false", "no", "0"}:
            return "No"

        return str(value)

    return ""


def extract_review_reasons(data: Dict[str, Any]) -> str:
    q = quality_dict(data)

    candidates = [
        data.get("review_reasons"),
        data.get("review_reason"),
        data.get("reasons"),
        q.get("review_reasons"),
        q.get("review_reason"),
        q.get("reasons"),
        get_nested(data, [["analysis", "review_reasons"]]),
    ]

    for value in candidates:
        if value in [None, ""]:
            continue

        if isinstance(value, list):
            return "; ".join(str(v) for v in value)

        return str(value)

    return ""


def extract_detected_topics(data: Dict[str, Any]) -> List[str]:
    topic_keys = [
        "detected_topics",
        "topics",
        "topic_names",
        "sub_intents",
        "detected_sub_intents",
        "intents",
    ]

    for key in topic_keys:
        value = data.get(key)
        topics = normalise_topic_list(value)
        if topics:
            return topics

    for container_key in ["analysis", "result", "email_analysis", "metadata"]:
        container = parse_maybe_dict(data.get(container_key))
        if isinstance(container, dict):
            for key in topic_keys:
                topics = normalise_topic_list(container.get(key))
                if topics:
                    return topics

    return []


def normalise_topic_list(value: Any) -> List[str]:
    if not value:
        return []

    topics = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                topic = (
                    item.get("topic")
                    or item.get("name")
                    or item.get("intent")
                    or item.get("sub_intent")
                    or item.get("label")
                    or ""
                )
                if topic:
                    topics.append(str(topic).strip())
            else:
                topics.append(str(item).strip())

    elif isinstance(value, str):
        topics = [v.strip() for v in value.split(",") if v.strip()]

    return [t for t in topics if t]


def extract_sources(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = [
        data.get("sources"),
        data.get("retrieved_sources"),
        data.get("citations"),
        get_nested(data, [["result", "sources"]], []),
        get_nested(data, [["quality", "citations"]], []),
    ]

    for value in candidates:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def extract_source_urls(sources: List[Dict[str, Any]]) -> List[str]:
    urls = []

    for source in sources:
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}

        url = (
            source.get("url")
            or source.get("source_url")
            or source.get("link")
            or metadata.get("url")
            or metadata.get("source_url")
        )

        if url:
            urls.append(str(url))

    return unique_keep_order_str(urls)


def unique_keep_order_str(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        item = str(item).strip()
        if not item:
            continue

        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def canonical_topic(topic: str) -> str:
    topic = str(topic).strip().lower()

    for canonical, aliases in TOPIC_EQUIVALENTS.items():
        if topic == canonical:
            return canonical
        if topic in {a.lower() for a in aliases}:
            return canonical

    return topic


def topic_matches(expected: str, detected: str) -> bool:
    expected_norm = canonical_topic(expected)
    detected_norm = canonical_topic(detected)

    if expected_norm == detected_norm:
        return True

    expected_aliases = {a.lower() for a in TOPIC_EQUIVALENTS.get(expected_norm, {expected_norm})}
    detected_aliases = {a.lower() for a in TOPIC_EQUIVALENTS.get(detected_norm, {detected_norm})}

    return bool(expected_aliases.intersection(detected_aliases))


def calculate_topic_coverage(expected: List[str], detected: List[str]) -> float:
    expected_clean = [e for e in expected if e]
    detected_clean = [d for d in detected if d]

    if not expected_clean:
        return 0.0

    matched = 0

    for expected_topic in expected_clean:
        if any(topic_matches(expected_topic, detected_topic) for detected_topic in detected_clean):
            matched += 1

    return round((matched / len(expected_clean)) * 100, 2)


def load_catalogue_programmes(catalogue: List[Dict[str, Any]]) -> List[str]:
    names = []

    for item in catalogue:
        if item.get("program_name"):
            names.append(str(item["program_name"]))

        for alias in item.get("aliases", []) or []:
            names.append(str(alias))

    return unique_keep_order_str(names)


def expected_programme_match_status(expected: Any, actual: str, email_text: str, catalogue: List[Dict[str, Any]]) -> str:
    if expected in [None, "", "null"]:
        catalogue_names = load_catalogue_programmes(catalogue)
        lowered_email = email_text.lower()

        matched_known = [
            name
            for name in catalogue_names
            if len(name) >= 4 and name.lower() in lowered_email
        ]

        if matched_known:
            return "unexpected_catalogue_match"

        return "unknown_programme_expected"

    expected_str = str(expected).strip().lower()
    actual_str = str(actual or "").strip().lower()

    if actual_str and (expected_str == actual_str or expected_str in actual_str or actual_str in expected_str):
        return "matched"

    if not actual_str:
        return "not_reported_by_backend"

    return "mismatch"


def extract_actual_programme(data: Dict[str, Any], email_text: str, catalogue: List[Dict[str, Any]]) -> str:
    candidates = [
        data.get("target_program"),
        data.get("target_programme"),
        data.get("matched_programme"),
        data.get("programme"),
        data.get("program"),
        get_nested(data, [["analysis", "target_program"]]),
        get_nested(data, [["analysis", "matched_programme"]]),
        get_nested(data, [["metadata", "target_program"]]),
    ]

    for value in candidates:
        if value:
            return str(value)

    # Fallback: infer from email text using catalogue.
    lowered = email_text.lower()

    for item in catalogue:
        name = str(item.get("program_name", ""))
        aliases = [str(a) for a in item.get("aliases", []) or []]

        values = [name] + aliases

        for value in values:
            if len(value) >= 4 and value.lower() in lowered:
                return name

    return ""


def contains_uncertain_phrase(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in UNCERTAIN_PHRASES)


def compute_evaluation_review_required(
    expected_programme: Any,
    actual_programme: str,
    review_required: str,
    draft: str,
    topic_coverage: float,
) -> str:
    if str(review_required).strip().lower() == "yes":
        return "Yes"

    if expected_programme in [None, "", "null"] and not actual_programme:
        return "Yes"

    if contains_uncertain_phrase(draft):
        return "Yes"

    if topic_coverage < 100:
        return "Yes"

    return "No"


def count_citations(draft: str, sources: List[Dict[str, Any]]) -> int:
    doc_count = len(re.findall(r"\[Doc\s*\d+\]", draft or "", flags=re.I))

    if doc_count > 0:
        return doc_count

    return len(sources)


def run_case(
    api_url: str,
    case: Dict[str, Any],
    turn: Dict[str, Any],
    turn_index: int,
    version: str,
    comment: str,
    catalogue: List[Dict[str, Any]],
) -> Dict[str, Any]:
    student_email = turn.get("student_email") or case.get("student_email") or ""
    thread_id = turn.get("thread_id") or case.get("thread_id") or case.get("case_id") or ""
    subject = turn.get("subject") or case.get("subject") or ""
    email_text = turn.get("email_text") or case.get("email_text") or ""

    expected_topics = as_list(turn.get("expected_topics", case.get("expected_topics", [])))
    expected_topic_count = turn.get("expected_topic_count", case.get("expected_topic_count", len(expected_topics)))
    expected_programme = turn.get("expected_programme", case.get("expected_programme", ""))
    expected_review_required = turn.get("expected_review_required", case.get("expected_review_required", ""))

    payload = {
        "student_email": student_email,
        "thread_id": thread_id,
        "subject": subject,
        "email_text": email_text,
    }

    start = time.time()

    try:
        response = requests.post(f"{api_url.rstrip('/')}/email", json=payload, timeout=120)
        response_time = round(time.time() - start, 2)
        http_status = response.status_code

        try:
            data = response.json()
        except Exception:
            data = {
                "draft_response": response.text,
                "error": "Response was not valid JSON",
            }

    except Exception as exc:
        response_time = round(time.time() - start, 2)
        http_status = 0
        data = {
            "draft_response": "",
            "error": str(exc),
        }

    detected_topics = extract_detected_topics(data)
    sources = extract_sources(data)
    source_urls = extract_source_urls(sources)
    draft = extract_draft(data)
    actual_programme = extract_actual_programme(data, email_text, catalogue)
    programme_status = expected_programme_match_status(expected_programme, actual_programme, email_text, catalogue)
    review_required = extract_review_required(data)
    topic_coverage = calculate_topic_coverage(expected_topics, detected_topics)

    evaluation_review_required = compute_evaluation_review_required(
        expected_programme=expected_programme,
        actual_programme=actual_programme,
        review_required=review_required,
        draft=draft,
        topic_coverage=topic_coverage,
    )

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "version": version,
        "comment": comment,
        "case_id": case.get("case_id", ""),
        "title": case.get("title", ""),
        "turn_index": turn_index,
        "student_email": student_email,
        "thread_id": thread_id,
        "subject": subject,
        "email_text": email_text,
        "expected_programme": "" if expected_programme is None else expected_programme,
        "actual_programme": actual_programme,
        "programme_match_status": programme_status,
        "expected_topic_count": expected_topic_count,
        "actual_topic_count": len(detected_topics),
        "expected_topics": ", ".join(expected_topics),
        "detected_topics": ", ".join(detected_topics),
        "topic_coverage_percent": topic_coverage,
        "expected_review_required": expected_review_required,
        "review_required": review_required,
        "evaluation_review_required": evaluation_review_required,
        "quality": extract_quality_label(data),
        "score": extract_score(data),
        "grounded": extract_grounded(data),
        "confidence": extract_confidence(data),
        "followup_type": data.get("followup_type", get_nested(data, [["analysis", "followup_type"]], "")),
        "response_time_sec": response_time,
        "citation_count": count_citations(draft, sources),
        "num_sources": len(sources),
        "http_status": http_status,
        "draft_response": draft,
        "review_reasons": extract_review_reasons(data),
        "source_urls": " | ".join(source_urls),
    }

    return row


def save_rows(rows: List[Dict[str, Any]]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No rows generated.")
        return

    total = len(rows)
    avg_time = sum(float(r.get("response_time_sec") or 0) for r in rows) / total
    avg_coverage = sum(float(r.get("topic_coverage_percent") or 0) for r in rows) / total
    review_count = sum(1 for r in rows if str(r.get("evaluation_review_required", "")).lower() == "yes")
    programme_matched = sum(1 for r in rows if str(r.get("programme_match_status", "")) == "matched")

    print("=" * 90)
    print("Programme catalogue test finished")
    print(f"Rows: {total}")
    print(f"Programme matched: {programme_matched}/{total}")
    print(f"Evaluation review required: {review_count}/{total}")
    print(f"Avg. response time: {avg_time:.2f} sec")
    print(f"Avg. topic coverage: {avg_coverage:.2f}%")
    print(f"CSV: {CSV_PATH}")
    print(f"JSONL: {JSONL_PATH}")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--version", default="v5_1_programme_catalog")
    parser.add_argument("--comment", default="")
    args = parser.parse_args()

    cases = load_cases(CASES_PATH)
    catalogue = load_programme_catalogue()

    print("=" * 90)
    print("HANS V5.1 Programme Catalogue Test Runner")
    print(f"Cases: {len(cases)} | Version: {args.version} | API: {args.api_url}")
    print(f"Programme catalogue entries: {len(catalogue)}")
    print("=" * 90)

    rows: List[Dict[str, Any]] = []

    for case in cases:
        turns = normalise_to_turns(case)

        for index, turn in enumerate(turns, start=1):
            print(f"[{case.get('case_id')}] Turn {index}: {case.get('title', '')}")

            row = run_case(
                api_url=args.api_url,
                case=case,
                turn=turn,
                turn_index=index,
                version=args.version,
                comment=args.comment,
                catalogue=catalogue,
            )
            rows.append(row)

    save_rows(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()