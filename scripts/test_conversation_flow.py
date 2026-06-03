# scripts/test_conversation_flow.py
"""
Run HANS test cases against the API and save results for later comparison.

Works for:
- baseline mode
- conversation mode
- multi-topic email testing

Usage examples:
    python scripts/test_conversation_flow.py --version v3_baseline_multitopic --mode baseline --comment "multi-topic email baseline test"

    python scripts/test_conversation_flow.py --version v3_conversation_multitopic --mode conversation --comment "multi-topic email conversation test"
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import config

EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

DEFAULT_CASES_PATH = os.path.join(EVAL_DIR, "conversation_test_cases.json")


def ensure_eval_dir() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def decide_quality(
    fallback_used: bool,
    is_grounded: Any,
    confidence: Any,
    error: str,
) -> Dict[str, str]:
    """
    Simple automatic quality indicator.

    This is not final human evaluation.
    It is only a first technical signal for review.
    """

    confidence_value = safe_float(confidence)

    if error:
        return {
            "quality_label": "error",
            "staff_review_required": "Yes",
            "review_reason": "API or runtime error",
        }

    if fallback_used:
        return {
            "quality_label": "unsafe",
            "staff_review_required": "Yes",
            "review_reason": "Fallback triggered",
        }

    if is_grounded is False or str(is_grounded).lower() == "false":
        return {
            "quality_label": "weak",
            "staff_review_required": "Yes",
            "review_reason": "Response not grounded",
        }

    if confidence_value is not None and confidence_value < 0.60:
        return {
            "quality_label": "partial",
            "staff_review_required": "Yes",
            "review_reason": "Low confidence",
        }

    return {
        "quality_label": "good",
        "staff_review_required": "No",
        "review_reason": "",
    }


def append_csv(path: str, rows: List[Dict[str, Any]]) -> None:

    fieldnames=[

    "run_timestamp",
    "version_label",
    "mode",
    "comment",

    "test_id",
    "title",

    "turn_index",

    "full_query",

    "response",

    "response_time_seconds",

    "session_id",

    "standalone_query",

    "detected_intent",

    "enhanced_query",

    "used_memory",

    "fallback_used",
    "fallback_reason",

    "citations",

    "citation_count",

    "num_sources",

    "is_grounded",

    "confidence",

    "expected_topic_count",
    "detected_topic_count",
    "topic_coverage_percent",

    "quality_label",

    "staff_review_required",

    "review_reason",

    "manual_quality",

    "response_style_match",

    "uncertain_sections",

    "http_status",

    "error"
    ]

    write_header=(
        not os.path.exists(path)
    ) or (
        os.path.getsize(path)==0
    )

    with open(
        path,
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer=csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)
            

def run_case(
    api_url: str,
    case: Dict[str, Any],
    version_label: str,
    mode: str,
    comment: str,
    timeout: int = 90,
) -> List[Dict[str, Any]]:
    run_timestamp = datetime.now().isoformat()
    rows: List[Dict[str, Any]] = []

    session_id: Optional[str] = None
    turns = case.get("turns", [])

    for idx, turn in enumerate(turns, start=1):
        payload = {
            "query": turn,
            "session_id": session_id,
            "top_k": 5,
            "mode": mode,
        }

        http_status = None
        error = ""
        response = ""
        response_time_seconds = ""

        standalone_query = ""
        detected_intent = ""
        used_memory = False
        enhanced_query = ""
        fallback_used = False
        fallback_reason = ""
        citations = ""
        citation_count = 0
        num_sources = 0
        is_grounded = ""
        confidence = ""

        try:
            start_time = time.time()

            resp = requests.post(
                f"{api_url}/query",
                json=payload,
                timeout=timeout,
            )

            response_time_seconds = round(time.time() - start_time, 2)
            http_status = resp.status_code

            if resp.ok:
                data = resp.json()

                session_id = data.get("session_id", session_id)

                response = data.get("answer", "")

                standalone_query = data.get("standalone_query", "")
                detected_intent = data.get("detected_intent", "")
                used_memory = data.get("used_memory", False)
                enhanced_query = data.get("enhanced_query", "")

                fallback_used = data.get("fallback_used", False)
                fallback_reason = data.get("fallback_reason", "")

                citation_list = data.get("citations", [])
                citations = ", ".join(citation_list)
                citation_count = len(citation_list)

                sources = data.get("sources", [])
                num_sources = len(sources)

                validation = data.get("validation", {})
                is_grounded = validation.get("is_grounded", "")
                confidence = validation.get("confidence", "")

            else:
                error = resp.text

        except Exception as e:
            error = str(e)

        quality = decide_quality(
            fallback_used=fallback_used,
            is_grounded=is_grounded,
            confidence=confidence,
            error=error,
        )

        row = {
            "run_timestamp": run_timestamp,
            "version_label": version_label,
            "mode": mode,
            "comment": comment,
            "test_id": case.get("case_id", ""),
            "title": case.get("title", ""),
            "turn_index": idx,
            "full_query": turn,
            "response": response,
            "response_time_seconds": response_time_seconds,
            "session_id": session_id or "",
            "standalone_query": standalone_query,
            "detected_intent": detected_intent,
            "enhanced_query": enhanced_query,
            "used_memory": used_memory,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "citations": citations,
            "citation_count": citation_count,
            "num_sources": num_sources,
            "is_grounded": is_grounded,
            "confidence": confidence,
            "quality_label": quality["quality_label"],
            "staff_review_required": quality["staff_review_required"],
            "review_reason": quality["review_reason"],
            "http_status": http_status,
            "error": error,
        }

        rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--api-url",
        default=getattr(config, "BACKEND_URL", "http://127.0.0.1:8001"),
        help="Base backend URL, e.g. http://127.0.0.1:8001",
    )

    p.add_argument(
        "--version",
        required=True,
        help="Version label for this run, e.g. v3_baseline_multitopic",
    )

    p.add_argument(
        "--mode",
        choices=["baseline", "conversation"],
        required=True,
        help="Testing mode",
    )

    p.add_argument(
        "--comment",
        default="",
        help="Short note about what is being tested in this run",
    )

    p.add_argument(
        "--cases-file",
        default=DEFAULT_CASES_PATH,
        help="Path to test cases JSON file",
    )

    p.add_argument(
        "--output-prefix",
        default="multitopic_test_results",
        help="Output file prefix inside evaluation folder",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_eval_dir()

    cases_path = args.cases_file

    if not os.path.isabs(cases_path):
        cases_path = os.path.join(PROJECT_ROOT, cases_path)

    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"Missing test cases file: {cases_path}")

    results_jsonl_path = os.path.join(EVAL_DIR, f"{args.output_prefix}.jsonl")
    results_csv_path = os.path.join(EVAL_DIR, f"{args.output_prefix}.csv")

    cases = load_cases(cases_path)
    all_rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("HANS Test Runner")
    print("=" * 80)
    print(f"API URL: {args.api_url}")
    print(f"Version: {args.version}")
    print(f"Mode: {args.mode}")
    print(f"Comment: {args.comment}")
    print(f"Cases file: {cases_path}")
    print(f"Cases loaded: {len(cases)}")
    print(f"Output CSV: {results_csv_path}")
    print()

    for case in cases:
        print(f"Running {case.get('case_id')} - {case.get('title')}")

        rows = run_case(
            api_url=args.api_url,
            case=case,
            version_label=args.version,
            mode=args.mode,
            comment=args.comment,
        )

        all_rows.extend(rows)

        for row in rows:
            print(f"  Turn {row['turn_index']}")
            print(f"    mode: {row['mode']}")
            print(f"    detected_intent: {row['detected_intent']}")
            print(f"    fallback_used: {row['fallback_used']} ({row['fallback_reason']})")
            print(f"    grounded: {row['is_grounded']}")
            print(f"    confidence: {row['confidence']}")
            print(f"    quality: {row['quality_label']}")
            print(f"    staff_review_required: {row['staff_review_required']}")
            print(f"    time: {row['response_time_seconds']} sec")

            if row["error"]:
                print(f"    error: {row['error']}")

        print()

    append_jsonl(results_jsonl_path, all_rows)
    append_csv(results_csv_path, all_rows)

    print("=" * 80)
    print("Saved results to:")
    print(f"  JSONL: {results_jsonl_path}")
    print(f"  CSV:   {results_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()