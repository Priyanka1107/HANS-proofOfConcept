# scripts/test_email_thread_flow.py
"""
Run V5 threaded email workflow tests against the /email endpoint.

Purpose:
- test that a first email creates thread memory,
- test that a normal follow-up can reuse prior context,
- test that unclear/correction follow-ups are flagged for human review.

Usage:
    python scripts/test_email_thread_flow.py --version v5_thread_memory_test --comment "thread memory and follow-up routing"
"""
from __future__ import annotations

import os
import sys
import json
import csv
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
CASES_PATH = os.path.join(EVAL_DIR, "email_thread_test_cases_v5.json")
RESULTS_CSV = os.path.join(EVAL_DIR, "email_thread_v5_results.csv")
RESULTS_JSONL = os.path.join(EVAL_DIR, "email_thread_v5_results.jsonl")


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
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


def run_turn(api_url: str, case: Dict[str, Any], turn: Dict[str, Any], version: str, comment: str) -> Dict[str, Any]:
    started = time.time()
    run_timestamp = datetime.now().isoformat()

    payload = {
        "email_text": turn["email_text"],
        "student_email": case.get("student_email"),
        "subject": case.get("subject"),
        "thread_id": case.get("thread_id"),
        "top_k": 3,
    }

    try:
        resp = requests.post(f"{api_url}/email", json=payload, timeout=120)
        elapsed = round(time.time() - started, 2)
        if not resp.ok:
            return {
                "run_timestamp": run_timestamp,
                "version_label": version,
                "comment": comment,
                "case_id": case.get("case_id", ""),
                "turn_index": turn.get("turn_index", ""),
                "student_email": case.get("student_email", ""),
                "thread_id": case.get("thread_id", ""),
                "subject": case.get("subject", ""),
                "email_text": turn["email_text"],
                "expected_followup_type": turn.get("expected_followup_type", ""),
                "actual_followup_type": "",
                "review_required": "Yes",
                "quality_label": "error",
                "detected_topics": "",
                "staff_draft": "",
                "response_time_seconds": elapsed,
                "http_status": resp.status_code,
                "error": resp.text[:500],
            }

        data = resp.json()
        quality = data.get("quality", {})
        detected_topics = [t.get("topic_id", "") for t in data.get("detected_topics", [])]
        return {
            "run_timestamp": run_timestamp,
            "version_label": version,
            "comment": comment,
            "case_id": case.get("case_id", ""),
            "turn_index": turn.get("turn_index", ""),
            "student_email": case.get("student_email", ""),
            "thread_id": data.get("thread_id") or case.get("thread_id", ""),
            "subject": case.get("subject", ""),
            "email_text": turn["email_text"],
            "expected_followup_type": turn.get("expected_followup_type", ""),
            "actual_followup_type": data.get("followup_type", ""),
            "review_required": "Yes" if quality.get("review_required") else "No",
            "quality_label": quality.get("quality_label", ""),
            "detected_topics": ", ".join(detected_topics),
            "staff_draft": data.get("staff_draft", ""),
            "response_time_seconds": elapsed,
            "http_status": resp.status_code,
            "error": "",
        }
    except Exception as e:
        elapsed = round(time.time() - started, 2)
        return {
            "run_timestamp": run_timestamp,
            "version_label": version,
            "comment": comment,
            "case_id": case.get("case_id", ""),
            "turn_index": turn.get("turn_index", ""),
            "student_email": case.get("student_email", ""),
            "thread_id": case.get("thread_id", ""),
            "subject": case.get("subject", ""),
            "email_text": turn["email_text"],
            "expected_followup_type": turn.get("expected_followup_type", ""),
            "actual_followup_type": "",
            "review_required": "Yes",
            "quality_label": "error",
            "detected_topics": "",
            "staff_draft": "",
            "response_time_seconds": elapsed,
            "http_status": "",
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
    print("HANS V5 Threaded Email Test Runner")
    print(f"Cases: {len(cases)} | Version: {args.version}")
    print("=" * 80)

    for case in cases:
        print(f"\n[{case.get('case_id')}] {case.get('title')}")
        for turn in case.get("turns", []):
            row = run_turn(args.api_url, case, turn, args.version, args.comment)
            rows.append(row)
            print(
                f"  Turn {row['turn_index']}: expected={row['expected_followup_type']} | "
                f"actual={row['actual_followup_type']} | quality_warning={row['review_required']} | "
                f"topics={row['detected_topics']} | time={row['response_time_seconds']}s"
            )

    write_csv(RESULTS_CSV, rows)
    write_jsonl(RESULTS_JSONL, rows)

    print("\nSaved results:")
    print(f"  CSV:   {RESULTS_CSV}")
    print(f"  JSONL: {RESULTS_JSONL}")


if __name__ == "__main__":
    main()
