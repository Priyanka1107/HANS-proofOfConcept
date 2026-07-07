# summarize_hans_response_times.py
"""
Summarise response-time metrics from HANS evaluation CSV files.

Default input:
  evaluation/hans_combined_10query_original_followups_latest.csv

Outputs:
  evaluation/hans_response_time_summary_<timestamp>.csv
  evaluation/hans_response_time_per_request_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except Exception:
        return None


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * pct
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return values[int(position)]

    lower_value = values[lower_index]
    upper_value = values[upper_index]
    weight = position - lower_index

    return lower_value + (upper_value - lower_value) * weight


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_summary(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped_rows: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        latency = parse_float(row.get("latency_seconds"))

        if latency is None:
            continue

        group_key = (
            row.get("suite", ""),
            row.get("system", ""),
            row.get("mode", ""),
            row.get("http_ok", ""),
        )

        grouped_rows[group_key].append(row)

    summary_rows: List[Dict[str, Any]] = []

    for (suite, system, mode, http_ok), group in sorted(grouped_rows.items()):
        latencies = [
            parse_float(row.get("latency_seconds"))
            for row in group
        ]
        latencies = [value for value in latencies if value is not None]

        if not latencies:
            continue

        successful_rows = [
            row for row in group
            if parse_bool(row.get("http_ok", ""))
        ]

        failed_rows = [
            row for row in group
            if not parse_bool(row.get("http_ok", ""))
        ]

        review_required_rows = [
            row for row in group
            if parse_bool(row.get("review_required", ""))
        ]

        grounded_rows = [
            row for row in group
            if parse_bool(row.get("grounded", ""))
        ]

        summary_rows.append(
            {
                "suite": suite,
                "system": system,
                "mode": mode,
                "http_ok_group": http_ok,
                "rows": len(group),
                "successful_rows": len(successful_rows),
                "failed_rows": len(failed_rows),
                "success_rate_percent": round((len(successful_rows) / len(group)) * 100, 1),
                "avg_latency_seconds": round(mean(latencies), 3),
                "median_latency_seconds": round(median(latencies), 3),
                "min_latency_seconds": round(min(latencies), 3),
                "max_latency_seconds": round(max(latencies), 3),
                "p90_latency_seconds": round(percentile(latencies, 0.90), 3),
                "stddev_latency_seconds": round(pstdev(latencies), 3) if len(latencies) > 1 else 0.0,
                "review_required_rows": len(review_required_rows),
                "grounded_rows": len(grounded_rows),
            }
        )

    return summary_rows


def build_per_request(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    per_request_rows: List[Dict[str, Any]] = []

    for row in rows:
        latency = parse_float(row.get("latency_seconds"))

        if latency is None:
            continue

        answer_preview = (row.get("answer", "") or "").replace("\n", " ")[:250]
        error_preview = (row.get("error", "") or "").replace("\n", " ")[:250]

        per_request_rows.append(
            {
                "suite": row.get("suite", ""),
                "case_id": row.get("case_id", ""),
                "turn_index": row.get("turn_index", ""),
                "title": row.get("title", ""),
                "system": row.get("system", ""),
                "mode": row.get("mode", ""),
                "http_ok": row.get("http_ok", ""),
                "status_code": row.get("status_code", ""),
                "latency_seconds": round(latency, 3),
                "quality_score": row.get("quality_score", ""),
                "quality_label": row.get("quality_label", ""),
                "review_required": row.get("review_required", ""),
                "grounded": row.get("grounded", ""),
                "detected_topics": row.get("detected_topics", ""),
                "answer_preview": answer_preview,
                "error_preview": error_preview,
            }
        )

    per_request_rows.sort(
        key=lambda row: (
            row["suite"],
            row["system"],
            row["mode"],
            str(row["case_id"]),
            str(row["turn_index"]),
        )
    )

    return per_request_rows


def print_original_hans_summary(per_request_rows: List[Dict[str, Any]]) -> None:
    original_rows = [
        row for row in per_request_rows
        if row.get("system") == "original_hans"
        and row.get("mode") == "original_ask"
    ]

    if not original_rows:
        print("\nNo original HANS rows found.")
        return

    latencies = [float(row["latency_seconds"]) for row in original_rows]

    print("\nOriginal HANS response time summary")
    print("-----------------------------------")
    print(f"Requests: {len(original_rows)}")
    print(f"Average:  {mean(latencies):.3f}s")
    print(f"Median:   {median(latencies):.3f}s")
    print(f"Min:      {min(latencies):.3f}s")
    print(f"Max:      {max(latencies):.3f}s")
    print(f"P90:      {percentile(latencies, 0.90):.3f}s")

    print("\nPer original HANS request:")
    for row in original_rows:
        print(f"- {row['case_id']}: {row['latency_seconds']}s | {row['title']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/hans_combined_10query_original_followups_latest.csv",
        help="Path to combined HANS result CSV.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Output tag. Defaults to current timestamp.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = read_rows(input_path)
    summary_rows = build_summary(rows)
    per_request_rows = build_per_request(rows)

    output_dir = input_path.parent

    summary_path = output_dir / f"hans_response_time_summary_{tag}.csv"
    per_request_path = output_dir / f"hans_response_time_per_request_{tag}.csv"

    write_csv(
        summary_path,
        summary_rows,
        [
            "suite",
            "system",
            "mode",
            "http_ok_group",
            "rows",
            "successful_rows",
            "failed_rows",
            "success_rate_percent",
            "avg_latency_seconds",
            "median_latency_seconds",
            "min_latency_seconds",
            "max_latency_seconds",
            "p90_latency_seconds",
            "stddev_latency_seconds",
            "review_required_rows",
            "grounded_rows",
        ],
    )

    write_csv(
        per_request_path,
        per_request_rows,
        [
            "suite",
            "case_id",
            "turn_index",
            "title",
            "system",
            "mode",
            "http_ok",
            "status_code",
            "latency_seconds",
            "quality_score",
            "quality_label",
            "review_required",
            "grounded",
            "detected_topics",
            "answer_preview",
            "error_preview",
        ],
    )

    print_original_hans_summary(per_request_rows)

    print("\nFiles written:")
    print(f"- {summary_path}")
    print(f"- {per_request_path}")


if __name__ == "__main__":
    main()
