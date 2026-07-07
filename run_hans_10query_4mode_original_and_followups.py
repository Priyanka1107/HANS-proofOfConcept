# run_hans_10query_4mode_original_and_followups.py
"""
Runs a compact final regression/evaluation set for HANS.

What it tests:
1) 10 mixed student-service queries on the final PoC in all 4 modes:
   - Email Assistant Claude  -> /email mode=email_claude
   - Email Assistant Mistral  -> /email mode=email_mistral
   - Baseline QA             -> /query mode=baseline_qa
   - Conversational QA       -> /query mode=conversational_qa

2) The same 10 queries on the original HANS backend:
   - Original HANS           -> /ask on port 8080

3) Extra email-thread follow-up tests on the final PoC Email Assistant.
4) Extra QA follow-up tests on final PoC QA modes.

Outputs are written to ./evaluation with timestamped filenames and *_latest files.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FINAL_API_URL = os.getenv("HANS_FINAL_URL", "http://127.0.0.1:8001").rstrip("/")
ORIGINAL_API_URL = os.getenv("HANS_ORIGINAL_URL", "http://127.0.0.1:8080").rstrip("/")
OUT_DIR = Path("evaluation")
TIMEOUT_SECONDS = int(os.getenv("HANS_TEST_TIMEOUT", "90"))


MAIN_CASES: List[Dict[str, str]] = [
    {
        "case_id": "mixed_001_dual_citizenship_french_moroccan_business_bachelor",
        "title": "Dual French/Moroccan citizenship, French Baccalauréat, Business Bachelor route",
        "text": """Dear Admissions Team, I am a dual citizen (French and Moroccan), currently residing in Morocco and holding a French Baccalauréat diploma. I want to apply for the English-taught Bachelor’s in Business but am unsure whether I should apply as an EU applicant or international. Also, would my French diploma fulfill the general university entrance requirements, or would I need to obtain a VPD from Uni-Assist?""",
    },
    {
        "case_id": "mixed_002_eu_portugal_ib_csb_route_motivation",
        "title": "EU citizen in Portugal, IB diploma, Cybersecurity and Business route + motivation letter",
        "text": """Hello, I am in the process of applying to the Cybersecurity and Business Bachelor’s degree and would like to confirm a few details. I am an EU citizen but currently living in Portugal and finishing high school with an International Baccalaureate diploma. Do I still need to apply via Uni-Assist, or is there a different process for IB graduates? Furthermore, is there a separate motivation letter required for this specific program?""",
    },
    {
        "case_id": "mixed_003_bachelor_computer_science_german_deadline_documents",
        "title": "Bachelor Computer Science German proof, deadline, documents",
        "text": """Hello, I would like to apply for a Bachelor's programme in Computer Science. Could you please tell me whether German language proof is needed, what the application deadline is, and which documents are required?""",
    },
    {
        "case_id": "mixed_004_masters_management_generic_long_enquiry",
        "title": "Generic Master's in Management long enquiry",
        "text": """Dear Sir/Madam,

I hope this message finds you well.

I am writing to inquire about the Master’s in Management program offered at your esteemed university. I am highly interested in applying and would appreciate it if you could provide me with the following details:

1. Eligibility Criteria

What academic background is required for this program?

2. Entry Requirements

Is there a minimum GPA or grade requirement?

Are standardized tests (e.g., GMAT, GRE, TOEFL/IELTS) required?

Are letters of recommendation and a motivation letter necessary?

3. Application Process and Deadlines

What are the application deadlines for international applicants for the upcoming intake (Winter 2026)?

Is there an online portal or application platform I should use?

4. Fees and Funding

Could you kindly confirm the tuition and semester fees for international students?

Are there any scholarships or funding opportunities available?

I would be grateful for any brochures, links, or official documents that outline this information.

Thank you very much for your time and assistance. I look forward to your response.""",
    },
    {
        "case_id": "mixed_005_portuguese_international_business_diploma_subjects_route",
        "title": "Portuguese applicant, International Business, diploma subject concern + uni-assist route",
        "text": """Dear Admissions Team,I am Portuguese, and I am interested in applying for the
International Business program at HTW Berlin for the winter semester 2026/27.
I have an official Portuguese secondary education diploma (Curso Secundário de Música –
Ensino Artístico Especializado, issued by Conservatório de Música do Porto in 2019), which
according to the online recognition tools and your guidelines, should technically make me
eligible to apply.
However, I have some concerns regarding my application, because my diploma does not
include certain specific subjects, such as Mathematics or Economics, which appear to be
part of the Bachelor’s curriculum.
Could you please confirm if this will affect my eligibility for the program?
Additionally, I understand that, since my diploma is not German, I must apply through
uni-assist. Could you please confirm that this is the correct application portal for me?
Thank you very much for your support. I look forward to your guidance.
Kind regards,""",
    },
    {
        "case_id": "mixed_006_nigerian_language_program_before_deadline",
        "title": "Nigerian applicant, German B2 language course after deadline",
        "text": """I am a Nigerian high school graduate interested in studying at the
University. I will join a language learning program in Munich to learn the German language
at B2 level because that's the level required by the University but the time it would take
for me to finish would exceed the application deadline of the summer semester. I want to
know if I can still apply before I finish the language learning program which would finish
in December 2026.""",
    },
    {
        "case_id": "mixed_007_continuing_education_master_work_fees_documents",
        "title": "Continuing education Master's work experience, tuition fees, documents",
        "text": """Dear Student Services, I am interested in a continuing education Master's programme. Could you please tell me whether work experience is required, whether tuition fees apply, and which application documents are needed?""",
    },
    {
        "case_id": "mixed_008_german_master_programme_count",
        "title": "German broad Master programme count",
        "text": """Wie viele Masterstudiengänge bietet die HTW Berlin an?""",
    },
    {
        "case_id": "mixed_009_hardcopy_or_digital_upload",
        "title": "Hard-copy documents by post or digital upload",
        "text": """Do I need to send hard-copy documents by post or is digital upload enough?""",
    },
    {
        "case_id": "mixed_010_german_mpmd_deadline_language_format",
        "title": "German MPMD deadline, English language, on-campus format",
        "text": """Sehr geehrtes Zulassungsteam,

ich habe meinen Bachelorabschluss in Electronics Engineering in Indien abgeschlossen und arbeite derzeit als Softwareentwickler. Ich möchte mich für den Masterstudiengang Project Management and Data Science bewerben. Können Sie mir bitte mitteilen, wann die Bewerbungsfrist ist, ob der Studiengang vollständig auf Englisch unterrichtet wird und ob es sich um ein Präsenzstudium handelt?

Mit freundlichen Grüßen
Arjun""",
    },
]


EMAIL_FOLLOWUP_SEQUENCES: List[Dict[str, Any]] = [
    {
        "sequence_id": "email_fup_001_mpmd_deadline_then_language_format_certificate",
        "title": "MPMD follow-up: deadline -> language/format -> certificate later",
        "student_email": "followup.mpmd@example.com",
        "subject": "MPMD application questions",
        "turns": [
            "Dear Student Services, I would like to apply for Project Management and Data Science. What is the application deadline?",
            "Is it fully taught in English and is it on campus?",
            "I am still in my final Bachelor semester. Can I submit the final certificate later?",
        ],
    },
    {
        "sequence_id": "email_fup_002_csb_route_then_motivation_vpd",
        "title": "CSB follow-up: route -> motivation letter -> VPD",
        "student_email": "followup.csb@example.com",
        "subject": "Cybersecurity and Business application",
        "turns": [
            "Hello, I am interested in Cybersecurity and Business. I am an EU citizen with an IB diploma. Where should I apply?",
            "Is a motivation letter required too?",
            "Do I need a VPD from uni-assist?",
        ],
    },
    {
        "sequence_id": "email_fup_003_documents_then_upload_translation",
        "title": "Documents follow-up: post -> digital upload -> translation",
        "student_email": "followup.docs@example.com",
        "subject": "Application documents",
        "turns": [
            "Dear team, for an HTW application, do I need to send hard-copy documents by post?",
            "Can I upload the documents digitally instead?",
            "What if my documents are not in German or English?",
        ],
    },
]


QA_FOLLOWUP_SEQUENCES: List[Dict[str, Any]] = [
    {
        "sequence_id": "qa_fup_001_mpmd",
        "title": "QA memory follow-up: MPMD",
        "turns": [
            "I want to apply for Project Management and Data Science. What is the application deadline?",
            "Is it fully taught in English?",
            "Is it on campus?",
        ],
    },
    {
        "sequence_id": "qa_fup_002_documents",
        "title": "QA memory follow-up: documents",
        "turns": [
            "Do I need to send hard-copy documents by post for an HTW application?",
            "Can I upload them digitally instead?",
            "What about certified translations?",
        ],
    },
]


CSV_COLUMNS = [
    "timestamp", "suite", "case_id", "turn_index", "title", "system", "mode", "endpoint",
    "http_ok", "status_code", "latency_seconds", "provider", "model", "detected_topics",
    "followup_type", "quality_score", "quality_label", "review_required", "review_reason",
    "grounded", "citations_valid", "citation_count", "sources_count", "question", "answer", "error",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def post_json(url: str, payload: Dict[str, Any], timeout: int = TIMEOUT_SECONDS) -> Tuple[bool, int, Dict[str, Any], str, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            latency = time.time() - start
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"raw_body": body}
            return True, int(resp.status), parsed, "", latency
    except urllib.error.HTTPError as e:
        latency = time.time() - start
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw_body": body}
        return False, int(e.code), parsed, str(e), latency
    except Exception as e:
        latency = time.time() - start
        return False, 0, {}, str(e), latency


def extract_answer(data: Dict[str, Any]) -> str:
    for key in ["staff_draft", "answer", "response", "message", "result"]:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(data, ensure_ascii=False)[:3000]


def extract_sources_count(data: Dict[str, Any]) -> int:
    sources = data.get("sources")
    if isinstance(sources, list):
        return len(sources)
    citations = data.get("citations")
    if isinstance(citations, list):
        return len(citations)
    return 0


def extract_topics(data: Dict[str, Any]) -> str:
    topics = data.get("detected_topics")
    if not isinstance(topics, list):
        return ""
    out = []
    for t in topics:
        if isinstance(t, dict):
            out.append(str(t.get("topic_id") or t.get("label") or ""))
        else:
            out.append(str(t))
    return "; ".join([x for x in out if x])


def build_row(*, suite: str, case_id: str, turn_index: Optional[int], title: str, system: str, mode: str, endpoint: str, question: str, ok: bool, status_code: int, data: Dict[str, Any], error: str, latency: float) -> Dict[str, Any]:
    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    quality = data.get("quality") if isinstance(data.get("quality"), dict) else {}
    citations = data.get("citations") if isinstance(data.get("citations"), list) else quality.get("citations", [])
    return {
        "timestamp": now_iso(),
        "suite": suite,
        "case_id": case_id,
        "turn_index": "" if turn_index is None else turn_index,
        "title": title,
        "system": system,
        "mode": mode,
        "endpoint": endpoint,
        "http_ok": ok,
        "status_code": status_code,
        "latency_seconds": round(latency, 3),
        "provider": data.get("generation_provider") or data.get("provider") or "",
        "model": data.get("generation_model") or data.get("model") or "",
        "detected_topics": extract_topics(data),
        "followup_type": data.get("followup_type", ""),
        "quality_score": quality.get("quality_score", ""),
        "quality_label": quality.get("quality_label", ""),
        "review_required": quality.get("review_required", ""),
        "review_reason": quality.get("review_reason", ""),
        "grounded": validation.get("is_grounded", ""),
        "citations_valid": validation.get("citations_valid", ""),
        "citation_count": len(citations) if isinstance(citations, list) else "",
        "sources_count": extract_sources_count(data),
        "question": question,
        "answer": extract_answer(data),
        "error": error,
    }


def write_outputs(rows: List[Dict[str, Any]], base_name: str) -> Tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{base_name}.csv"
    json_path = OUT_DIR / f"{base_name}.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return csv_path, json_path


def run_final_main_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    final_modes = [
        ("final_poc", "email_claude", f"{FINAL_API_URL}/email"),
        ("final_poc", "email_mistral", f"{FINAL_API_URL}/email"),
        ("final_poc", "baseline_qa", f"{FINAL_API_URL}/query"),
        ("final_poc", "conversational_qa", f"{FINAL_API_URL}/query"),
    ]
    for case in MAIN_CASES:
        for system, mode, url in final_modes:
            print(f"[FINAL] {mode} | {case['case_id']}")
            if mode.startswith("email_"):
                payload = {
                    "email_text": case["text"],
                    "student_email": f"{case['case_id']}@example.com",
                    "subject": case["title"],
                    "thread_id": f"run10_{mode}_{case['case_id']}",
                    "mode": mode,
                    "test_id": case["case_id"],
                    "top_k": 3,
                }
            else:
                payload = {
                    "query": case["text"],
                    "mode": mode,
                    "session_id": f"run10_{mode}_{case['case_id']}",
                    "top_k": 5,
                }
            ok, status, data, err, latency = post_json(url, payload)
            rows.append(build_row(suite="main_10_final_4_modes", case_id=case["case_id"], turn_index=None, title=case["title"], system=system, mode=mode, endpoint=url, question=case["text"], ok=ok, status_code=status, data=data, error=err, latency=latency))
    return rows


def run_original_main_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    url = f"{ORIGINAL_API_URL}/ask"
    for case in MAIN_CASES:
        print(f"[ORIGINAL] /ask | {case['case_id']}")
        ok, status, data, err, latency = post_json(url, {"q": case["text"]})
        rows.append(build_row(suite="main_10_original_hans", case_id=case["case_id"], turn_index=None, title=case["title"], system="original_hans", mode="original_ask", endpoint=url, question=case["text"], ok=ok, status_code=status, data=data, error=err, latency=latency))
    return rows


def run_email_followups() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seq in EMAIL_FOLLOWUP_SEQUENCES:
        for mode in ["email_claude", "email_mistral"]:
            thread_id = f"followup_{mode}_{seq['sequence_id']}"
            for idx, question in enumerate(seq["turns"], start=1):
                print(f"[EMAIL FOLLOWUP] {mode} | {seq['sequence_id']} | turn {idx}")
                payload = {
                    "email_text": question,
                    "student_email": seq["student_email"],
                    "subject": seq["subject"],
                    "thread_id": thread_id,
                    "mode": mode,
                    "test_id": f"{seq['sequence_id']}_turn_{idx}",
                    "top_k": 3,
                }
                ok, status, data, err, latency = post_json(f"{FINAL_API_URL}/email", payload)
                rows.append(build_row(suite="email_thread_followups", case_id=seq["sequence_id"], turn_index=idx, title=seq["title"], system="final_poc", mode=mode, endpoint=f"{FINAL_API_URL}/email", question=question, ok=ok, status_code=status, data=data, error=err, latency=latency))
    return rows


def run_qa_followups() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seq in QA_FOLLOWUP_SEQUENCES:
        for mode in ["baseline_qa", "conversational_qa"]:
            session_id = f"qa_followup_{mode}_{seq['sequence_id']}"
            for idx, question in enumerate(seq["turns"], start=1):
                print(f"[QA FOLLOWUP] {mode} | {seq['sequence_id']} | turn {idx}")
                payload = {"query": question, "mode": mode, "session_id": session_id, "top_k": 5}
                ok, status, data, err, latency = post_json(f"{FINAL_API_URL}/query", payload)
                rows.append(build_row(suite="qa_memory_followups", case_id=seq["sequence_id"], turn_index=idx, title=seq["title"], system="final_poc", mode=mode, endpoint=f"{FINAL_API_URL}/query", question=question, ok=ok, status_code=status, data=data, error=err, latency=latency))
    return rows


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 80)
    print("HANS 10-query + follow-up evaluation")
    print(f"FINAL_API_URL    = {FINAL_API_URL}")
    print(f"ORIGINAL_API_URL = {ORIGINAL_API_URL}")
    print(f"Output directory = {OUT_DIR.resolve()}")
    print("=" * 80)

    final_rows = run_final_main_cases()
    original_rows = run_original_main_cases()
    email_followup_rows = run_email_followups()
    qa_followup_rows = run_qa_followups()
    all_rows = final_rows + original_rows + email_followup_rows + qa_followup_rows

    timestamped = [
        write_outputs(final_rows, f"hans_final_10query_4mode_results_{run_id}"),
        write_outputs(original_rows, f"hans_original_10query_results_{run_id}"),
        write_outputs(email_followup_rows, f"hans_email_followup_thread_results_{run_id}"),
        write_outputs(qa_followup_rows, f"hans_qa_followup_memory_results_{run_id}"),
        write_outputs(all_rows, f"hans_combined_10query_original_followups_{run_id}"),
    ]

    write_outputs(final_rows, "hans_final_10query_4mode_results_latest")
    write_outputs(original_rows, "hans_original_10query_results_latest")
    write_outputs(email_followup_rows, "hans_email_followup_thread_results_latest")
    write_outputs(qa_followup_rows, "hans_qa_followup_memory_results_latest")
    write_outputs(all_rows, "hans_combined_10query_original_followups_latest")

    print("\nDONE. Files written:")
    for csv_path, json_path in timestamped:
        print(f"- {csv_path}")
        print(f"- {json_path}")
    print("\nTip: Open the *_latest.csv files first for quick review.")


if __name__ == "__main__":
    main()
