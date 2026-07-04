"""
Mixed scripted evaluation tests for HANS PoC v4 / Email Assistant v5.

Purpose:
- Run a broader test set than the small smoke script.
- Cover English, German, multi-topic email, thread follow-up memory,
  programme catalogue aliases, unknown-programme review, Conversational QA,
  Baseline QA, and a small Email Mistral comparison.

How to use:
1) Start the backend first:
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

2) From the project root, run:
   python run_hans_mixed_version_tests.py

3) Results are written to:
   evaluation/script_mixed_version_test_results_v2.csv
   evaluation/script_mixed_version_test_results_v2.json

Important:
- This script calls the backend API directly. It does not use Streamlit.
- Claude Email Assistant cases are the primary stability checks.
- Mistral cases are marked as comparison cases. They are logged, but not counted
  in the primary pass summary, because Mistral is used for comparison, not as the
  final preferred staff-draft path.
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

CSV_PATH = os.path.join(EVAL_DIR, "script_mixed_version_test_results_v2.csv")
JSON_PATH = os.path.join(EVAL_DIR, "script_mixed_version_test_results_v2.json")

HEADERS = [
    "run_timestamp",
    "case_id",
    "case_group",
    "case_type",
    "count_for_primary_pass",
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
    "expected_checks",
    "check_results",
    "passed_basic_checks",
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
    "final_certificate_submission": {"application_before_graduation", "conditional_enrolment", "final_certificate_submission"},
    "conditional_enrolment": {"application_before_graduation", "conditional_enrolment", "final_certificate_submission"},
    "english_language_requirements": {"english_language_requirements", "language_requirements"},
    "german_language_requirements": {"german_language_requirements", "language_requirements"},
    "language_requirements": {"english_language_requirements", "german_language_requirements", "language_requirements", "language_of_instruction"},
    "language_of_instruction": {"language_of_instruction", "language_requirements"},
    "application_fee": {"application_fee", "fees", "tuition_fees", "semester_contribution"},
    "fees": {"application_fee", "fees", "tuition_fees", "semester_contribution"},
    "tuition_fees": {"tuition_fees", "fees", "application_fee", "semester_contribution"},
    "semester_contribution": {"semester_contribution", "fees"},
    "qualification_recognition": {"qualification_recognition", "admission_requirements"},
    "admission_requirements": {"admission_requirements", "qualification_recognition"},
    "motivation_letter": {"motivation_letter", "required_documents"},
    "application_deadline": {"application_deadline"},
    "study_format": {"study_format"},
    "work_experience": {"work_experience", "admission_requirements"},
    "required_documents": {"required_documents", "document_uploads", "hard_copy_documents", "certified_translations", "motivation_letter"},
    "certified_translations": {"certified_translations", "required_documents"},
    "credit_recognition": {"credit_recognition"},
    "grade_conversion": {"grade_conversion", "admission_requirements"},
    "accommodation": {"accommodation"},
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


def contains_any(text: str, phrases: List[str]) -> bool:
    lower = (text or "").lower()
    return any(p.lower() in lower for p in phrases)


def contains_none(text: str, phrases: List[str]) -> bool:
    lower = (text or "").lower()
    return all(p.lower() not in lower for p in phrases)


def get_review_required(data: Dict[str, Any]) -> bool:
    quality = data.get("quality", {}) or {}
    return bool(quality.get("review_required", False) or data.get("flagged_for_human", False))


def run_check(output: str, check: Dict[str, Any], data: Dict[str, Any]) -> bool:
    kind = check.get("type")
    if kind == "contains_all":
        return contains_all(output, check.get("phrases", []))
    if kind == "contains_any":
        return contains_any(output, check.get("phrases", []))
    if kind == "contains_none":
        return contains_none(output, check.get("phrases", []))
    if kind == "quality_min":
        try:
            return float((data.get("quality") or {}).get("quality_score", 0)) >= float(check.get("value", 0))
        except Exception:
            return False
    if kind == "review_required_is":
        return get_review_required(data) == bool(check.get("value", False))
    if kind == "provider_is":
        return str(data.get("generation_provider", "")).lower() == str(check.get("value", "")).lower()
    if kind == "programme_contains":
        expected = str(check.get("value", "")).lower()
        context = data.get("email_context", {}) or {}
        matched = " ".join([
            str(data.get("matched_programme", "")),
            str(context.get("matched_programme", "")),
            str(context.get("target_program", "")),
            str(context.get("target_programme", "")),
        ]).lower()
        return expected in matched
    if kind == "used_memory_is_true":
        return bool(data.get("used_memory")) is True
    if kind == "used_memory_is_false_or_blank":
        return not bool(data.get("used_memory"))
    if kind == "standalone_contains_all":
        return contains_all(data.get("standalone_query", ""), check.get("phrases", []))
    return True


# ---------------------------------------------------------------------
# Email tests: broad Claude coverage from earlier JSON suites + current fixes
# ---------------------------------------------------------------------
EMAIL_TESTS: List[Dict[str, Any]] = [
    {
        "case_id": "mix_email_001_ib_master_en",
        "case_group": "primary_claude_core",
        "mode": "email_claude",
        "thread_id": "mix_thread_ib_master_en",
        "student_email": "ananya.mix001@example.com",
        "subject": "Application for International Business Master",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nMy name is Ananya and I am completing my Bachelor's degree in Business Administration in India. I would like to apply for the Master's in International Business for the winter semester. Since I will receive my final transcript only in July, can I still apply before graduation? Do I need English language proof and are there application fees?\n\nKind regards,\nAnanya""",
        "expected_programme": "International Business",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_before_graduation", "english_language_requirements", "application_fee"],
        "expected_review_required": "No",
        "checks": [
            {"name": "provider Claude", "type": "provider_is", "value": "anthropic"},
            {"name": "greets Ananya", "type": "contains_all", "phrases": ["Dear Ananya"]},
            {"name": "keeps Master context", "type": "contains_all", "phrases": ["Master", "International Business"]},
            {"name": "fee verification link", "type": "contains_all", "phrases": ["Official uni-assist handling fees"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "mix_email_002_ib_followup_motivation",
        "case_group": "primary_claude_thread_memory",
        "mode": "email_claude",
        "thread_id": "mix_thread_ib_master_en",
        "student_email": "ananya.mix001@example.com",
        "subject": "Application for International Business Master",
        "language": None,
        "email_text": "Thank you for your reply. Is a motivation letter also required?",
        "expected_programme": "International Business",
        "expected_followup_type": "followup_new_topic",
        "expected_topics": ["motivation_letter"],
        "expected_review_required": "No",
        "checks": [
            {"name": "greets Ananya", "type": "contains_all", "phrases": ["Dear Ananya"]},
            {"name": "has motivation letter answer", "type": "contains_all", "phrases": ["motivation letter"]},
            {"name": "does not switch to Bachelor", "type": "contains_none", "phrases": ["Bachelor's programme", "Bachelor programme"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
            {"name": "score at least 70", "type": "quality_min", "value": 70},
        ],
    },
    {
        "case_id": "mix_email_003_mpmd_en_deadline_language_format",
        "case_group": "primary_claude_programme_catalogue",
        "mode": "email_claude",
        "thread_id": "mix_thread_mpmd_en",
        "student_email": "arjun.mix003@example.com",
        "subject": "Project Management and Data Science application",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI have completed my Bachelor's degree in Electronics Engineering from India and I am currently working as a software engineer. I would like to apply for the Master's programme in Project Management and Data Science. Could you please tell me the application deadline, whether the programme is taught entirely in English, and whether it is an on-campus course?\n\nKind regards,\nArjun""",
        "expected_programme": "Project Management and Data Science",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_deadline", "language_of_instruction", "study_format"],
        "expected_review_required": "No",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "Project Management and Data Science"},
            {"name": "answers deadline/period", "type": "contains_any", "phrases": ["deadline", "application period", "application window"]},
            {"name": "answers language", "type": "contains_all", "phrases": ["English"]},
            {"name": "answers format", "type": "contains_any", "phrases": ["on-campus", "campus", "Präsenz"]},
            {"name": "no APS/profile advice", "type": "contains_none", "phrases": ["APS", "work experience is required"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "mix_email_004_mpmd_review_work_docs_english",
        "case_group": "primary_claude_programme_catalogue",
        "mode": "email_claude",
        "thread_id": "mix_thread_mpmd_work_docs",
        "student_email": "neha.mix004@example.com",
        "subject": "MPMD admission requirements",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI am interested in the Project Management and Data Science Master's programme at HTW Berlin. I have a Bachelor's degree in Computer Science and around 10 months of work experience. Could you please tell me whether work experience is required, which documents I should prepare, and whether I need English language proof?\n\nKind regards,\nNeha""",
        "expected_programme": "Project Management and Data Science",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["work_experience", "required_documents", "english_language_requirements"],
        "expected_review_required": "Yes",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "Project Management and Data Science"},
            {"name": "answers work/documents/English", "type": "contains_any", "phrases": ["work experience", "professional experience", "documents", "English"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_005_mpmd_abbreviation_final_certificate",
        "case_group": "primary_claude_programme_catalogue",
        "mode": "email_claude",
        "thread_id": "mix_thread_mpmd_abbrev",
        "student_email": "samir.mix005@example.com",
        "subject": "MPMD application before graduation",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI am in the final semester of my Bachelor's degree and would like to apply for MPMD. I will receive my final certificate after the application deadline. Can I still apply, and do I need to submit the final certificate later?\n\nKind regards,\nSamir""",
        "expected_programme": "Project Management and Data Science",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_before_graduation", "final_certificate_submission"],
        "expected_review_required": "No",
        "checks": [
            {"name": "MPMD resolved", "type": "programme_contains", "value": "Project Management and Data Science"},
            {"name": "answers final certificate", "type": "contains_any", "phrases": ["final certificate", "final transcript", "after", "later"]},
            {"name": "score at least 70", "type": "quality_min", "value": 70},
        ],
    },
    {
        "case_id": "mix_email_006_proitd_route_fee_language",
        "case_group": "primary_claude_programme_catalogue",
        "mode": "email_claude",
        "thread_id": "mix_thread_proitd_route",
        "student_email": "lina.mix006@example.com",
        "subject": "Professional IT and Digitalization application",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI am interested in the Professional IT and Digitalization Master's programme. I completed my Bachelor's degree outside Germany. Could you please tell me whether I need to apply through uni-assist, whether there are application fees, and what language proof is required?\n\nKind regards,\nLina""",
        "expected_programme": "Professional IT Business and Digitalization",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "application_fee", "language_requirements"],
        "expected_review_required": "No",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "Professional IT"},
            {"name": "answers route/fee/language", "type": "contains_any", "phrases": ["uni-assist", "application fee", "language proof", "English"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_007_proitd_abbreviation_admission_deadline",
        "case_group": "primary_claude_programme_catalogue",
        "mode": "email_claude",
        "thread_id": "mix_thread_proitd_abbrev",
        "student_email": "sara.mix007@example.com",
        "subject": "PROITD admission",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI would like to apply for PROITD at HTW Berlin. Could you please tell me the admission requirements, the application deadline, and whether English proof is needed?\n\nKind regards,\nSara""",
        "expected_programme": "Professional IT Business and Digitalization",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["admission_requirements", "application_deadline", "english_language_requirements"],
        "expected_review_required": "No",
        "checks": [
            {"name": "PROITD resolved", "type": "programme_contains", "value": "Professional IT"},
            {"name": "answers admission/deadline/English", "type": "contains_any", "phrases": ["admission", "deadline", "English"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_008_csb_en_eu_ib_motivation",
        "case_group": "primary_claude_route_safety",
        "mode": "email_claude",
        "thread_id": "mix_thread_csb_en",
        "student_email": "lucas.mix008@example.com",
        "subject": "Cybersecurity and Business Bachelor application",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI am an EU citizen living in Portugal and I am finishing high school with an International Baccalaureate diploma. I want to apply for the Cybersecurity and Business Bachelor's programme. Do I apply through uni-assist, does my IB diploma meet the admission requirements, and is a motivation letter required?\n\nKind regards,\nLucas""",
        "expected_programme": "Cyber Security and Business",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "qualification_recognition", "motivation_letter"],
        "expected_review_required": "No",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "Cyber Security and Business"},
            {"name": "EU route", "type": "contains_all", "phrases": ["EU", "HTW"]},
            {"name": "does not require uni-assist as main route", "type": "contains_none", "phrases": ["you must apply through uni-assist", "you need to apply through uni-assist because", "apply through uni-assist as the main route"]},
            {"name": "no unsupported English exemption", "type": "contains_none", "phrases": ["additional English language proof is not required"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "mix_email_009_conrem_docs_english_semester",
        "case_group": "primary_claude_programme_catalogue",
        "mode": "email_claude",
        "thread_id": "mix_thread_conrem",
        "student_email": "maria.mix009@example.com",
        "subject": "Construction and Real Estate Management",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI am interested in Construction and Real Estate Management at HTW Berlin. Could you please tell me whether English language proof is required, which application documents I should prepare, and whether I need to pay a semester contribution?\n\nKind regards,\nMaria""",
        "expected_programme": "Construction and Real Estate Management",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["english_language_requirements", "required_documents", "semester_contribution"],
        "expected_review_required": "No",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "Construction and Real Estate"},
            {"name": "answers English/documents/semester", "type": "contains_any", "phrases": ["English", "documents", "semester"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_010_unknown_programme_review",
        "case_group": "primary_claude_review_guardrail",
        "mode": "email_claude",
        "thread_id": "mix_thread_unknown_programme",
        "student_email": "unknown.mix010@example.com",
        "subject": "Unknown programme application",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI would like to apply for the Master's programme in Space Robotics and Ocean Finance at HTW Berlin. Could you please tell me the admission requirements, the application deadline, and whether English proof is required?\n\nKind regards,\nTest Applicant""",
        "expected_programme": "",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["admission_requirements", "application_deadline", "english_language_requirements"],
        "expected_review_required": "Yes",
        "checks": [
            {"name": "should require review", "type": "review_required_is", "value": True},
            {"name": "does not hallucinate programme certainty", "type": "contains_none", "phrases": ["Space Robotics and Ocean Finance at HTW Berlin requires"]},
        ],
    },
    {
        "case_id": "mix_email_011_dual_french_moroccan_business",
        "case_group": "primary_claude_route_safety",
        "mode": "email_claude",
        "thread_id": "mix_thread_dual_business",
        "student_email": "dual.mix011@example.com",
        "subject": "Business Bachelor application route and French Baccalaureate recognition",
        "language": None,
        "email_text": "Dear Admissions Team, I am a dual citizen (French and Moroccan), currently residing in Morocco and holding a French Baccalauréat diploma. I want to apply for the English-taught Bachelor’s in Business but am unsure whether I should apply as an EU applicant or international. Also, would my French diploma fulfill the general university entrance requirements, or would I need to obtain a VPD from Uni-Assist?",
        "expected_programme": "",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "qualification_recognition"],
        "expected_review_required": "No",
        "checks": [
            {"name": "French/EU citizenship handled", "type": "contains_all", "phrases": ["French", "EU"]},
            {"name": "Morocco does not force non-EU route", "type": "contains_none", "phrases": ["because you live in Morocco, you must apply through uni-assist"]},
            {"name": "qualification recognition separate", "type": "contains_any", "phrases": ["French Baccala", "qualification", "recognition", "checked"]},
        ],
    },
    {
        "case_id": "mix_email_012_design_translations_language",
        "case_group": "primary_claude_multitopic_diverse",
        "mode": "email_claude",
        "thread_id": "mix_thread_design_translations",
        "student_email": "design.mix012@example.com",
        "subject": "Design and Culture Bachelor international application enquiry",
        "language": None,
        "email_text": "Dear Admissions Office, I completed secondary school outside Germany and want to apply for a Design and Culture Bachelor's programme. Do I need to apply through uni-assist, do my documents need certified translations, and what language proof is needed?",
        "expected_programme": "",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "certified_translations", "language_requirements"],
        "expected_review_required": "Yes",
        "checks": [
            {"name": "answers route/translations/language", "type": "contains_any", "phrases": ["uni-assist", "certified translation", "language proof"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_013_computer_science_deadline_english_accommodation",
        "case_group": "primary_claude_multitopic_diverse",
        "mode": "email_claude",
        "thread_id": "mix_thread_cs_accommodation",
        "student_email": "cs.mix013@example.com",
        "subject": "Computer Science Master application and accommodation enquiry",
        "language": None,
        "email_text": "Dear Team, I would like to apply for a Computer Science Master's programme. Could you please tell me the application period, whether English language proof is required, and whether HTW Berlin provides accommodation support?",
        "expected_programme": "",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_deadline", "english_language_requirements", "accommodation"],
        "expected_review_required": "No",
        "checks": [
            {"name": "answers deadline/English/accommodation", "type": "contains_any", "phrases": ["application", "English", "accommodation", "housing"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_014_ib_transfer_credits_docs",
        "case_group": "primary_claude_multitopic_diverse",
        "mode": "email_claude",
        "thread_id": "mix_thread_ib_transfer",
        "student_email": "transfer.mix014@example.com",
        "subject": "International Business Bachelor transfer and credit recognition",
        "language": None,
        "email_text": "Hello, I am interested in transferring into your Bachelor’s program in International Business from an institution in South Korea, where I’ve completed three semesters of Business Administration. I would like to understand whether any of my previously earned ECTS-equivalent credits could be transferred, and what documentation is needed to have these credits officially evaluated by your admissions office.",
        "expected_programme": "International Business",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["credit_recognition", "required_documents"],
        "expected_review_required": "No",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "International Business"},
            {"name": "answers credit recognition/documents", "type": "contains_any", "phrases": ["credits", "recognised", "recognized", "documents", "documentation"]},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_015_mpmd_de_deadline_language_format",
        "case_group": "primary_claude_german",
        "mode": "email_claude",
        "thread_id": "mix_thread_mpmd_de",
        "student_email": "arjun.mix015@example.com",
        "subject": "Bewerbung Project Management and Data Science",
        "language": None,
        "email_text": """Sehr geehrtes Zulassungsteam,\n\nich habe meinen Bachelorabschluss in Electronics Engineering in Indien abgeschlossen und arbeite derzeit als Softwareentwickler. Ich möchte mich für den Masterstudiengang Project Management and Data Science bewerben. Können Sie mir bitte mitteilen, wann die Bewerbungsfrist ist, ob der Studiengang vollständig auf Englisch unterrichtet wird und ob es sich um ein Präsenzstudium handelt?\n\nMit freundlichen Grüßen\nArjun""",
        "expected_programme": "Project Management and Data Science",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_deadline", "language_of_instruction", "study_format"],
        "expected_review_required": "No",
        "checks": [
            {"name": "German reply", "type": "contains_all", "phrases": ["Mit freundlichen Grüßen"]},
            {"name": "deadline/language/format", "type": "contains_all", "phrases": ["Bewerbungsfrist", "englisch", "Präsenz"]},
            {"name": "no APS profile advice", "type": "contains_none", "phrases": ["APS", "Berufserfahrung"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "mix_email_016_ib_de_pending_english_fee",
        "case_group": "primary_claude_german",
        "mode": "email_claude",
        "thread_id": "mix_thread_ib_de",
        "student_email": "ananya.mix016@example.com",
        "subject": "Bewerbung International Business Master",
        "language": None,
        "email_text": """Sehr geehrtes Zulassungsteam,\n\nmein Name ist Ananya und ich schließe derzeit meinen Bachelor in Betriebswirtschaftslehre in Indien ab. Ich möchte mich für den Masterstudiengang International Business zum Wintersemester bewerben. Da ich mein endgültiges Zeugnis erst im Juli erhalte, kann ich mich trotzdem vor dem Abschluss bewerben? Brauche ich einen Nachweis über Englischkenntnisse und fallen Bewerbungsgebühren an?\n\nMit freundlichen Grüßen\nAnanya""",
        "expected_programme": "International Business",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_before_graduation", "english_language_requirements", "application_fee"],
        "expected_review_required": "No",
        "checks": [
            {"name": "greets Ananya", "type": "contains_all", "phrases": ["Ananya"]},
            {"name": "German reply", "type": "contains_all", "phrases": ["Mit freundlichen Grüßen"]},
            {"name": "answers fees", "type": "contains_all", "phrases": ["Bewerbungsgeb"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "mix_email_017_csb_de_eu_ib_motivation",
        "case_group": "primary_claude_german",
        "mode": "email_claude",
        "thread_id": "mix_thread_csb_de",
        "student_email": "csb.mix017@example.com",
        "subject": "Bewerbung Cyber Security and Business",
        "language": None,
        "email_text": """Sehr geehrtes Zulassungsteam,\n\nich bin EU-Bürger und lebe in Portugal. Ich mache derzeit mein International Baccalaureate Diploma und möchte mich für den Bachelorstudiengang Cyber Security and Business bewerben. Muss ich mich über uni-assist bewerben, erfüllt mein IB-Diplom die Zulassungsvoraussetzungen und brauche ich ein Motivationsschreiben?\n\nMit freundlichen Grüßen""",
        "expected_programme": "Cyber Security and Business",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "qualification_recognition", "admission_requirements", "motivation_letter"],
        "expected_review_required": "No",
        "checks": [
            {"name": "German reply", "type": "contains_all", "phrases": ["Mit freundlichen Grüßen"]},
            {"name": "EU route", "type": "contains_all", "phrases": ["EU", "Hochschulstart"]},
            {"name": "not uni-assist main route", "type": "contains_none", "phrases": ["müssen Sie sich über uni-assist bewerben", "apply through uni-assist"]},
            {"name": "score at least 80", "type": "quality_min", "value": 80},
        ],
    },
    {
        "case_id": "mix_email_018_csb_followup_clarification_turn1",
        "case_group": "primary_claude_thread_memory",
        "mode": "email_claude",
        "thread_id": "mix_thread_csb_clarification",
        "student_email": "leo.mix018@example.com",
        "subject": "Cybersecurity and Business Bachelor application",
        "language": None,
        "email_text": """Hello,\n\nI am an EU citizen living in Portugal and I am finishing high school with an International Baccalaureate diploma. I want to apply for the Cybersecurity and Business Bachelor's programme. Do I apply through uni-assist, does my IB diploma meet the admission requirements, and is a motivation letter required?\n\nKind regards,\nLeo""",
        "expected_programme": "Cyber Security and Business",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["application_route", "qualification_recognition", "motivation_letter"],
        "expected_review_required": "No",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "Cyber Security and Business"},
            {"name": "EU route", "type": "contains_all", "phrases": ["EU", "HTW"]},
            {"name": "no unsupported English exemption", "type": "contains_none", "phrases": ["additional English language proof is not required"]},
        ],
    },
    {
        "case_id": "mix_email_019_csb_followup_clarification_turn2",
        "case_group": "primary_claude_thread_memory",
        "mode": "email_claude",
        "thread_id": "mix_thread_csb_clarification",
        "student_email": "leo.mix018@example.com",
        "subject": "Cybersecurity and Business Bachelor application",
        "language": None,
        "email_text": "Thank you for your reply, but I still do not understand the application route. Your answer was not clear to me.",
        "expected_programme": "Cyber Security and Business",
        "expected_followup_type": "clarification_or_complaint",
        "expected_topics": [],
        "expected_review_required": "Yes",
        "checks": [
            {"name": "should require review", "type": "review_required_is", "value": True},
            {"name": "mentions previous/review", "type": "contains_any", "phrases": ["previous", "review", "clarification", "check"]},
        ],
    },
    {
        "case_id": "mix_email_020_conrem_followup_turn1",
        "case_group": "primary_claude_thread_memory",
        "mode": "email_claude",
        "thread_id": "mix_thread_conrem_followup",
        "student_email": "maria.mix020@example.com",
        "subject": "Construction and Real Estate Management application",
        "language": None,
        "email_text": """Dear Admissions Team,\n\nI am interested in Construction and Real Estate Management at HTW Berlin. Could you please tell me whether English language proof is required, which application documents I should prepare, and whether I need to pay a semester contribution?\n\nKind regards,\nMaria""",
        "expected_programme": "Construction and Real Estate Management",
        "expected_followup_type": "new_enquiry",
        "expected_topics": ["english_language_requirements", "required_documents", "semester_contribution"],
        "expected_review_required": "No",
        "checks": [
            {"name": "programme match", "type": "programme_contains", "value": "Construction and Real Estate"},
            {"name": "has citation", "type": "contains_all", "phrases": ["[Doc"]},
        ],
    },
    {
        "case_id": "mix_email_021_conrem_followup_deadline_turn2",
        "case_group": "primary_claude_thread_memory",
        "mode": "email_claude",
        "thread_id": "mix_thread_conrem_followup",
        "student_email": "maria.mix020@example.com",
        "subject": "Construction and Real Estate Management application",
        "language": None,
        "email_text": "And what about the application deadline?",
        "expected_programme": "Construction and Real Estate Management",
        "expected_followup_type": "followup_new_topic",
        "expected_topics": ["application_deadline"],
        "expected_review_required": "No",
        "checks": [
            {"name": "keeps programme context", "type": "programme_contains", "value": "Construction and Real Estate"},
            {"name": "answers deadline", "type": "contains_any", "phrases": ["deadline", "application period", "Bewerbungsfrist"]},
        ],
    },
]

# Comparison-only Mistral cases: useful for thesis comparison, but not counted in primary pass.
MISTRAL_COMPARISON_TESTS: List[Dict[str, Any]] = []
for base in [EMAIL_TESTS[0], EMAIL_TESTS[7], EMAIL_TESTS[14]]:
    t = dict(base)
    t["case_id"] = t["case_id"].replace("mix_email", "mix_mistral")
    t["case_group"] = "comparison_mistral_email"
    t["mode"] = "email_mistral"
    t["thread_id"] = t["thread_id"].replace("mix_thread", "mix_mistral_thread")
    t["count_for_primary_pass"] = False
    # Keep checks light for comparison; the output is mostly inspected by metrics.
    t["checks"] = [
        {"name": "provider Mistral", "type": "provider_is", "value": "mistral"},
        {"name": "has some cited answer", "type": "contains_all", "phrases": ["[Doc"]},
    ]
    MISTRAL_COMPARISON_TESTS.append(t)

EMAIL_TESTS.extend(MISTRAL_COMPARISON_TESTS)

# ---------------------------------------------------------------------
# QA tests: Conversational vs Baseline
# ---------------------------------------------------------------------
QA_TESTS: List[Dict[str, Any]] = [
    {
        "case_id": "mix_qa_001_conv_mpmd_first",
        "case_group": "primary_conversational_qa",
        "mode": "conversational_qa",
        "session_key": "conv_mpmd",
        "language": None,
        "query": "Ich interessiere mich für den Masterstudiengang Project Management and Data Science an der HTW Berlin. Wo finde ich die Zulassungsvoraussetzungen?",
        "expected_topics": ["admission_requirements"],
        "checks": [
            {"name": "German answer", "type": "contains_any", "phrases": ["Zulassung", "Voraussetzungen"]},
            {"name": "mentions programme", "type": "contains_all", "phrases": ["Project Management and Data Science"]},
        ],
    },
    {
        "case_id": "mix_qa_002_conv_mpmd_followup_language",
        "case_group": "primary_conversational_qa",
        "mode": "conversational_qa",
        "session_key": "conv_mpmd",
        "language": None,
        "query": "Wird er auf Englisch oder Deutsch unterrichtet?",
        "expected_topics": ["language_of_instruction"],
        "checks": [
            {"name": "uses memory", "type": "used_memory_is_true"},
            {"name": "standalone mentions programme", "type": "standalone_contains_all", "phrases": ["Project Management and Data Science"]},
            {"name": "answers English", "type": "contains_any", "phrases": ["Englisch", "English"]},
        ],
    },
    {
        "case_id": "mix_qa_003_conv_conrem_first",
        "case_group": "primary_conversational_qa",
        "mode": "conversational_qa",
        "session_key": "conv_conrem",
        "language": None,
        "query": "I am interested in Construction and Real Estate Management at HTW Berlin. Where can I find the admission requirements?",
        "expected_topics": ["admission_requirements"],
        "checks": [
            {"name": "mentions programme", "type": "contains_all", "phrases": ["Construction and Real Estate Management"]},
        ],
    },
    {
        "case_id": "mix_qa_004_conv_conrem_followup_deadline",
        "case_group": "primary_conversational_qa",
        "mode": "conversational_qa",
        "session_key": "conv_conrem",
        "language": None,
        "query": "What is the application deadline for it?",
        "expected_topics": ["application_deadline"],
        "checks": [
            {"name": "uses memory", "type": "used_memory_is_true"},
            {"name": "standalone mentions programme", "type": "standalone_contains_all", "phrases": ["Construction and Real Estate Management"]},
            {"name": "answers deadline", "type": "contains_any", "phrases": ["deadline", "application period"]},
        ],
    },
    {
        "case_id": "mix_qa_005_baseline_mpmd_first",
        "case_group": "comparison_baseline_qa",
        "mode": "baseline_qa",
        "session_key": "base_mpmd",
        "language": None,
        "query": "Ich interessiere mich für den Masterstudiengang Project Management and Data Science an der HTW Berlin. Wo finde ich die Zulassungsvoraussetzungen?",
        "expected_topics": ["admission_requirements"],
        "count_for_primary_pass": False,
        "checks": [
            {"name": "baseline answers first turn", "type": "contains_any", "phrases": ["Zulassung", "Voraussetzungen", "admission"]},
        ],
    },
    {
        "case_id": "mix_qa_006_baseline_followup_no_memory",
        "case_group": "comparison_baseline_qa",
        "mode": "baseline_qa",
        "session_key": "base_mpmd",
        "language": None,
        "query": "Wird er auf Englisch oder Deutsch unterrichtet?",
        "expected_topics": ["language_of_instruction"],
        "count_for_primary_pass": False,
        "checks": [
            {"name": "baseline should not report memory", "type": "used_memory_is_false_or_blank"},
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
    resp = requests.post(f"{API_URL}/email", json=payload, timeout=240)
    elapsed = time.time() - start
    if resp.ok:
        data = resp.json()
        error = ""
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
    resp = requests.post(f"{API_URL}/query", json=payload, timeout=180)
    elapsed = time.time() - start
    if resp.ok:
        data = resp.json()
        if data.get("session_id"):
            sessions[session_key] = data.get("session_id")
        error = ""
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
    context = data.get("email_context", {}) or {}

    check_results = []
    for check in test.get("checks", []):
        passed = run_check(output, check, data)
        check_results.append({"name": check.get("name", "check"), "passed": passed})

    expected_review = test.get("expected_review_required", "")
    review_match = True
    if expected_review in {"Yes", "No"}:
        review_match = ("Yes" if get_review_required(data) else "No") == expected_review

    # Comparison cases still compute pass/fail, but are not included in primary summary.
    passed_basic = (
        resp.ok
        and (not test.get("expected_followup_type") or test.get("expected_followup_type") == (data.get("followup_type", "") if data else ""))
        and not metrics["missing"]
        and review_match
        and all(item["passed"] for item in check_results)
    )

    return {
        "run_timestamp": datetime.now().isoformat(),
        "case_id": test["case_id"],
        "case_group": test.get("case_group", "email"),
        "case_type": "email",
        "count_for_primary_pass": str(test.get("count_for_primary_pass", True)),
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
        "expected_programme": test.get("expected_programme", ""),
        "matched_programme": data.get("matched_programme", "") or context.get("matched_programme", "") or context.get("target_program", ""),
        "expected_followup_type": test.get("expected_followup_type", ""),
        "actual_followup_type": data.get("followup_type", ""),
        "expected_topics": ", ".join(test.get("expected_topics", [])),
        "detected_topics": ", ".join(detected),
        "missing_topics": ", ".join(metrics["missing"]),
        "extra_topics": ", ".join(metrics["extra"]),
        "topic_coverage_percent": metrics["coverage"],
        "expected_review_required": expected_review,
        "review_required": "Yes" if get_review_required(data) else "No",
        "review_reason": quality.get("review_reason", ""),
        "expected_checks": json.dumps(test.get("checks", []), ensure_ascii=False),
        "check_results": json.dumps(check_results, ensure_ascii=False),
        "passed_basic_checks": "Yes" if passed_basic else "No",
        "quality_label": quality.get("quality_label", ""),
        "quality_score": quality.get("quality_score", ""),
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
        "case_group": test.get("case_group", "qa"),
        "case_type": "qa",
        "count_for_primary_pass": str(test.get("count_for_primary_pass", True)),
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
        "expected_programme": "",
        "matched_programme": "",
        "expected_followup_type": "",
        "actual_followup_type": "",
        "expected_topics": ", ".join(test.get("expected_topics", [])),
        "detected_topics": "",
        "missing_topics": "",
        "extra_topics": "",
        "topic_coverage_percent": "manual_QA",
        "expected_review_required": "",
        "review_required": "not_applicable",
        "review_reason": "QA mode has no email review logic",
        "expected_checks": json.dumps(test.get("checks", []), ensure_ascii=False),
        "check_results": json.dumps(check_results, ensure_ascii=False),
        "passed_basic_checks": "Yes" if passed_basic else "No",
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


def main() -> None:
    print(f"Using backend: {API_URL}")
    rows: List[Dict[str, Any]] = []

    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Health check: {health_resp.status_code}")
    except Exception as exc:
        print("Could not reach backend. Start uvicorn first.")
        print(exc)
        return

    print("\nRunning mixed Email Assistant tests...")
    for test in EMAIL_TESTS:
        print(f"- {test['case_id']} [{test['mode']}]")
        try:
            result = call_email(test)
            row = row_for_email(test, result)
        except Exception as exc:
            row = {h: "" for h in HEADERS}
            row.update({
                "run_timestamp": datetime.now().isoformat(),
                "case_id": test["case_id"],
                "case_group": test.get("case_group", "email"),
                "case_type": "email",
                "count_for_primary_pass": str(test.get("count_for_primary_pass", True)),
                "backend_endpoint": "/email",
                "backend_mode": test["mode"],
                "thread_id": test.get("thread_id", ""),
                "input_text": test.get("email_text", ""),
                "http_status": 0,
                "error": str(exc),
                "passed_basic_checks": "No",
            })
        rows.append(row)
        print(
            f"  pass={row['passed_basic_checks']} | primary={row['count_for_primary_pass']} | "
            f"score={row.get('quality_score')} | review={row.get('review_required')} | topics={row.get('detected_topics')}"
        )

    print("\nRunning QA comparison tests...")
    sessions: Dict[str, Optional[str]] = {}
    for test in QA_TESTS:
        print(f"- {test['case_id']} [{test['mode']}]")
        try:
            result = call_query(test, sessions)
            row = row_for_query(test, result)
        except Exception as exc:
            row = {h: "" for h in HEADERS}
            row.update({
                "run_timestamp": datetime.now().isoformat(),
                "case_id": test["case_id"],
                "case_group": test.get("case_group", "qa"),
                "case_type": "qa",
                "count_for_primary_pass": str(test.get("count_for_primary_pass", True)),
                "backend_endpoint": "/query",
                "backend_mode": test["mode"],
                "input_text": test.get("query", ""),
                "http_status": 0,
                "error": str(exc),
                "passed_basic_checks": "No",
            })
        rows.append(row)
        print(
            f"  pass={row['passed_basic_checks']} | primary={row['count_for_primary_pass']} | "
            f"used_memory={row.get('used_memory')} | grounded={row.get('is_grounded')}"
        )

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    total_all = len(rows)
    passed_all = sum(1 for r in rows if r.get("passed_basic_checks") == "Yes")
    primary_rows = [r for r in rows if str(r.get("count_for_primary_pass", "True")).lower() == "true"]
    primary_total = len(primary_rows)
    primary_passed = sum(1 for r in primary_rows if r.get("passed_basic_checks") == "Yes")

    print("\nDone.")
    print(f"Primary passed basic checks: {primary_passed}/{primary_total}")
    print(f"All rows passed basic checks: {passed_all}/{total_all}")
    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")

    if primary_passed != primary_total:
        print("\nSome PRIMARY checks failed. Open the CSV and inspect check_results + output_text.")
    if passed_all != total_all:
        print("Some comparison or primary checks failed. This may be useful for thesis comparison, not necessarily a release blocker.")


if __name__ == "__main__":
    main()
