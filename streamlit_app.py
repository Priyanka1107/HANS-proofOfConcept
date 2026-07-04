# streamlit_app.py
"""
Streamlit UI for HANS PoC.

This UI is aligned with the current four-mode evaluation setup:

1. Email Assistant – Claude      -> /email with mode=email_claude
2. Email Assistant – Mistral     -> /email with mode=email_mistral
3. Conversational QA             -> /query with mode=conversational_qa
4. Baseline QA                   -> /query with mode=baseline_qa

Every UI request is logged automatically under the evaluation folder:
- evaluation/hans_ui_all_test_results.csv

For compatibility with earlier Email Assistant tests, email runs are also saved to:
- evaluation/email_assistant_v5_ui_results.csv
- evaluation/email_thread_v5_ui_results.csv
"""

from __future__ import annotations

import os
import time
import csv
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("BACKEND_URL", "http://localhost:8001")


# ---------------------------------------------------------------------
# UI CSV logging
# ---------------------------------------------------------------------
EVAL_DIR = os.path.join(os.path.dirname(__file__), "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# New unified UI log: every UI request goes here.
UI_ALL_TESTS_CSV = os.path.join(EVAL_DIR, "hans_ui_all_test_results.csv")

# Legacy email logs kept for comparison with earlier versions.
UI_EMAIL_ASSISTANT_CSV = os.path.join(EVAL_DIR, "email_assistant_v5_ui_results.csv")
UI_EMAIL_THREAD_CSV = os.path.join(EVAL_DIR, "email_thread_v5_ui_results.csv")


UI_ALL_HEADERS = [
    "run_timestamp",
    "version_label",
    "comment",
    "ui_mode",
    "backend_endpoint",
    "backend_mode",
    "generation_provider",
    "generation_model",
    "case_id",
    "title",
    "turn_index",
    "input_text",
    "student_email",
    "subject",
    "thread_id",
    "sent_session_id",
    "session_id",
    "used_memory",
    "standalone_query",
    "detected_intent",
    "enhanced_query",
    "is_followup",
    "expected_topics",
    "detected_topics",
    "missing_topics",
    "extra_topics",
    "topic_coverage_percent",
    "expected_followup_type",
    "actual_followup_type",
    "review_required",
    "review_reason",
    "quality_label",
    "quality_score",
    "is_grounded",
    "grounding_confidence",
    "citation_count",
    "citations",
    "source_count",
    "response_time_seconds",
    "http_status",
    "error",
    "output_text",
]

EMAIL_ASSISTANT_HEADERS = [
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
    "generation_provider",
    "generation_model",
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

EMAIL_THREAD_HEADERS = [
    "run_timestamp",
    "version_label",
    "comment",
    "case_id",
    "turn_index",
    "student_email",
    "thread_id",
    "subject",
    "email_text",
    "expected_followup_type",
    "actual_followup_type",
    "review_required",
    "quality_label",
    "detected_topics",
    "staff_draft",
    "generation_provider",
    "generation_model",
    "response_time_seconds",
    "http_status",
    "error",
]


TOPIC_EQUIVALENTS = {
    "language_requirements": {
        "language_requirements",
        "english_language_requirements",
        "german_language_requirements",
    },
    "english_language_requirements": {
        "language_requirements",
        "english_language_requirements",
    },
    "required_documents": {
        "required_documents",
        "document_uploads",
        "certified_translations",
        "official_transcripts",
        "hard_copy_documents",
        "final_certificate_submission",
        "motivation_letter",
    },
    "fees": {
        "fees",
        "tuition_fees",
        "application_fee",
        "semester_contribution",
    },
    "application_fee": {
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
    "application_before_graduation": {
        "final_certificate_submission",
        "application_before_graduation",
        "conditional_enrolment",
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


UNCERTAINTY_PHRASES = [
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


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
def _append_csv_row(path: str, headers: list[str], row: dict) -> None:
    """Append one UI test result row."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _split_expected_topics(raw: str) -> list[str]:
    """Convert comma-separated UI input into a clean topic list."""
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _topic_metrics(expected_topics: list[str], detected_topics: list[str]) -> dict:
    expected_set = set(expected_topics or [])
    detected_set = set(detected_topics or [])

    if not expected_set:
        return {
            "missing_topics": "",
            "extra_topics": "",
            "topic_coverage_percent": "not_applicable",
        }

    covered = []
    for expected_topic in sorted(expected_set):
        equivalent_topics = TOPIC_EQUIVALENTS.get(expected_topic, {expected_topic})
        if equivalent_topics & detected_set:
            covered.append(expected_topic)

    missing = [t for t in sorted(expected_set) if t not in covered]

    extra = []
    for detected_topic in detected_set:
        matched_expected = False
        for expected_topic in expected_set:
            equivalent_topics = TOPIC_EQUIVALENTS.get(expected_topic, {expected_topic})
            if detected_topic in equivalent_topics:
                matched_expected = True
                break
        if not matched_expected and detected_topic not in expected_set:
            extra.append(detected_topic)

    return {
        "missing_topics": ", ".join(missing),
        "extra_topics": ", ".join(sorted(extra)),
        "topic_coverage_percent": round((len(covered) / len(expected_set)) * 100, 2),
    }


def _detect_topics_from_email_response(data: dict) -> list[str]:
    return [
        t.get("topic_id", "")
        for t in data.get("detected_topics", [])
        if t.get("topic_id")
    ]


def _normalise_citations(citations) -> list[str]:
    if not citations:
        return []
    if isinstance(citations, list):
        return [str(c) for c in citations]
    return [str(citations)]


def _build_all_ui_row(
    *,
    run_timestamp: str,
    version_label: str,
    comment: str,
    ui_mode: str,
    backend_endpoint: str,
    backend_mode: str,
    generation_provider: str = "",
    generation_model: str = "",
    case_id: str,
    title: str,
    turn_index: str,
    input_text: str,
    student_email: str,
    subject: str,
    thread_id: str,
    expected_topics: list[str],
    detected_topics: list[str],
    expected_followup_type: str,
    actual_followup_type: str,
    review_required: str,
    review_reason: str,
    quality_label: str,
    quality_score,
    is_grounded,
    grounding_confidence,
    citation_count,
    citations: list[str],
    source_count,
    elapsed: float,
    http_status: int,
    error: str,
    output_text: str,
    sent_session_id: str = "",
    session_id: str = "",
    used_memory: str = "",
    standalone_query: str = "",
    detected_intent: str = "",
    enhanced_query: str = "",
    is_followup: str = "",
) -> dict:
    metrics = _topic_metrics(expected_topics, detected_topics)

    return {
        "run_timestamp": run_timestamp,
        "version_label": version_label,
        "comment": comment,
        "ui_mode": ui_mode,
        "backend_endpoint": backend_endpoint,
        "backend_mode": backend_mode,
        "generation_provider": generation_provider,
        "generation_model": generation_model,
        "case_id": case_id,
        "title": title,
        "turn_index": turn_index,
        "input_text": input_text,
        "student_email": student_email,
        "subject": subject,
        "thread_id": thread_id,
        "sent_session_id": sent_session_id,
        "session_id": session_id,
        "used_memory": used_memory,
        "standalone_query": standalone_query,
        "detected_intent": detected_intent,
        "enhanced_query": enhanced_query,
        "is_followup": is_followup,
        "expected_topics": ", ".join(expected_topics),
        "detected_topics": ", ".join(detected_topics),
        "missing_topics": metrics["missing_topics"],
        "extra_topics": metrics["extra_topics"],
        "topic_coverage_percent": metrics["topic_coverage_percent"],
        "expected_followup_type": expected_followup_type,
        "actual_followup_type": actual_followup_type,
        "review_required": review_required,
        "review_reason": review_reason,
        "quality_label": quality_label,
        "quality_score": quality_score,
        "is_grounded": is_grounded,
        "grounding_confidence": grounding_confidence,
        "citation_count": citation_count,
        "citations": ", ".join(citations),
        "source_count": source_count,
        "response_time_seconds": round(elapsed, 2),
        "http_status": http_status,
        "error": error,
        "output_text": output_text,
    }


def _build_ui_email_assistant_row(
    *,
    data: dict,
    run_timestamp: str,
    version_label: str,
    comment: str,
    case_id: str,
    title: str,
    email_text: str,
    student_email: str,
    subject: str,
    thread_id: str,
    expected_topics: list[str],
    elapsed: float,
    http_status: int,
    error: str = "",
) -> dict:
    detected_topics = _detect_topics_from_email_response(data)
    metrics = _topic_metrics(expected_topics, detected_topics)

    quality = data.get("quality", {}) or {}
    validation = data.get("validation", {}) or {}
    citations = _normalise_citations(data.get("citations", []) or quality.get("citations", []))
    sources = data.get("sources", []) or []
    staff_draft = data.get("staff_draft", "") or ""

    review_required = bool(quality.get("review_required", False))
    review_reason = quality.get("review_reason", "") or ""

    if metrics["missing_topics"]:
        review_required = True
        review_reason = (review_reason + "; " if review_reason else "") + "Missing expected topic(s): " + metrics["missing_topics"]

    if metrics["extra_topics"]:
        review_reason = (review_reason + "; " if review_reason else "") + "Extra detected topic(s): " + metrics["extra_topics"]

    lower_draft = staff_draft.lower()
    if any(p in lower_draft for p in UNCERTAINTY_PHRASES):
        review_required = True
        review_reason = (review_reason + "; " if review_reason else "") + "Draft contains uncertainty or needs staff check"

    base_score = quality.get("quality_score", 100)
    try:
        adjusted_score = float(base_score)
    except Exception:
        adjusted_score = 100.0

    if metrics["missing_topics"]:
        adjusted_score -= 20
    if any(p in lower_draft for p in UNCERTAINTY_PHRASES):
        adjusted_score -= 15
    if validation and validation.get("is_grounded") is False:
        adjusted_score -= 20
    if not citations:
        adjusted_score -= 20

    adjusted_score = max(0, min(100, round(adjusted_score, 2)))

    if review_required:
        adjusted_label = "partial" if adjusted_score >= 75 else "review"
    else:
        adjusted_label = "good" if adjusted_score >= 80 else "partial"

    return {
        "run_timestamp": run_timestamp,
        "version_label": version_label,
        "comment": comment,
        "case_id": case_id,
        "title": title,
        "full_email": email_text,
        "student_email": student_email,
        "subject": subject,
        "thread_id": data.get("thread_id") or thread_id,
        "followup_type": data.get("followup_type", ""),
        "generation_provider": data.get("generation_provider", ""),
        "generation_model": data.get("generation_model", ""),
        "expected_topic_count": len(expected_topics),
        "expected_topics": ", ".join(expected_topics),
        "detected_topic_count": len(detected_topics),
        "detected_topics": ", ".join(detected_topics),
        "missing_topics": metrics["missing_topics"],
        "extra_topics": metrics["extra_topics"],
        "topic_coverage_percent": metrics["topic_coverage_percent"],
        "response_time_seconds": round(elapsed, 2),
        "quality_score": adjusted_score,
        "quality_label": adjusted_label,
        "review_required": "Yes" if review_required else "No",
        "review_reason": review_reason,
        "bad_draft_phrase": quality.get("bad_draft_phrase", ""),
        "is_grounded": validation.get("is_grounded", ""),
        "grounding_confidence": validation.get("confidence", ""),
        "citation_count": quality.get("citation_count", len(citations)),
        "citations": ", ".join(citations),
        "source_count": len(sources),
        "staff_draft": staff_draft,
        "http_status": http_status,
        "error": error,
    }


def _build_ui_thread_row(
    *,
    data: dict,
    run_timestamp: str,
    version_label: str,
    comment: str,
    case_id: str,
    turn_index: str,
    email_text: str,
    student_email: str,
    subject: str,
    thread_id: str,
    expected_followup_type: str,
    elapsed: float,
    http_status: int,
    error: str = "",
) -> dict:
    quality = data.get("quality", {}) or {}
    detected_topics = _detect_topics_from_email_response(data)

    return {
        "run_timestamp": run_timestamp,
        "version_label": version_label,
        "comment": comment,
        "case_id": case_id,
        "turn_index": turn_index,
        "student_email": student_email,
        "thread_id": data.get("thread_id") or thread_id,
        "subject": subject,
        "email_text": email_text,
        "expected_followup_type": expected_followup_type,
        "actual_followup_type": data.get("followup_type", ""),
        "review_required": "Yes" if quality.get("review_required") else "No",
        "quality_label": quality.get("quality_label", ""),
        "detected_topics": ", ".join(detected_topics),
        "staff_draft": data.get("staff_draft", ""),
        "generation_provider": data.get("generation_provider", ""),
        "generation_model": data.get("generation_model", ""),
        "response_time_seconds": round(elapsed, 2),
        "http_status": http_status,
        "error": error,
    }


# ---------------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="HANS - HTW AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.answer-box {
    background-color: #f0f8ff;
    padding: 18px;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    margin: 10px 0;
}
.review-box {
    background-color: #fff3cd;
    padding: 16px;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
    margin: 10px 0;
}
.good-box {
    background-color: #e8f5e9;
    padding: 16px;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    margin: 10px 0;
}
.source-box {
    background-color: #fff8dc;
    padding: 14px;
    border-radius: 8px;
    border-left: 4px solid #ff9800;
    margin: 8px 0;
}
.small-muted {
    color: #666;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    # Legacy shared session key. Kept only for compatibility.
    st.session_state.session_id = None
if "qa_session_id" not in st.session_state:
    # Dedicated session for Conversational QA memory.
    st.session_state.qa_session_id = None
if "email_session_id" not in st.session_state:
    # Dedicated session for Email Assistant draft workflow.
    st.session_state.email_session_id = None
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0


def _language_value(choice: str):
    if choice == "Auto-detect":
        return None
    if choice == "English":
        return "en"
    return "de"


def _default_case_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 HANS")
    st.markdown("**HTW AI Navigation System**")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        [
            "Email Assistant – Claude",
            "Email Assistant – Mistral",
            "Conversational QA",
            "Baseline QA",
        ],
        index=0,
        help=(
            "Email Assistant modes generate staff-facing drafts. "
            "Conversational QA uses session context. "
            "Baseline QA ignores previous turns."
        ),
    )

    language_choice = st.selectbox(
        "Preferred language",
        ["Auto-detect", "English", "German"],
        index=0,
    )

    show_sources = st.checkbox("Show sources", value=True)
    show_details = st.checkbox("Show technical details", value=False)

    st.markdown("---")
    st.subheader("Automatic UI logging")
    st.caption("Every UI request is saved automatically.")
    ui_version_label = st.text_input("Version label", value="ui_manual_4mode")
    ui_comment = st.text_input("Comment", value="Manual UI test")
    ui_case_id = st.text_input("Case ID", value="", help="Leave blank to auto-generate.")
    ui_title = st.text_input("Test title", value="")
    ui_turn_index = st.text_input("Turn index", value="1")
    st.caption(f"Unified log: {UI_ALL_TESTS_CSV}")

    st.markdown("---")
    st.subheader("Session")
    st.metric("Requests", st.session_state.query_count)
    avg = st.session_state.total_time / st.session_state.query_count if st.session_state.query_count else 0
    st.metric("Avg. time", f"{avg:.2f}s")
    st.caption(f"QA session: {st.session_state.qa_session_id or 'None'}")
    st.caption(f"Email session: {st.session_state.email_session_id or 'None'}")

    if st.button("Reset session"):
        st.session_state.history = []
        st.session_state.session_id = None
        st.session_state.qa_session_id = None
        st.session_state.email_session_id = None
        st.session_state.query_count = 0
        st.session_state.total_time = 0.0
        st.rerun()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("HANS PoC")
st.caption("Experimental assistant for HTW Berlin support scenarios")

if mode == "Email Assistant – Claude":
    backend_endpoint = "/email"
    backend_mode = "email_claude"
elif mode == "Email Assistant – Mistral":
    backend_endpoint = "/email"
    backend_mode = "email_mistral"
elif mode == "Conversational QA":
    backend_endpoint = "/query"
    backend_mode = "conversational_qa"
else:
    backend_endpoint = "/query"
    backend_mode = "baseline_qa"

st.caption(f"Backend endpoint: `{backend_endpoint}` | backend mode: `{backend_mode}`")


# ---------------------------------------------------------------------
# Email Assistant modes
# ---------------------------------------------------------------------
if mode in {"Email Assistant – Claude", "Email Assistant – Mistral"}:
    st.subheader("📧 Email Assistant Mode with Thread Memory")
    st.markdown(
        "This mode simulates a staff inbox workflow. Paste or select an incoming student email. "
        "HANS detects topics, retrieves evidence, keeps thread context, and generates one staff-ready draft."
    )

    demo_emails = {
        "Manual input": {"student_email": "", "subject": "", "thread_id": "", "email_text": ""},
        "New enquiry - International Business": {
            "student_email": "ananya@example.com",
            "subject": "Application for International Business Master",
            "thread_id": "thread_ananya_ib",
            "email_text": (
                "Dear Admissions Team,\n\n"
                "My name is Ananya and I am completing my Bachelor's degree in Business Administration in India. "
                "I would like to apply for the Master's in International Business for the winter semester. "
                "Since I will receive my final transcript only in July, can I still apply before graduation? "
                "Do I need English language proof and are there application fees?"
            ),
        },
        "New enquiry - Cyber Security and Business": {
            "student_email": "student@example.com",
            "subject": "Cyber Security and Business application",
            "thread_id": "thread_csb_001",
            "email_text": (
                "Dear Admissions Team,\n\n"
                "I am an EU citizen living in Portugal and I am finishing high school with an International "
                "Baccalaureate diploma. I want to apply for the Cyber Security and Business Bachelor's programme. "
                "Do I apply through uni-assist, does my IB diploma meet the admission requirements, and is a "
                "motivation letter required?\n\n"
                "Kind regards"
            ),
        },
        "New enquiry - MPMD": {
            "student_email": "student@example.com",
            "subject": "Project Management and Data Science application",
            "thread_id": "thread_mpmd_001",
            "email_text": (
                "Dear Admissions Team,\n\n"
                "I am interested in the Master's programme Project Management and Data Science. "
                "Could you please tell me whether the programme is taught in English or German, "
                "how I should apply, and where I can find the admission requirements?\n\n"
                "Kind regards"
            ),
        },
        "Follow-up - same thread new topic": {
            "student_email": "ananya@example.com",
            "subject": "Application for International Business Master",
            "thread_id": "thread_ananya_ib",
            "email_text": "Thank you for your reply. Is a motivation letter also required?",
        },
        "Follow-up - unclear answer": {
            "student_email": "ananya@example.com",
            "subject": "Application for International Business Master",
            "thread_id": "thread_ananya_ib",
            "email_text": "Thank you for your reply, but I still do not understand your answer about the application route.",
        },
    }

    selected_demo = st.selectbox("Inbox simulation", list(demo_emails.keys()))
    chosen = demo_emails[selected_demo]

    c1, c2 = st.columns(2)
    with c1:
        student_email = st.text_input("Student email", value=chosen.get("student_email", ""))
    with c2:
        thread_id = st.text_input(
            "Thread ID",
            value=chosen.get("thread_id", ""),
            help="Use the same Thread ID for follow-up emails.",
        )

    subject = st.text_input("Subject", value=chosen.get("subject", ""))

    email_text = st.text_area(
        "Incoming student email",
        value=chosen.get("email_text", ""),
        height=240,
        placeholder="Example: Dear Admissions Team, I am interested in Cyber Security and Business...",
    )

    with st.expander("Expected values for UI logging", expanded=False):
        ui_expected_followup_type = st.selectbox(
            "Expected follow-up type",
            ["", "new_enquiry", "followup_new_topic", "clarification_or_complaint"],
            index=0,
        )
        ui_expected_topics_raw = st.text_input(
            "Expected topics, comma-separated",
            value="",
            help="Example: application_route, admission_requirements, required_documents",
        )
        st.caption(
            "These fields are only for evaluation logging. They do not change the backend response."
        )

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        generate = st.button("Generate draft", type="primary")
    with col_b:
        clear_ui = st.button("Clear UI session")
    with col_c:
        st.markdown(
            '<span class="small-muted">Real email integration is not implemented. '
            "This UI simulates an inbox workflow for the thesis PoC.</span>",
            unsafe_allow_html=True,
        )

    if clear_ui:
        st.session_state.email_session_id = None
        st.rerun()

    if generate:
        case_id = ui_case_id.strip() or _default_case_id("ui_email")
        title = ui_title.strip() or subject or selected_demo
        expected_topics = _split_expected_topics(ui_expected_topics_raw)
        run_timestamp = datetime.now().isoformat()

        if not email_text.strip():
            st.warning("Please paste or select a student email first.")

            all_row = _build_all_ui_row(
                run_timestamp=run_timestamp,
                version_label=ui_version_label,
                comment=ui_comment,
                ui_mode=mode,
                backend_endpoint=backend_endpoint,
                backend_mode=backend_mode,
                case_id=case_id,
                title=title,
                turn_index=ui_turn_index,
                input_text=email_text,
                student_email=student_email,
                subject=subject,
                thread_id=thread_id,
                expected_topics=expected_topics,
                detected_topics=[],
                expected_followup_type=ui_expected_followup_type,
                actual_followup_type="",
                review_required="",
                review_reason="",
                quality_label="",
                quality_score="",
                is_grounded="",
                grounding_confidence="",
                citation_count="",
                citations=[],
                source_count="",
                elapsed=0.0,
                http_status=0,
                error="Missing email text",
                output_text="",
            )
            _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)

        else:
            sent_session_id = st.session_state.email_session_id

            payload = {
                "email_text": email_text,
                "student_email": student_email or None,
                "subject": subject or None,
                "thread_id": thread_id or None,
                "session_id": sent_session_id,
                "language": _language_value(language_choice),
                "top_k": 3,
                "mode": backend_mode,
            }

            start = time.time()
            with st.spinner("Processing email..."):
                try:
                    resp = requests.post(f"{API_URL}/email", json=payload, timeout=120)
                    elapsed = time.time() - start
                    st.session_state.query_count += 1
                    st.session_state.total_time += elapsed

                    if not resp.ok:
                        error_text = resp.text
                        st.error(f"Backend error: {resp.status_code}")
                        st.code(error_text)

                        all_row = _build_all_ui_row(
                            run_timestamp=run_timestamp,
                            version_label=ui_version_label,
                            comment=ui_comment,
                            ui_mode=mode,
                            backend_endpoint=backend_endpoint,
                            backend_mode=backend_mode,
                            generation_provider="",
                            generation_model="",
                            case_id=case_id,
                            title=title,
                            turn_index=ui_turn_index,
                            input_text=email_text,
                            student_email=student_email,
                            subject=subject,
                            thread_id=thread_id,
                            expected_topics=expected_topics,
                            detected_topics=[],
                            expected_followup_type=ui_expected_followup_type,
                            actual_followup_type="",
                            review_required="",
                            review_reason="",
                            quality_label="",
                            quality_score="",
                            is_grounded="",
                            grounding_confidence="",
                            citation_count="",
                            citations=[],
                            source_count="",
                            elapsed=elapsed,
                            http_status=resp.status_code,
                            error=error_text,
                            output_text="",
                        )
                        _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)

                    else:
                        data = resp.json()
                        returned_session_id = data.get("session_id")
                        if returned_session_id:
                            st.session_state.email_session_id = returned_session_id
                            st.session_state.session_id = returned_session_id

                        quality = data.get("quality", {}) or {}
                        validation = data.get("validation", {}) or {}
                        followup_type = data.get("followup_type") or "new_enquiry"
                        detected_topics = _detect_topics_from_email_response(data)
                        citations = _normalise_citations(data.get("citations", []) or quality.get("citations", []))
                        sources = data.get("sources", []) or []
                        staff_draft = data.get("staff_draft", "") or ""

                        if data.get("flagged_for_human") or quality.get("review_required"):
                            st.markdown(
                                f"""
<div class="review-box">
<b>Quality warning triggered.</b><br>
Follow-up type: {followup_type}<br>
Reason: {quality.get('review_reason', 'Follow-up, low confidence, missing citation, or another quality-risk trigger')}<br>
Please check and edit the draft before sending.
</div>
""",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
<div class="good-box">
<b>Quality warning: No.</b><br>
Follow-up type: {followup_type}. This means no extra quality-risk trigger was detected. Please still check the sources before sending.
</div>
""",
                                unsafe_allow_html=True,
                            )

                        st.markdown("### Email/thread context")
                        ctx_cols = st.columns(3)
                        ctx_cols[0].metric("Thread ID", data.get("thread_id") or "")
                        ctx_cols[1].metric("Follow-up type", followup_type)
                        ctx_cols[2].metric("Quality warning", "Yes" if quality.get("review_required") else "No")
                        st.caption("Quality warning = additional risk flag from the system. All Email Assistant drafts are still staff-facing and must be checked before sending.")

                        if show_details:
                            st.json({
                                "email_context": data.get("email_context"),
                                "previous_thread_context": data.get("thread_context"),
                            })

                        st.markdown("### Detected topics")
                        if detected_topics:
                            for t in data.get("detected_topics", []):
                                st.write(f"- **{t.get('label')}** (`{t.get('topic_id')}`)")
                        else:
                            st.write("No topics detected.")

                        st.markdown("### Draft response")
                        st.text_area(
                            "Ready-to-paste draft",
                            value=staff_draft,
                            height=380,
                        )

                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        c1.metric("Quality", quality.get("quality_label", ""))
                        c2.metric("Score", quality.get("quality_score", ""))
                        c3.metric("Grounded", validation.get("is_grounded", ""))
                        c4.metric("Time", f"{elapsed:.2f}s")
                        c5.metric("Provider", data.get("generation_provider", ""))
                        c6.metric("Model", data.get("generation_model", ""))

                        if show_sources and sources:
                            with st.expander(f"Sources ({len(sources)})", expanded=False):
                                for i, src in enumerate(sources, start=1):
                                    title_value = src.get("title", "Untitled")
                                    url_value = src.get("url", "")
                                    excerpt_value = src.get("excerpt", "")
                                    st.markdown(
                                        f"""
<div class="source-box">
<b>[Doc {i}] {title_value}</b><br>
<a href="{url_value}" target="_blank">{url_value}</a><br>
{excerpt_value}
</div>
""",
                                        unsafe_allow_html=True,
                                    )

                        if show_details:
                            st.markdown("### Technical details")
                            st.json({
                                "quality": quality,
                                "validation": validation,
                                "timing": data.get("timing"),
                                "citations": citations,
                            })

                        # Unified logging: every email UI success goes here.
                        all_row = _build_all_ui_row(
                            run_timestamp=run_timestamp,
                            version_label=ui_version_label,
                            comment=ui_comment,
                            ui_mode=mode,
                            backend_endpoint=backend_endpoint,
                            backend_mode=backend_mode,
                            generation_provider=data.get("generation_provider", ""),
                            generation_model=data.get("generation_model", ""),
                            case_id=case_id,
                            title=title,
                            turn_index=ui_turn_index,
                            input_text=email_text,
                            student_email=student_email,
                            subject=subject,
                            thread_id=data.get("thread_id") or thread_id,
                            expected_topics=expected_topics,
                            detected_topics=detected_topics,
                            expected_followup_type=ui_expected_followup_type,
                            actual_followup_type=followup_type,
                            review_required="Yes" if quality.get("review_required") else "No",
                            review_reason=quality.get("review_reason", ""),
                            quality_label=quality.get("quality_label", ""),
                            quality_score=quality.get("quality_score", ""),
                            is_grounded=validation.get("is_grounded", ""),
                            grounding_confidence=validation.get("confidence", ""),
                            citation_count=quality.get("citation_count", len(citations)),
                            citations=citations,
                            source_count=len(sources),
                            elapsed=elapsed,
                            http_status=resp.status_code,
                            error="",
                            output_text=staff_draft,
                            sent_session_id=sent_session_id or "",
                            session_id=data.get("session_id") or st.session_state.email_session_id or "",
                            is_followup=data.get("is_followup", ""),
                        )
                        _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)

                        # Legacy email logs are also kept.
                        email_row = _build_ui_email_assistant_row(
                            data=data,
                            run_timestamp=run_timestamp,
                            version_label=ui_version_label,
                            comment=ui_comment,
                            case_id=case_id,
                            title=title,
                            email_text=email_text,
                            student_email=student_email,
                            subject=subject,
                            thread_id=thread_id,
                            expected_topics=expected_topics,
                            elapsed=elapsed,
                            http_status=resp.status_code,
                        )
                        _append_csv_row(UI_EMAIL_ASSISTANT_CSV, EMAIL_ASSISTANT_HEADERS, email_row)

                        thread_row = _build_ui_thread_row(
                            data=data,
                            run_timestamp=run_timestamp,
                            version_label=ui_version_label,
                            comment=ui_comment,
                            case_id=case_id,
                            turn_index=ui_turn_index,
                            email_text=email_text,
                            student_email=student_email,
                            subject=subject,
                            thread_id=thread_id,
                            expected_followup_type=ui_expected_followup_type,
                            elapsed=elapsed,
                            http_status=resp.status_code,
                        )
                        _append_csv_row(UI_EMAIL_THREAD_CSV, EMAIL_THREAD_HEADERS, thread_row)

                        st.success(
                            "UI test saved to: "
                            "evaluation/hans_ui_all_test_results.csv, "
                            "evaluation/email_assistant_v5_ui_results.csv, and "
                            "evaluation/email_thread_v5_ui_results.csv"
                        )

                except Exception as e:
                    elapsed = time.time() - start
                    error_text = str(e)
                    st.error(f"Could not connect to backend: {error_text}")

                    all_row = _build_all_ui_row(
                        run_timestamp=run_timestamp,
                        version_label=ui_version_label,
                        comment=ui_comment,
                        ui_mode=mode,
                        backend_endpoint=backend_endpoint,
                        backend_mode=backend_mode,
                        case_id=case_id,
                        title=title,
                        turn_index=ui_turn_index,
                        input_text=email_text,
                        student_email=student_email,
                        subject=subject,
                        thread_id=thread_id,
                        expected_topics=expected_topics,
                        detected_topics=[],
                        expected_followup_type=ui_expected_followup_type,
                        actual_followup_type="",
                        review_required="",
                        review_reason="",
                        quality_label="",
                        quality_score="",
                        is_grounded="",
                        grounding_confidence="",
                        citation_count="",
                        citations=[],
                        source_count="",
                        elapsed=elapsed,
                        http_status=0,
                        error=error_text,
                        output_text="",
                    )
                    _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)


# ---------------------------------------------------------------------
# QA modes
# ---------------------------------------------------------------------
else:
    st.subheader("💬 Query Mode")
    st.markdown(
        "Use this mode for comparison. Baseline ignores previous turns. "
        "Conversational QA uses a dedicated QA session so follow-up questions can reuse previous context."
    )

    if backend_mode == "conversational_qa":
        st.info(f"Current Conversational QA session: {st.session_state.qa_session_id or 'None yet. A session will be created after the first question.'}")
    else:
        st.info("Baseline QA does not use session memory.")

    demo_questions = {
        "Manual input": "",
        "MPMD first question": "For the Master's programme Project Management and Data Science at HTW Berlin, where can I find the admission requirements?",
        "MPMD follow-up": "Is it taught in English or German?",
        "General uni-assist documents": "What documents are usually required when applying via uni-assist to HTW Berlin?",
        "Semester fee": "What is the semester fee at HTW Berlin and where can students find fee information?",
        "English proficiency": "What proof of English proficiency is accepted for applications via uni-assist?",
    }

    selected_question = st.selectbox("QA test example", list(demo_questions.keys()))
    default_question = demo_questions[selected_question]

    query = st.text_area(
        "Ask a question",
        value=default_question,
        height=140,
        placeholder="Example: What are the admission requirements for the Master in Information Technology?",
    )

    with st.expander("Expected values for UI logging", expanded=False):
        qa_expected_topics_raw = st.text_input(
            "Expected topics, comma-separated",
            value="",
            help="Optional. QA modes do not perform topic detection, so this is mainly for notes.",
        )
        st.caption(
            "For QA mode, topic coverage is usually not applicable. The important comparison is whether "
            "Conversational QA uses session context better than Baseline QA."
        )

    if st.button("Ask HANS", type="primary"):
        case_id = ui_case_id.strip() or _default_case_id("ui_qa")
        title = ui_title.strip() or selected_question
        expected_topics = _split_expected_topics(qa_expected_topics_raw)
        run_timestamp = datetime.now().isoformat()

        if not query.strip():
            st.warning("Please enter a question first.")

            all_row = _build_all_ui_row(
                run_timestamp=run_timestamp,
                version_label=ui_version_label,
                comment=ui_comment,
                ui_mode=mode,
                backend_endpoint=backend_endpoint,
                backend_mode=backend_mode,
                case_id=case_id,
                title=title,
                turn_index=ui_turn_index,
                input_text=query,
                student_email="",
                subject="",
                thread_id=st.session_state.qa_session_id or "",
                expected_topics=expected_topics,
                detected_topics=[],
                expected_followup_type="",
                actual_followup_type="",
                review_required="",
                review_reason="",
                quality_label="",
                quality_score="",
                is_grounded="",
                grounding_confidence="",
                citation_count="",
                citations=[],
                source_count="",
                elapsed=0.0,
                http_status=0,
                error="Missing query text",
                output_text="",
            )
            _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)

        else:
            sent_session_id = st.session_state.qa_session_id if backend_mode == "conversational_qa" else None

            payload = {
                "query": query,
                "language": _language_value(language_choice),
                "top_k": 5,
                "session_id": sent_session_id,
                "mode": backend_mode,
            }

            start = time.time()
            with st.spinner("Searching..."):
                try:
                    resp = requests.post(f"{API_URL}/query", json=payload, timeout=90)
                    elapsed = time.time() - start
                    st.session_state.query_count += 1
                    st.session_state.total_time += elapsed

                    if not resp.ok:
                        error_text = resp.text
                        st.error(f"Backend error: {resp.status_code}")
                        st.code(error_text)

                        all_row = _build_all_ui_row(
                            run_timestamp=run_timestamp,
                            version_label=ui_version_label,
                            comment=ui_comment,
                            ui_mode=mode,
                            backend_endpoint=backend_endpoint,
                            backend_mode=backend_mode,
                            case_id=case_id,
                            title=title,
                            turn_index=ui_turn_index,
                            input_text=query,
                            student_email="",
                            subject="",
                            thread_id=st.session_state.qa_session_id or "",
                            expected_topics=expected_topics,
                            detected_topics=[],
                            expected_followup_type="",
                            actual_followup_type="",
                            review_required="",
                            review_reason="",
                            quality_label="",
                            quality_score="",
                            is_grounded="",
                            grounding_confidence="",
                            citation_count="",
                            citations=[],
                            source_count="",
                            elapsed=elapsed,
                            http_status=resp.status_code,
                            error=error_text,
                            output_text="",
                        )
                        _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)

                    else:
                        data = resp.json()
                        returned_session_id = data.get("session_id")
                        if returned_session_id and backend_mode == "conversational_qa":
                            st.session_state.qa_session_id = returned_session_id
                            st.session_state.session_id = returned_session_id

                        answer = data.get("answer", "") or ""
                        validation = data.get("validation", {}) or {}
                        citations = _normalise_citations(data.get("citations", []))
                        sources = data.get("sources", []) or []

                        st.markdown("### Answer")
                        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Grounded", validation.get("is_grounded", ""))
                        c2.metric("Confidence", validation.get("confidence", ""))
                        c3.metric("Used memory", data.get("used_memory", ""))
                        c4.metric("Time", f"{elapsed:.2f}s")

                        if show_sources:
                            st.markdown("### Sources")
                            for i, src in enumerate(sources, start=1):
                                title_value = src.get("title", "Untitled")
                                url_value = src.get("url", "")
                                excerpt_value = src.get("excerpt", "")
                                st.markdown(
                                    f"""
<div class="source-box">
<b>[Doc {i}] {title_value}</b><br>
<a href="{url_value}" target="_blank">{url_value}</a><br>
{excerpt_value}
</div>
""",
                                    unsafe_allow_html=True,
                                )

                        if show_details:
                            st.markdown("### Technical details")
                            st.json(data)

                        all_row = _build_all_ui_row(
                            run_timestamp=run_timestamp,
                            version_label=ui_version_label,
                            comment=ui_comment,
                            ui_mode=mode,
                            backend_endpoint=backend_endpoint,
                            backend_mode=backend_mode,
                            case_id=case_id,
                            title=title,
                            turn_index=ui_turn_index,
                            input_text=query,
                            student_email="",
                            subject="",
                            thread_id=st.session_state.qa_session_id or "",
                            expected_topics=expected_topics,
                            detected_topics=[],
                            expected_followup_type="",
                            actual_followup_type="",
                            review_required="not_applicable",
                            review_reason="QA mode has no email review logic",
                            quality_label="",
                            quality_score="",
                            is_grounded=validation.get("is_grounded", ""),
                            grounding_confidence=validation.get("confidence", ""),
                            citation_count=len(citations),
                            citations=citations,
                            source_count=len(sources),
                            elapsed=elapsed,
                            http_status=resp.status_code,
                            error="",
                            output_text=answer,
                            sent_session_id=sent_session_id or "",
                            session_id=data.get("session_id") or st.session_state.qa_session_id or "",
                            used_memory=data.get("used_memory", ""),
                            standalone_query=data.get("standalone_query", ""),
                            detected_intent=data.get("detected_intent", ""),
                            enhanced_query=data.get("enhanced_query", ""),
                        )
                        _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)

                        st.success("UI test saved to evaluation/hans_ui_all_test_results.csv")

                except Exception as e:
                    elapsed = time.time() - start
                    error_text = str(e)
                    st.error(f"Could not connect to backend: {error_text}")

                    all_row = _build_all_ui_row(
                        run_timestamp=run_timestamp,
                        version_label=ui_version_label,
                        comment=ui_comment,
                        ui_mode=mode,
                        backend_endpoint=backend_endpoint,
                        backend_mode=backend_mode,
                        case_id=case_id,
                        title=title,
                        turn_index=ui_turn_index,
                        input_text=query,
                        student_email="",
                        subject="",
                        thread_id=st.session_state.qa_session_id or "",
                        expected_topics=expected_topics,
                        detected_topics=[],
                        expected_followup_type="",
                        actual_followup_type="",
                        review_required="",
                        review_reason="",
                        quality_label="",
                        quality_score="",
                        is_grounded="",
                        grounding_confidence="",
                        citation_count="",
                        citations=[],
                        source_count="",
                        elapsed=elapsed,
                        http_status=0,
                        error=error_text,
                        output_text="",
                    )
                    _append_csv_row(UI_ALL_TESTS_CSV, UI_ALL_HEADERS, all_row)
