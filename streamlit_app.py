# streamlit_app.py
"""
Streamlit UI for HANS PoC.

V4.5 adds a staff-facing Email Assistant mode:
- paste incoming student email,
- detect topics,
- retrieve evidence per topic,
- generate one final staff-ready draft,
- show review decision and sources.

The older Baseline and Conversational modes are still available for comparison.
"""

from __future__ import annotations

import os
import time
import csv
import json
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("BACKEND_URL", "http://localhost:8001")


# ---------------------------------------------------------------------
# UI CSV logging
# ---------------------------------------------------------------------
# The script-based tests already save CSV files. These UI files are only for
# manual UI test runs. Their headers intentionally match the script result CSVs.
EVAL_DIR = os.path.join(os.path.dirname(__file__), "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

UI_EMAIL_ASSISTANT_CSV = os.path.join(EVAL_DIR, "email_assistant_v5_ui_results.csv")
UI_EMAIL_THREAD_CSV = os.path.join(EVAL_DIR, "email_thread_v5_ui_results.csv")

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


def _append_csv_row(path: str, headers: list[str], row: dict) -> None:
    """Append one UI test result row using the same headers as the script CSV."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _topic_metrics(expected_topics: list[str], detected_topics: list[str]) -> dict:
    expected_set = set(expected_topics or [])
    detected_set = set(detected_topics or [])

    if not expected_set:
        return {
            "missing_topics": "",
            "extra_topics": ", ".join(sorted(detected_set)),
            "topic_coverage_percent": "",
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


def _split_expected_topics(raw: str) -> list[str]:
    """Convert comma-separated UI input into a clean topic list."""
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


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
) -> dict:
    detected_topics = [
        t.get("topic_id", "")
        for t in data.get("detected_topics", [])
        if t.get("topic_id")
    ]
    metrics = _topic_metrics(expected_topics, detected_topics)

    quality = data.get("quality", {}) or {}
    validation = data.get("validation", {}) or {}
    citations = data.get("citations", []) or []
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
        "error": "",
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
) -> dict:
    quality = data.get("quality", {}) or {}
    detected_topics = [
        t.get("topic_id", "")
        for t in data.get("detected_topics", [])
        if t.get("topic_id")
    ]

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
        "response_time_seconds": round(elapsed, 2),
        "http_status": http_status,
        "error": "",
    }

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
    st.session_state.session_id = None
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 HANS")
    st.markdown("**HTW AI Navigation System**")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Email Assistant", "Conversational QA", "Baseline QA"],
        index=0,
        help=(
            "Email Assistant is the V4.5 staff-facing workflow. "
            "Conversational QA and Baseline QA are kept for comparison."
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
    st.subheader("Session")
    st.metric("Requests", st.session_state.query_count)
    avg = st.session_state.total_time / st.session_state.query_count if st.session_state.query_count else 0
    st.metric("Avg. time", f"{avg:.2f}s")

    if st.button("Reset session"):
        st.session_state.history = []
        st.session_state.session_id = None
        st.session_state.query_count = 0
        st.session_state.total_time = 0.0
        st.rerun()


def _language_value(choice: str):
    if choice == "Auto-detect":
        return None
    if choice == "English":
        return "en"
    return "de"


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("HANS PoC")
st.caption("Experimental assistant for HTW Berlin support scenarios")


# ---------------------------------------------------------------------
# Email Assistant mode
# ---------------------------------------------------------------------
if mode == "Email Assistant":
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
        thread_id = st.text_input("Thread ID", value=chosen.get("thread_id", ""), help="Use the same Thread ID for follow-up emails.")

    subject = st.text_input("Subject", value=chosen.get("subject", ""))

    email_text = st.text_area(
        "Incoming student email",
        value=chosen.get("email_text", ""),
        height=240,
        placeholder="Example: Dear Admissions Team, I am interested in Cybersecurity and Business...",
    )

    with st.expander("Manual UI test logging", expanded=False):
        log_ui_result = st.checkbox("Save this UI run to CSV", value=True)
        ui_version_label = st.text_input("Version label", value="v5_ui_manual")
        ui_comment = st.text_input("Comment", value="Manual UI test")
        ui_case_id = st.text_input("Case ID", value="")
        ui_title = st.text_input("Test title", value=subject or selected_demo)
        ui_turn_index = st.text_input("Turn index", value="1")
        ui_expected_followup_type = st.selectbox(
            "Expected follow-up type",
            ["", "new_enquiry", "followup_new_topic", "clarification_or_complaint"],
            index=0,
        )
        ui_expected_topics_raw = st.text_input(
            "Expected topics, comma-separated",
            value="",
            help="Example: application_route, qualification_recognition, motivation_letter",
        )
        st.caption(
            "UI results are saved with the same headers as the script CSV files. "
            "Email assistant rows go to evaluation/email_assistant_v5_ui_results.csv. "
            "Thread rows go to evaluation/email_thread_v5_ui_results.csv."
        )

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        generate = st.button("Generate draft", type="primary")
    with col_b:
        clear_ui = st.button("Clear UI session")
    with col_c:
        st.markdown('<span class="small-muted">Real email integration is not implemented. This UI simulates an inbox workflow for the thesis PoC.</span>', unsafe_allow_html=True)

    if clear_ui:
        st.session_state.session_id = None
        st.rerun()

    if generate:
        if not email_text.strip():
            st.warning("Please paste or select a student email first.")
        else:
            payload = {
                "email_text": email_text,
                "student_email": student_email or None,
                "subject": subject or None,
                "thread_id": thread_id or None,
                "session_id": st.session_state.session_id,
                "language": _language_value(language_choice),
                "top_k": 3,
            }

            start = time.time()
            with st.spinner("Processing email..."):
                try:
                    resp = requests.post(f"{API_URL}/email", json=payload, timeout=120)
                    elapsed = time.time() - start
                    st.session_state.query_count += 1
                    st.session_state.total_time += elapsed

                    if not resp.ok:
                        st.error(f"Backend error: {resp.status_code}")
                        st.code(resp.text)
                    else:
                        data = resp.json()
                        st.session_state.session_id = data.get("session_id", st.session_state.session_id)

                        quality = data.get("quality", {})
                        validation = data.get("validation", {})
                        followup_type = data.get("followup_type") or "new_enquiry"

                        if data.get("flagged_for_human") or quality.get("review_required"):
                            st.markdown(
                                f"""
<div class="review-box">
<b>Human review required</b><br>
Follow-up type: {followup_type}<br>
Reason: {quality.get('review_reason', 'Follow-up or low confidence')}
</div>
""",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
<div class="good-box">
<b>Draft looks usable for staff review.</b><br>
Follow-up type: {followup_type}. Please still check the sources before sending.
</div>
""",
                                unsafe_allow_html=True,
                            )

                        st.markdown("### Email/thread context")
                        ctx_cols = st.columns(3)
                        ctx_cols[0].metric("Thread ID", data.get("thread_id") or "")
                        ctx_cols[1].metric("Follow-up type", followup_type)
                        ctx_cols[2].metric("Review", "Yes" if quality.get("review_required") else "No")

                        if show_details:
                            st.json({
                                "email_context": data.get("email_context"),
                                "previous_thread_context": data.get("thread_context"),
                            })

                        st.markdown("### Detected topics")
                        topics = data.get("detected_topics", [])
                        if topics:
                            for t in topics:
                                st.write(f"- **{t.get('label')}**")
                        else:
                            st.write("No topics detected.")

                        st.markdown("### Draft response")
                        st.text_area(
                            "Ready-to-paste draft",
                            value=data.get("staff_draft", ""),
                            height=380,
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Quality", quality.get("quality_label", ""))
                        c2.metric("Score", quality.get("quality_score", ""))
                        c3.metric("Grounded", validation.get("is_grounded", ""))
                        c4.metric("Time", f"{elapsed:.2f}s")

                        if show_sources:
                            st.markdown("### Sources")
                            for i, src in enumerate(data.get("sources", []), start=1):
                                st.markdown(
                                    f"""
<div class="source-box">
<b>[Doc {i}] {src.get('title', 'Untitled')}</b><br>
<a href="{src.get('url', '')}" target="_blank">{src.get('url', '')}</a><br>
{src.get('excerpt', '')}
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
                                "citations": data.get("citations"),
                            })

                        if log_ui_result:
                            run_timestamp = datetime.now().isoformat()
                            expected_topics = _split_expected_topics(ui_expected_topics_raw)
                            case_id = ui_case_id.strip() or f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                            email_row = _build_ui_email_assistant_row(
                                data=data,
                                run_timestamp=run_timestamp,
                                version_label=ui_version_label,
                                comment=ui_comment,
                                case_id=case_id,
                                title=ui_title,
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
                                "UI test saved to CSV files: "
                                "evaluation/email_assistant_v5_ui_results.csv and "
                                "evaluation/email_thread_v5_ui_results.csv"
                            )

                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")


# ---------------------------------------------------------------------
# QA modes
# ---------------------------------------------------------------------
else:
    selected_mode = "conversation" if mode == "Conversational QA" else "baseline"

    st.subheader("💬 Query Mode")
    st.markdown(
        "Use this mode for comparison with earlier PoC versions. "
        "Baseline ignores previous turns; Conversational uses session context."
    )

    query = st.text_area(
        "Ask a question",
        height=140,
        placeholder="Example: What are the admission requirements for the Master in Information Technology?",
    )

    if st.button("Ask HANS", type="primary"):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            payload = {
                "query": query,
                "language": _language_value(language_choice),
                "top_k": 5,
                "session_id": st.session_state.session_id,
                "mode": selected_mode,
            }

            start = time.time()
            with st.spinner("Searching..."):
                try:
                    resp = requests.post(f"{API_URL}/query", json=payload, timeout=90)
                    elapsed = time.time() - start
                    st.session_state.query_count += 1
                    st.session_state.total_time += elapsed

                    if not resp.ok:
                        st.error(f"Backend error: {resp.status_code}")
                        st.code(resp.text)
                    else:
                        data = resp.json()
                        st.session_state.session_id = data.get("session_id", st.session_state.session_id)

                        st.markdown("### Answer")
                        st.markdown(f"<div class='answer-box'>{data.get('answer', '')}</div>", unsafe_allow_html=True)

                        validation = data.get("validation", {})
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Grounded", validation.get("is_grounded", ""))
                        c2.metric("Confidence", validation.get("confidence", ""))
                        c3.metric("Time", f"{elapsed:.2f}s")

                        if show_sources:
                            st.markdown("### Sources")
                            for i, src in enumerate(data.get("sources", []), start=1):
                                st.markdown(
                                    f"""
<div class="source-box">
<b>[Doc {i}] {src.get("title", "Untitled")}</b><br>
<a href="{src.get("url", "")}" target="_blank">{src.get("url", "")}</a><br>
{src.get("excerpt", "")}
</div>
""",
                                    unsafe_allow_html=True,
                                )

                        if show_details:
                            st.markdown("### Technical details")
                            st.json(data)

                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")
