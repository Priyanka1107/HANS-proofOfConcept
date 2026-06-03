"""
Lightweight thread memory for the HANS staff-facing email assistant.

Purpose:
- Keep a small amount of context for each email thread.
- Detect whether a new incoming email is:
    1. a new enquiry,
    2. a follow-up asking a new topic,
    3. a clarification/complaint that should be flagged for human review.
- Help short follow-up emails reuse the previous programme/context.

This is intentionally simple and local for the PoC.
It does not send emails automatically.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "evaluation" / "email_thread_memory.json"


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _normalise_key(value: str) -> str:
    """
    Convert a thread/email/subject value into a safe local key.
    """
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9@._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def build_thread_key(
    *,
    thread_id: Optional[str] = None,
    student_email: Optional[str] = None,
    subject: Optional[str] = None,
) -> str:
    """
    Build a stable key for an email thread.

    Priority:
    1. Use explicit thread_id if available.
    2. Otherwise combine student_email and subject.
    3. Otherwise return a safe fallback key.

    For the tests, this is important:
    the same thread_id must be used for all turns in one thread.
    """
    if thread_id and str(thread_id).strip():
        return _normalise_key(str(thread_id))

    email_part = _normalise_key(student_email or "")
    subject_part = _normalise_key(subject or "")

    if email_part != "unknown" or subject_part != "unknown":
        return f"{email_part}__{subject_part}"

    return "unknown_thread"


def _load_all_memory(path: Path = DEFAULT_MEMORY_PATH) -> Dict[str, Any]:
    """
    Load complete thread memory file.
    """
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}

    return {}


def _save_all_memory(data: Dict[str, Any], path: Path = DEFAULT_MEMORY_PATH) -> None:
    """
    Save complete thread memory file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Public memory functions used by app/main.py
# ---------------------------------------------------------------------

def load_thread_context(thread_key: str) -> Optional[Dict[str, Any]]:
    """
    Return stored context for a thread, or None if no previous context exists.
    """
    key = _normalise_key(thread_key)
    data = _load_all_memory()
    item = data.get(key)

    if isinstance(item, dict):
        return item

    return None


def save_thread_context(
    thread_key: str,
    *,
    student_email: Optional[str] = None,
    subject: Optional[str] = None,
    email_context: Optional[Dict[str, Any]] = None,
    detected_topics: Optional[List[Dict[str, Any]]] = None,
    staff_draft: Optional[str] = None,
    quality: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save lightweight context for future emails in the same thread.

    Stored data is intentionally small:
    - latest interpreted email context,
    - latest detected topics,
    - a short draft preview,
    - basic quality status.

    This is enough for the PoC to demonstrate thread memory without storing
    unnecessary full conversation history.
    """
    key = _normalise_key(thread_key)
    data = _load_all_memory()

    previous = data.get(key, {})
    if not isinstance(previous, dict):
        previous = {}

    turn_count = int(previous.get("turn_count", 0) or 0) + 1

    detected_topic_ids: List[str] = []
    for topic in detected_topics or []:
        if isinstance(topic, dict):
            tid = topic.get("topic_id")
        else:
            tid = getattr(topic, "topic_id", None)
        if tid:
            detected_topic_ids.append(str(tid))

    data[key] = {
        "thread_key": key,
        "student_email": student_email or previous.get("student_email"),
        "subject": subject or previous.get("subject"),
        "turn_count": turn_count,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "email_context": email_context or previous.get("email_context", {}),
        "detected_topics": detected_topic_ids,
        "last_quality_label": (quality or {}).get("quality_label", previous.get("last_quality_label", "")),
        "last_review_required": (quality or {}).get("review_required", previous.get("last_review_required", "")),
        "last_review_reason": (quality or {}).get("review_reason", previous.get("last_review_reason", "")),
        "last_draft_preview": (staff_draft or previous.get("last_draft_preview", ""))[:1000],
    }

    _save_all_memory(data)


# ---------------------------------------------------------------------
# Follow-up classification
# ---------------------------------------------------------------------

def classify_followup_email(email_text: str, has_thread_memory: bool = False) -> str:
    """
    Classify an incoming email in the staff-facing email workflow.

    Return values:
    - new_enquiry
    - followup_new_topic
    - clarification_or_complaint

    Important safety rule:
    Clear clarification or complaint wording is enough to flag human review,
    even if thread memory is missing.
    """
    text = (email_text or "").lower().strip()

    # Strong clarification / complaint signals.
    # These should be flagged for human review.
    clarification_patterns = [
        r"\bi still (do not|don't) understand\b",
        r"\bi do not understand\b",
        r"\bi don't understand\b",
        r"\bi am still confused\b",
        r"\bi am confused\b",
        r"\bthis does not answer\b",
        r"\bthis did not answer\b",
        r"\bthis is not clear\b",
        r"\bthat is not clear\b",
        r"\bthe answer was not clear\b",
        r"\byour answer was not clear\b",
        r"\byour reply was not clear\b",
        r"\byour response was not clear\b",
        r"\byour answer is unclear\b",
        r"\byour reply is unclear\b",
        r"\byour response is unclear\b",
        r"\bunclear\b",
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\bnot correct\b",
        r"\bdoes not seem correct\b",
        r"\bplease clarify\b",
        r"\bcan you clarify\b",
        r"\bcould you clarify\b",
        r"\bclarification\b",
    ]

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in clarification_patterns):
        return "clarification_or_complaint"

    # Normal follow-up signals.
    # These can still receive a draft if thread memory exists.
    followup_patterns = [
        r"\bthank you for your reply\b",
        r"\bthank you for the reply\b",
        r"\bthanks for your reply\b",
        r"\bthanks for the reply\b",
        r"\bfollowing up\b",
        r"\bfollow up\b",
        r"\bin addition\b",
        r"\balso\b",
        r"\band what about\b",
        r"\bwhat about\b",
        r"\bone more question\b",
        r"\banother question\b",
        r"\bis .* also required\b",
        r"\bdo i also need\b",
        r"\bcan i also\b",
        r"\bcould you also\b",
    ]

    if has_thread_memory and any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in followup_patterns):
        return "followup_new_topic"

    return "new_enquiry"


# ---------------------------------------------------------------------
# Context merge for short follow-ups
# ---------------------------------------------------------------------

def merge_context_with_thread(
    current_context: Optional[Dict[str, Any]],
    previous_thread_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge current email context with previous thread context.

    The current email context wins.
    Missing values are filled from previous thread memory.

    This helps short follow-ups such as:
        "And what about the application deadline?"

    because the programme from the previous turn can still be used.
    """
    merged: Dict[str, Any] = {}

    if isinstance(previous_thread_context, dict):
        previous_email_context = previous_thread_context.get("email_context", {})
        if isinstance(previous_email_context, dict):
            merged.update(previous_email_context)

    if isinstance(current_context, dict):
        for key, value in current_context.items():
            if value not in (None, "", [], {}):
                merged[key] = value

    return merged