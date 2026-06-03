# app/conversation.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import uuid

# In-memory session store for PoC
# NOTE: resets when backend restarts
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def default_memory() -> Dict[str, Any]:
    return {
        "turn_count": 0,
        "language": None,
        "current_topic": None,
        "current_program": None,
        "current_degree_level": None,
        "current_semester": None,
        "current_deadline_type": None,
        "last_user_query": None,
        "last_standalone_query": None,
        "last_answer_summary": None,
    }


def get_or_create_session(session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if session_id and session_id in SESSION_STORE:
        return session_id, SESSION_STORE[session_id]

    new_session_id = str(uuid.uuid4())
    SESSION_STORE[new_session_id] = default_memory()
    return new_session_id, SESSION_STORE[new_session_id]


def is_follow_up_query(query: str) -> bool:
    q = (query or "").strip().lower()

    follow_up_starts = [
        "and ",
        "what about",
        "how about",
        "and what",
        "and for",
        "for that",
        "for this",
        "what is the deadline",
        "do i need",
        "what documents",
    ]

    if len(q.split()) <= 5:
        return True

    if any(q.startswith(p) for p in follow_up_starts):
        return True

    vague_terms = ["it", "that", "this", "there", "deadline", "documents", "requirements"]
    if any(term in q.split() for term in vague_terms) and len(q.split()) <= 8:
        return True

    return False


def detect_intent(query: str) -> str:
    """
    Lightweight intent detection for the PoC.

    This intentionally remains rule-based so the behaviour is transparent
    for thesis testing and easy to inspect in Streamlit.
    """
    q = (query or "").lower()

    if any(x in q for x in ["fee", "fees", "tuition", "cost", "semester fee"]):
        return "fees"

    if any(x in q for x in ["duration", "how long", "semesters", "study period"]):
        return "duration"

    if "deadline" in q or "frist" in q:
        if "enrollment" in q or "enrolment" in q or "einschreibung" in q:
            return "enrollment_deadline"
        return "application_deadline"

    if "enrollment" in q or "enrolment" in q or "einschreibung" in q:
        return "enrollment_process"

    if "document" in q or "documents" in q or "unterlagen" in q or "certificate" in q or "transcript" in q:
        return "required_documents"

    if "german" in q or "english" in q or "language" in q or "deutsch" in q or "englisch" in q or "b2" in q or "c1" in q:
        return "language_requirements"

    if "requirement" in q or "requirements" in q or "admission" in q or "eligible" in q or "eligibility" in q:
        return "admission_requirements"

    if "apply" in q or "application" in q or "uni-assist" in q:
        return "application_process"

    if "semester" in q and ("start" in q or "begin" in q):
        return "semester_dates"

    if "tell me about" in q or "overview" in q or "program" in q or "programme" in q:
        return "programme_overview"

    return "general_info"


def enhance_query_by_intent(query: str, intent: str) -> str:
    """
    Add intent-specific retrieval hints without changing the user-facing query.

    The original/standalone query is still used for generation. This enhanced
    query is only used during retrieval and reranking to make relevant chunks
    easier to find.
    """
    intent_keywords = {
        "application_deadline": "application deadline application period apply summer semester winter semester",
        "enrollment_deadline": "enrollment enrolment deadline registration semester",
        "enrollment_process": "enrollment enrolment registration process documents",
        "application_process": "application apply uni-assist application portal admission",
        "admission_requirements": "admission requirements eligibility bachelor degree ECTS prerequisites",
        "required_documents": "required documents application documents certificates transcript upload proof",
        "language_requirements": "language requirements English German B2 C1 proof certificate",
        "fees": "tuition fees semester fee cost contribution",
        "duration": "duration semesters study period standard period",
        "semester_dates": "semester dates start begin summer semester winter semester",
        "programme_overview": "programme overview structure modules degree curriculum",
    }
    extra = intent_keywords.get(intent, "")
    return f"{query} {extra}".strip()


def extract_entities(query: str) -> Dict[str, Optional[str]]:
    q = (query or "").lower()

    entities = {
        "degree_level": None,
        "semester": None,
        "program": None,
    }

    if "master" in q:
        entities["degree_level"] = "master"
    elif "bachelor" in q:
        entities["degree_level"] = "bachelor"

    if "summer semester" in q:
        entities["semester"] = "summer semester"
    elif "winter semester" in q:
        entities["semester"] = "winter semester"

    known_programs = [
        "information technology",
        "data science",
        "business administration",
        "computer science",
        "engineering",
    ]

    for p in known_programs:
        if p in q:
            entities["program"] = p
            break

    return entities


def reformulate_query_with_memory(query: str, memory: Dict[str, Any]) -> str:
    q = (query or "").strip()

    if not is_follow_up_query(q):
        return q

    lower_q = q.lower()

    topic = memory.get("current_topic")
    program = memory.get("current_program")
    degree_level = memory.get("current_degree_level")
    semester = memory.get("current_semester")

    if "deadline" in lower_q:
        if "enrollment" in lower_q or topic == "enrollment_deadline":
            base = "What is the enrollment deadline"
        else:
            base = "What is the application deadline"

        if program:
            base += f" for the {program}"
        elif degree_level:
            base += f" for the {degree_level} program"

        if semester:
            base += f" for the {semester}"

        base += " at HTW Berlin?"
        return base

    if "document" in lower_q or "requirement" in lower_q or "need" in lower_q:
        base = "What documents or requirements are needed"

        if program:
            base += f" for the {program}"
        elif degree_level:
            base += f" for the {degree_level} program"

        if semester:
            base += f" for the {semester}"

        base += " at HTW Berlin?"
        return base

    if "german" in lower_q or "english" in lower_q or "language" in lower_q:
        base = "What are the language requirements"

        if program:
            base += f" for the {program}"
        elif degree_level:
            base += f" for the {degree_level} program"

        base += " at HTW Berlin?"
        return base

    if "summer semester" in lower_q or "winter semester" in lower_q:
        if topic == "application_deadline":
            base = "What is the application deadline"
        elif topic == "enrollment_deadline":
            base = "What is the enrollment deadline"
        else:
            base = "What is the relevant information"

        if program:
            base += f" for the {program}"
        elif degree_level:
            base += f" for the {degree_level} program"

        if "summer semester" in lower_q:
            base += " for the summer semester"
        elif "winter semester" in lower_q:
            base += " for the winter semester"

        base += " at HTW Berlin?"
        return base

    if memory.get("last_standalone_query"):
        return f"{q} (context: {memory['last_standalone_query']})"

    return q


def update_session_memory(
    memory: Dict[str, Any],
    raw_query: str,
    standalone_query: str,
    intent: str,
    entities: Dict[str, Optional[str]],
    answer: str,
) -> None:
    memory["turn_count"] += 1
    memory["last_user_query"] = raw_query
    memory["last_standalone_query"] = standalone_query
    memory["current_topic"] = intent or memory.get("current_topic")

    if entities.get("program"):
        memory["current_program"] = entities["program"]

    if entities.get("degree_level"):
        memory["current_degree_level"] = entities["degree_level"]

    if entities.get("semester"):
        memory["current_semester"] = entities["semester"]

    if intent == "application_deadline":
        memory["current_deadline_type"] = "application_deadline"
    elif intent == "enrollment_deadline":
        memory["current_deadline_type"] = "enrollment_deadline"

    memory["last_answer_summary"] = (answer or "")[:300]


def get_memory_snapshot(memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "turn_count": memory.get("turn_count"),
        "current_topic": memory.get("current_topic"),
        "current_program": memory.get("current_program"),
        "current_degree_level": memory.get("current_degree_level"),
        "current_semester": memory.get("current_semester"),
        "last_standalone_query": memory.get("last_standalone_query"),
    }