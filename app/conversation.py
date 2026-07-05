# app/conversation.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import json
import os
import re
import uuid


# ---------------------------------------------------------------------
# Lightweight conversation memory for QA mode
# ---------------------------------------------------------------------
# PoC session memory only: no model training, no long-term learning.
# Used to reformulate follow-up QA questions inside one UI/session.
# Stored locally so UI tests do not lose context if uvicorn reloads.
# ---------------------------------------------------------------------

SESSION_STORE: Dict[str, Dict[str, Any]] = {}

_MEMORY_DIR = os.path.join(os.getcwd(), "evaluation")
_MEMORY_PATH = os.path.join(_MEMORY_DIR, "query_session_memory.json")
_STORE_LOADED = False


def _default_memory() -> Dict[str, Any]:
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


def default_memory() -> Dict[str, Any]:
    return _default_memory()


def _load_store_once() -> None:
    global _STORE_LOADED, SESSION_STORE
    if _STORE_LOADED:
        return
    _STORE_LOADED = True

    try:
        if os.path.exists(_MEMORY_PATH):
            with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                cleaned = {}
                for sid, memory in data.items():
                    if isinstance(memory, dict):
                        merged = _default_memory()
                        merged.update(memory)
                        cleaned[str(sid)] = merged
                SESSION_STORE = cleaned
    except Exception:
        SESSION_STORE = {}


def _save_store() -> None:
    try:
        os.makedirs(_MEMORY_DIR, exist_ok=True)
        with open(_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(SESSION_STORE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_or_create_session(session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Return an existing QA session or create a new one."""
    _load_store_once()

    cleaned_id = (session_id or "").strip()

    if cleaned_id:
        if cleaned_id not in SESSION_STORE:
            SESSION_STORE[cleaned_id] = _default_memory()
            _save_store()
        return cleaned_id, SESSION_STORE[cleaned_id]

    new_session_id = str(uuid.uuid4())
    SESSION_STORE[new_session_id] = _default_memory()
    _save_store()
    return new_session_id, SESSION_STORE[new_session_id]


# ---------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------

def _looks_german(text: str) -> bool:
    q = (text or "").lower()
    german_markers = [
        "ich ", "mich ", "für ", "studiengang", "zulassung",
        "zulassungsvoraussetzungen", "bewerbung", "bewerbungsfrist",
        "unterlagen", "deutsch", "englisch", "unterrichtet",
        "unterrichtssprache", "gebühren", "semesterbeitrag", "zeugnis",
        "nachweis", "wird ", "brauche ", "muss ", "kann ",
    ]
    return any(marker in q for marker in german_markers) or bool(re.search(r"[äöüß]", q))


# ---------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------

def is_follow_up_query(query: str) -> bool:
    q = (query or "").strip().lower()
    words = q.split()

    if not q:
        return False

    # Very short questions are often follow-ups:
    # "And fees?", "Documents?", "In English?", "Und Gebühren?", etc.
    if len(words) <= 6:
        return True

    follow_up_starts = [
        # English
        "and ", "what about", "how about", "and what", "and for",
        "for that", "for this", "for the same", "same for",
        "what is the deadline", "do i need", "what documents",
        "is it", "is this", "is the programme", "is the program",
        "does it", "does this", "can i",
        # German
        "und ", "und was", "was ist mit", "wie sieht es",
        "gilt das", "ist er", "ist sie", "ist es",
        "wird er", "wird sie", "wird es", "wird der",
        "wird die", "wird das", "brauche ich", "muss ich",
        "kann ich", "welche unterlagen", "welche dokumente",
        "wann ist", "wie ist",
    ]

    if any(q.startswith(p) for p in follow_up_starts):
        return True

    vague_terms = {
        # English
        "it", "that", "this", "there", "deadline", "documents",
        "requirements", "fees", "language", "english", "german",
        # German
        "er", "sie", "es", "das", "dieser", "diese", "dieses",
        "dort", "frist", "bewerbungsfrist", "unterlagen", "dokumente",
        "voraussetzungen", "zulassungsvoraussetzungen", "gebühren",
        "kosten", "sprache", "englisch", "deutsch", "unterrichtet",
    }

    if any(term in words for term in vague_terms) and len(words) <= 14:
        return True

    return False


def detect_intent(query: str) -> str:
    """Transparent rule-based intent detection for the PoC."""
    q = (query or "").lower()

    # Language/teaching-language checks first.
    if (
        "language of instruction" in q
        or "teaching language" in q
        or "taught in english" in q
        or "taught in german" in q
        or "taught in english or german" in q
        or "language" in q
        or "english" in q
        or "german" in q
        or "deutsch" in q
        or "englisch" in q
        or "unterrichtssprache" in q
        or "unterrichtet" in q
        or "auf englisch" in q
        or "auf deutsch" in q
        or "englisch oder deutsch" in q
        or "sprachnachweis" in q
        or "sprachkenntnisse" in q
        or "b2" in q
        or "c1" in q
    ):
        return "language_requirements"

    if any(x in q for x in ["fee", "fees", "tuition", "cost", "semester fee", "semester contribution", "gebühr", "gebühren", "kosten", "semesterbeitrag"]):
        return "fees"

    if any(x in q for x in ["duration", "how long", "semesters", "study period", "dauer", "semesteranzahl", "regelstudienzeit"]):
        return "duration"

    if "deadline" in q or "frist" in q or "bewerbungsfrist" in q:
        if "enrollment" in q or "enrolment" in q or "einschreibung" in q or "immatrikulation" in q:
            return "enrollment_deadline"
        return "application_deadline"

    if "enrollment" in q or "enrolment" in q or "einschreibung" in q or "immatrikulation" in q:
        return "enrollment_process"

    if (
        "document" in q or "documents" in q or "unterlagen" in q
        or "dokumente" in q or "certificate" in q or "transcript" in q
        or "zeugnis" in q or "motivation letter" in q
        or "letter of motivation" in q or "motivationsschreiben" in q
    ):
        return "required_documents"

    if (
        "requirement" in q or "requirements" in q or "admission" in q
        or "eligible" in q or "eligibility" in q or "zulassung" in q
        or "zulassungsvoraussetzungen" in q or "voraussetzungen" in q
    ):
        return "admission_requirements"

    if (
        "apply" in q or "application" in q or "uni-assist" in q
        or "application portal" in q or "bewerben" in q
        or "bewerbung" in q or "bewerbungsportal" in q
    ):
        return "application_process"

    if "semester" in q and ("start" in q or "begin" in q or "beginn" in q):
        return "semester_dates"

    if "tell me about" in q or "overview" in q or "program" in q or "programme" in q or "studiengang" in q or "überblick" in q:
        return "programme_overview"

    return "general_info"


def enhance_query_by_intent(query: str, intent: str) -> str:
    """Add intent-specific retrieval hints without changing the user-facing query."""
    intent_keywords = {
        "application_deadline": "application deadline application period apply Bewerbungsfrist Bewerbungszeitraum summer semester winter semester",
        "enrollment_deadline": "enrollment enrolment deadline registration Einschreibung Immatrikulation semester",
        "enrollment_process": "enrollment enrolment registration process documents Einschreibung Immatrikulation Unterlagen",
        "application_process": "application apply uni-assist application portal admission Bewerbung bewerben Bewerbungsportal Zulassung",
        "admission_requirements": "admission requirements eligibility bachelor degree ECTS prerequisites Zulassungsvoraussetzungen Voraussetzungen",
        "required_documents": "required documents application documents certificates transcript upload proof motivation letter Unterlagen Dokumente Zeugnis Nachweis Motivationsschreiben",
        "language_requirements": "language requirements language of instruction taught in English German Unterrichtssprache auf Englisch auf Deutsch Sprachnachweis B2 C1 proof certificate",
        "fees": "tuition fees application fee semester fee semester contribution cost Gebühren Kosten Semesterbeitrag",
        "duration": "duration semesters study period standard period Dauer Regelstudienzeit",
        "semester_dates": "semester dates start begin summer semester winter semester Studienbeginn Sommersemester Wintersemester",
        "programme_overview": "programme overview structure modules degree curriculum Studiengang Überblick Module Curriculum",
    }
    extra = intent_keywords.get(intent, "")
    return f"{query} {extra}".strip()


_PROGRAM_PATTERNS = [
    (r"\bproject management and data science\b", "Master's programme Project Management and Data Science"),
    (r"\bmpmd\b", "Master's programme Project Management and Data Science"),
    (r"\binternational business\b", "International Business"),
    (r"\bcyber\s*security and business\b", "Cyber Security and Business"),
    (r"\bcybersecurity and business\b", "Cyber Security and Business"),
    (r"\binformation technology\b", "Information Technology"),
    (r"\bcomputer engineering\b", "Computer Engineering"),
    (r"\bcomputer science\b", "Computer Science"),
    (r"\binformatik\b", "Computer Science"),
    (r"\bbusiness administration\b", "Business Administration"),
    (r"\bbetriebswirtschaftslehre\b", "Business Administration"),
    (r"\bdata science\b", "Data Science"),
    (r"\bprofessional it and digitalization\b", "Professional IT and Digitalization"),
    (r"\bprofessional it business and digitalization\b", "Professional IT Business and Digitalization"),
    (r"\bproitd\b", "Professional IT Business and Digitalization"),
    (r"\bconstruction and real estate management\b", "Construction and Real Estate Management"),
    (r"\bquantitative finance and data science\b", "Quantitative Finance and Data Science"),
]


def _extract_program(query: str) -> Optional[str]:
    q = (query or "").lower()

    for pattern, canonical_name in _PROGRAM_PATTERNS:
        if re.search(pattern, q):
            return canonical_name

    # Conservative fallback after "programme/program/studiengang".
    match = re.search(
        r"(?:programme|program|master'?s programme|bachelor'?s programme|studiengang|masterstudiengang|bachelorstudiengang)\s+(?:in|of|für|zum|zur)?\s+([A-Za-zÄÖÜäöüß &-]{4,100})",
        query or "",
        flags=re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip(" .?,-")
        if candidate:
            return candidate

    return None


def extract_entities(query: str) -> Dict[str, Optional[str]]:
    q = (query or "").lower()

    entities: Dict[str, Optional[str]] = {
        "degree_level": None,
        "semester": None,
        "program": None,
    }

    if "master" in q or "master's" in q or "masters" in q or "masterstudiengang" in q:
        entities["degree_level"] = "master"
    elif "bachelor" in q or "bachelor's" in q or "bachelorstudiengang" in q:
        entities["degree_level"] = "bachelor"

    if "summer semester" in q or "sommersemester" in q:
        entities["semester"] = "summer semester"
    elif "winter semester" in q or "wintersemester" in q:
        entities["semester"] = "winter semester"

    entities["program"] = _extract_program(query)

    return entities


def _program_or_degree_phrase_en(program: Optional[str], degree_level: Optional[str]) -> str:
    if program:
        return f" for {program}"
    if degree_level:
        return f" for the {degree_level} programme"
    return ""


def _program_or_degree_phrase_de(program: Optional[str], degree_level: Optional[str]) -> str:
    if program:
        return f" für {program}"
    if degree_level == "master":
        return " für den Masterstudiengang"
    if degree_level == "bachelor":
        return " für den Bachelorstudiengang"
    return ""


def reformulate_query_with_memory(query: str, memory: Dict[str, Any]) -> str:
    """Rewrite short follow-up questions into standalone questions."""
    q = (query or "").strip()

    if not is_follow_up_query(q):
        return q

    lower_q = q.lower()
    is_german = _looks_german(q)

    topic = memory.get("current_topic")
    program = memory.get("current_program")
    degree_level = memory.get("current_degree_level")
    semester = memory.get("current_semester")

    context_phrase_en = _program_or_degree_phrase_en(program, degree_level)
    context_phrase_de = _program_or_degree_phrase_de(program, degree_level)

    asks_teaching_language = (
        "taught" in lower_q
        or "teaching language" in lower_q
        or "language of instruction" in lower_q
        or "unterrichtet" in lower_q
        or "unterrichtssprache" in lower_q
        or "auf englisch oder deutsch" in lower_q
        or "englisch oder deutsch" in lower_q
        or (("is it" in lower_q or "is this" in lower_q) and ("english" in lower_q or "german" in lower_q))
        or (("wird er" in lower_q or "wird sie" in lower_q or "wird es" in lower_q or "wird der" in lower_q) and ("englisch" in lower_q or "deutsch" in lower_q))
    )

    if asks_teaching_language:
        if is_german:
            return f"In welcher Sprache wird der Studiengang{context_phrase_de} an der HTW Berlin unterrichtet?"
        return f"What is the language of instruction{context_phrase_en} at HTW Berlin?"

    asks_language_proof = (
        "language" in lower_q
        or "english" in lower_q
        or "german" in lower_q
        or "proof" in lower_q
        or "certificate" in lower_q
        or "sprachnachweis" in lower_q
        or "sprachkenntnisse" in lower_q
        or "englischkenntnisse" in lower_q
        or "deutschkenntnisse" in lower_q
        or "nachweis" in lower_q
        or topic == "language_requirements"
    )

    if asks_language_proof:
        if is_german:
            return f"Welche Sprachanforderungen gelten{context_phrase_de} an der HTW Berlin?"
        return f"What are the language requirements{context_phrase_en} at HTW Berlin?"

    if "deadline" in lower_q or "frist" in lower_q or "bewerbungsfrist" in lower_q:
        if "enrollment" in lower_q or "enrolment" in lower_q or "einschreibung" in lower_q or topic == "enrollment_deadline":
            base_en = "What is the enrollment deadline"
            base_de = "Was ist die Einschreibefrist"
        else:
            base_en = "What is the application deadline"
            base_de = "Was ist die Bewerbungsfrist"

        if is_german:
            base = base_de + context_phrase_de
            if semester:
                base += f" für das {semester}"
            return f"{base} an der HTW Berlin?"

        base = base_en + context_phrase_en
        if semester:
            base += f" for the {semester}"
        return f"{base} at HTW Berlin?"

    if (
        "document" in lower_q or "documents" in lower_q or "unterlagen" in lower_q
        or "dokumente" in lower_q or "requirement" in lower_q or "requirements" in lower_q
        or "voraussetzungen" in lower_q or "zulassungsvoraussetzungen" in lower_q
        or "need" in lower_q or "brauche" in lower_q or "muss" in lower_q
        or "motivation letter" in lower_q or "motivationsschreiben" in lower_q
    ):
        if is_german:
            return f"Welche Unterlagen oder Voraussetzungen werden{context_phrase_de} an der HTW Berlin benötigt?"
        return f"What documents or requirements are needed{context_phrase_en} at HTW Berlin?"

    if "summer semester" in lower_q or "winter semester" in lower_q or "sommersemester" in lower_q or "wintersemester" in lower_q:
        if topic == "application_deadline":
            base_en = "What is the application deadline"
            base_de = "Was ist die Bewerbungsfrist"
        elif topic == "enrollment_deadline":
            base_en = "What is the enrollment deadline"
            base_de = "Was ist die Einschreibefrist"
        else:
            base_en = "What is the relevant information"
            base_de = "Welche relevanten Informationen gelten"

        if is_german:
            base = base_de + context_phrase_de
            if "sommersemester" in lower_q or "summer semester" in lower_q:
                base += " für das Sommersemester"
            elif "wintersemester" in lower_q or "winter semester" in lower_q:
                base += " für das Wintersemester"
            return f"{base} an der HTW Berlin?"

        base = base_en + context_phrase_en
        if "summer semester" in lower_q or "sommersemester" in lower_q:
            base += " for the summer semester"
        elif "winter semester" in lower_q or "wintersemester" in lower_q:
            base += " for the winter semester"
        return f"{base} at HTW Berlin?"

    if memory.get("last_standalone_query"):
        if is_german:
            return f"{q} (Kontext: {memory['last_standalone_query']})"
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
    _load_store_once()

    memory["turn_count"] = int(memory.get("turn_count") or 0) + 1
    memory["last_user_query"] = raw_query
    memory["last_standalone_query"] = standalone_query
    memory["current_topic"] = intent or memory.get("current_topic")

    raw_entities = extract_entities(raw_query)
    standalone_entities = extract_entities(standalone_query)

    merged_entities = {
        "program": entities.get("program") or raw_entities.get("program") or standalone_entities.get("program"),
        "degree_level": entities.get("degree_level") or raw_entities.get("degree_level") or standalone_entities.get("degree_level"),
        "semester": entities.get("semester") or raw_entities.get("semester") or standalone_entities.get("semester"),
    }

    if merged_entities.get("program"):
        memory["current_program"] = merged_entities["program"]

    if merged_entities.get("degree_level"):
        memory["current_degree_level"] = merged_entities["degree_level"]

    if merged_entities.get("semester"):
        memory["current_semester"] = merged_entities["semester"]

    if intent == "application_deadline":
        memory["current_deadline_type"] = "application_deadline"
    elif intent == "enrollment_deadline":
        memory["current_deadline_type"] = "enrollment_deadline"

    memory["last_answer_summary"] = (answer or "")[:300]

    _save_store()


def get_memory_snapshot(memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "turn_count": memory.get("turn_count"),
        "current_topic": memory.get("current_topic"),
        "current_program": memory.get("current_program"),
        "current_degree_level": memory.get("current_degree_level"),
        "current_semester": memory.get("current_semester"),
        "last_standalone_query": memory.get("last_standalone_query"),
    }
