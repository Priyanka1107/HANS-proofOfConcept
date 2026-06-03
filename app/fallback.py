# app/fallback.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple


LOW_SCORE_THRESHOLD = 0.52
PROGRAM_QUERY_SCORE_THRESHOLD = 0.58


def top_score(docs: List[Dict[str, Any]]) -> float:
    if not docs:
        return 0.0
    try:
        return float(docs[0].get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _combined_doc_text(docs: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for d in docs or []:
        parts.append(str(d.get("title", "") or ""))
        parts.append(str(d.get("content", "") or ""))
        parts.append(str(d.get("source_url", "") or ""))
    return "\n".join(parts).lower()


def should_use_fallback(
    query: str,
    docs: List[Dict[str, Any]],
    intent: str,
    entities: Dict[str, Any] | None = None,
) -> Tuple[bool, str]:
    """
    Decide whether HANS should avoid normal generation and return a safe fallback.

    This is intentionally conservative. It does not replace generation for normal cases;
    it only catches weak retrieval or unsupported programme-specific cases.
    """
    entities = entities or {}

    if not docs:
        return True, "no_retrieved_documents"

    score = top_score(docs)
    if score < LOW_SCORE_THRESHOLD:
        return True, "low_retrieval_score"

    # If the user asks about a specific programme, require that the retrieved evidence
    # still contains that programme. This prevents generic HTW-wide answers being
    # presented as programme-specific answers.
    program = entities.get("program")
    if program:
        evidence_text = _combined_doc_text(docs)
        if program.lower() not in evidence_text and score < PROGRAM_QUERY_SCORE_THRESHOLD:
            return True, "program_not_supported_by_top_evidence"

    # Fee and duration information is often very specific. If retrieval is weak,
    # prefer safe fallback rather than forcing a generated answer.
    if intent in {"fees", "duration"} and score < PROGRAM_QUERY_SCORE_THRESHOLD:
        return True, f"weak_evidence_for_{intent}"

    return False, ""


def build_fallback_answer(
    query: str,
    intent: str,
    reason: str,
    entities: Dict[str, Any] | None = None,
) -> str:
    """Return a helpful but safe answer when evidence is not strong enough."""
    entities = entities or {}
    program = entities.get("program")

    program_text = f" for {program}" if program else ""

    if reason == "program_not_supported_by_top_evidence":
        return (
            f"I could not find enough programme-specific evidence{program_text} in the currently scraped HTW Berlin documents. "
            "The system found related HTW Berlin information, but it was not specific enough to answer this safely. "
            "Please check the official HTW Berlin programme page or application portal for confirmation."
        )

    if intent == "application_deadline":
        return (
            f"I could not find a clear application deadline{program_text} in the currently scraped HTW Berlin documents. "
            "This may mean the information is missing from the scraped data, or that the retrieval result was not specific enough. "
            "Please verify the deadline on the official HTW Berlin programme page or application portal."
        )

    if intent == "required_documents":
        return (
            f"I could not find a clear list of required documents{program_text} in the currently scraped HTW Berlin documents. "
            "The information may be available on a programme-specific page or in the application portal."
        )

    if intent == "language_requirements":
        return (
            f"I could not find enough reliable evidence about the language requirements{program_text} in the currently scraped documents. "
            "Please check the official programme page before relying on this information."
        )

    if intent == "fees":
        return (
            f"I could not find clear fee information{program_text} in the currently scraped HTW Berlin documents. "
            "Please confirm this on the official programme page or HTW Berlin fee information page."
        )

    if intent == "duration":
        return (
            f"I could not find clear programme duration information{program_text} in the currently scraped HTW Berlin documents. "
            "Please confirm this on the official programme page."
        )

    return (
        "I could not find enough reliable information in the currently scraped HTW Berlin documents to answer this confidently. "
        "The information may be missing from the scraped data, or the retrieved evidence may not be specific enough."
    )
