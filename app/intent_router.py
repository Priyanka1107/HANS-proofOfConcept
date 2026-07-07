# app/intent_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class RoutedIntent:
    intent: str
    confidence: float
    reason: str
    degree_level: Optional[str] = None


def _has_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def route_user_intent(text: str) -> RoutedIntent:
    """
    High-level intent router for the Email Assistant.

    This separates broad programme catalogue questions from admissions/application
    questions before topic detection and retrieval. It prevents catalogue-style
    questions from falling back into application_process / uni-assist answers,
    while leaving explicit application-route questions in admissions_support.
    """
    t = (text or "").strip().lower()

    programme_words = [
        r"program(?:me)?s?",
        r"degree\s+program(?:me)?s?",
        r"study\s+program(?:me)?s?",
        r"studieng[aä]nge",
        r"studienangebot",
    ]

    overview_words = [
        r"\bhow many\b",
        r"\bwhat\s+(all\s+)?",
        r"\bwhich\b",
        r"\blist\b",
        r"\boffered\b",
        r"\bavailable\b",
        r"\boverview\b",
        r"\bcatalog(?:ue)?\b",
        r"\bangeboten\b",
        r"\bwelche\b",
        r"\bwie viele\b",
        r"\bübersicht\b",
        r"\bliste\b",
    ]

    application_words = [
        r"\bapply\b",
        r"\bapplication\b",
        r"\bdeadline\b",
        r"\bfee\b",
        r"\buni-assist\b",
        r"\buni\s+assist\b",
        r"\bdocuments?\b",
        r"\brequirements?\b",
        r"\benglish proof\b",
        r"\blanguage proof\b",
        r"\bbewerb",
        r"\bbewerbungsfrist\b",
        r"\bunterlagen\b",
        r"\bgebühren\b",
        r"\bzulassungsvoraussetzungen\b",
    ]

    degree_level = None
    if re.search(r"\bmaster|master'?s|masterstudieng", t, flags=re.IGNORECASE):
        degree_level = "master"
    elif re.search(r"\bbachelor|bachelorstudieng", t, flags=re.IGNORECASE):
        degree_level = "bachelor"

    is_programme_question = _has_any(programme_words, t)
    is_overview_question = _has_any(overview_words, t)
    is_application_question = _has_any(application_words, t)

    if is_programme_question and is_overview_question and not is_application_question:
        return RoutedIntent(
            intent="programme_overview",
            confidence=0.9,
            reason="Broad programme list/count/overview question without explicit application-process request.",
            degree_level=degree_level,
        )

    return RoutedIntent(
        intent="admissions_support",
        confidence=0.7,
        reason="Default route for admissions, programme-specific, or application-related support.",
        degree_level=degree_level,
    )
