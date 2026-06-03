"""
Programme-specific official evidence helper.

Loads:
    data/programme_official_pages.json

Purpose:
    Add official programme-page snippets to retrieval results when the student
    asks programme-specific questions such as fees, work experience, language
    proof, documents, or study format.

This avoids answering specific programme questions from overly general HTW pages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "programme_official_pages.json"


TOPIC_KEYWORDS = {
    "tuition_fees": [
        "tuition",
        "tuition fees",
        "programme fee",
        "program fee",
        "fees",
        "financing",
        "instalment",
        "installment",
        "15,750",
        "15.750",
        "17,600",
        "17.600",
        "5,250",
        "5.250",
        "4,400",
        "4.400",
    ],
    "fees": [
        "tuition",
        "tuition fees",
        "semester fees",
        "programme fee",
        "fees",
        "financing",
        "instalment",
        "installment",
    ],
    "application_fee": [
        "application fee",
        "processing fee",
        "uni-assist",
        "handling fees",
        "processing costs",
    ],
    "work_experience": [
        "work experience",
        "professional experience",
        "relevant professional experience",
        "one year",
        "at least one year",
    ],
    "english_language_requirements": [
        "english",
        "english proficiency",
        "proof of english",
        "ielts",
        "toefl",
        "toeic",
        "cambridge",
        "language proof",
    ],
    "german_language_requirements": [
        "german",
        "proof of german",
        "german language",
        "language proof",
    ],
    "language_of_instruction": [
        "language of instruction",
        "taught in english",
        "taught entirely in english",
        "english-language",
        "english language",
    ],
    "study_format": [
        "on campus",
        "on-campus",
        "online",
        "distance learning",
        "study format",
        "attendance",
    ],
    "required_documents": [
        "documents",
        "application documents",
        "transcript",
        "certificate",
        "upload",
        "proof",
    ],
    "admission_requirements": [
        "admission requirements",
        "required qualifications",
        "bachelor",
        "ects",
        "credit points",
        "requirements",
    ],
    "application_route": [
        "apply",
        "application",
        "application form",
        "application portal",
        "uni-assist",
        "hochschulstart",
    ],
    "application_deadline": [
        "deadline",
        "application period",
        "apply by",
        "application deadline",
    ],
}


def _load_cache() -> List[Dict[str, Any]]:
    if not CACHE_PATH.exists():
        return []

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
    except Exception:
        return []

    return []


def _normalise(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9äöüß]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _programme_matches_page(context: Dict[str, Any], page: Dict[str, Any]) -> bool:
    target_program = _normalise(context.get("target_program") or context.get("matched_programme") or "")
    target_url = str(
        context.get("target_program_url")
        or context.get("matched_programme_url")
        or context.get("programme_url")
        or ""
    )

    page_program = _normalise(page.get("program_name") or "")
    page_url = str(page.get("url") or page.get("final_url") or "")

    if target_program and page_program and target_program == page_program:
        return True

    if target_program and page_program and (target_program in page_program or page_program in target_program):
        return True

    if target_url and page_url and _host(target_url) == _host(page_url):
        return True

    aliases = page.get("aliases", []) or []
    for alias in aliases:
        alias_norm = _normalise(str(alias))
        if alias_norm and target_program and alias_norm in target_program:
            return True

    return False


def _topic_terms(topic_id: str) -> List[str]:
    terms = list(TOPIC_KEYWORDS.get(topic_id, []))

    # Some scripts use general "fees" instead of tuition_fees.
    if topic_id in {"fees", "application_fee", "tuition_fees", "semester_contribution"}:
        terms.extend(TOPIC_KEYWORDS["tuition_fees"])

    return list(dict.fromkeys(terms))


def _find_best_snippet(text: str, terms: List[str], max_chars: int = 1200) -> str:
    if not text:
        return ""

    lower = text.lower()
    best_pos: Optional[int] = None

    for term in terms:
        term = term.lower()
        pos = lower.find(term)
        if pos >= 0:
            if best_pos is None or pos < best_pos:
                best_pos = pos

    if best_pos is None:
        return text[:max_chars].strip()

    start = max(0, best_pos - 350)
    end = min(len(text), best_pos + max_chars)

    snippet = text[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()

    return snippet


def _page_score(page: Dict[str, Any], topic_id: str) -> int:
    url = str(page.get("url") or page.get("final_url") or "").lower()
    title = str(page.get("title") or "").lower()
    text = str(page.get("text") or "").lower()
    terms = _topic_terms(topic_id)

    score = 0

    for term in terms:
        term_lower = term.lower()
        if term_lower in title:
            score += 5
        if term_lower in url:
            score += 5
        if term_lower in text:
            score += 2

    if topic_id in {"tuition_fees", "fees", "semester_contribution"}:
        if "fees-financing" in url or "finances" in url or "organise-your-finances" in url:
            score += 15
        if re.search(r"\b(15[,.]750|17[,.]600|5[,.]250|4[,.]400)\b", text):
            score += 20

    if topic_id == "work_experience":
        if "applying" in url or "faq" in url:
            score += 8
        if "professional experience" in text:
            score += 15

    if topic_id in {"admission_requirements", "required_documents", "application_route"}:
        if "applying" in url:
            score += 10

    if topic_id in {"english_language_requirements", "german_language_requirements", "language_of_instruction"}:
        if "english" in text or "language" in text:
            score += 8

    return score


def get_official_programme_docs(
    context: Dict[str, Any],
    topic_id: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """
    Return programme-specific official page snippets as retrieval documents.
    """
    if not context.get("target_program") and not context.get("matched_programme"):
        return []

    cache = _load_cache()
    if not cache:
        return []

    matching_pages = [
        page
        for page in cache
        if _programme_matches_page(context, page)
    ]

    if not matching_pages:
        return []

    scored = [
        (page, _page_score(page, topic_id))
        for page in matching_pages
    ]

    scored.sort(key=lambda item: item[1], reverse=True)

    selected = [
        page
        for page, score in scored
        if score > 0
    ][:limit]

    if not selected:
        selected = [page for page, score in scored[:limit]]

    docs: List[Dict[str, Any]] = []
    terms = _topic_terms(topic_id)

    for index, page in enumerate(selected, start=1):
        text = str(page.get("text") or "")
        snippet = _find_best_snippet(text, terms)

        if not snippet:
            continue

        url = str(page.get("final_url") or page.get("url") or "")

        docs.append(
            {
                "id": f"official_programme_page::{page.get('program_name')}::{url}::{topic_id}",
                "title": page.get("title") or page.get("program_name") or "Official programme page",
                "content": snippet,
                "chunk_text": snippet,
                "source_url": url,
                "url": url,
                "object_type": "official_programme_page_cache",
                "object_id": page.get("program_name") or "",
                "score": 1.0,
                "metadata": {
                    "program_name": page.get("program_name"),
                    "topic_id": topic_id,
                    "retrieved_at": page.get("retrieved_at"),
                    "source": "official_programme_page_cache",
                },
            }
        )

    return docs