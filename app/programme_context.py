"""
Programme context helper for HANS PoC.

This file connects data/programme_catalog.json to the live email assistant flow.

It does not hardcode admission facts.
It only detects programme names and adds programme context for retrieval/generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = PROJECT_ROOT / "data" / "programme_catalog.json"


def _load_catalogue() -> List[Dict[str, Any]]:
    if not CATALOGUE_PATH.exists():
        return []

    try:
        with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        return []

    return []


def _normalise(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9äöüß]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_programme(text: str) -> Optional[Dict[str, Any]]:
    """
    Match programme from catalogue using programme name and aliases.

    Returns the best matched catalogue entry or None.
    """
    catalogue = _load_catalogue()
    text_norm = _normalise(text)

    if not text_norm:
        return None

    best_match = None
    best_score = 0

    for item in catalogue:
        program_name = str(item.get("program_name", "")).strip()
        aliases = item.get("aliases", []) or []

        candidates = [program_name] + [str(a) for a in aliases]

        for candidate in candidates:
            candidate = str(candidate or "").strip()
            if not candidate:
                continue

            candidate_norm = _normalise(candidate)

            if not candidate_norm:
                continue

            # Avoid accidental tiny matches.
            if len(candidate_norm) < 4:
                continue

            score = 0

            # Exact phrase match.
            if candidate_norm in text_norm:
                score = len(candidate_norm)

            # Special abbreviation match, e.g. MPMD, PROITD.
            if candidate.isupper() and re.search(rf"\b{re.escape(candidate.lower())}\b", text_norm):
                score = max(score, 100 + len(candidate))

            if score > best_score:
                best_score = score
                best_match = item

    return best_match


def build_programme_context_block(email_text: str, subject: str = "") -> Dict[str, Any]:
    """
    Build a programme context block to inject before retrieval/generation.
    """
    combined_text = f"{subject}\n{email_text}"
    programme = match_programme(combined_text)

    if not programme:
        return {
            "matched": False,
            "program_name": "",
            "context_block": "",
            "url": "",
            "application_url": "",
        }

    program_name = str(programme.get("program_name", "") or "").strip()
    aliases = programme.get("aliases", []) or []
    degree = str(programme.get("degree", "") or "").strip()
    language = str(programme.get("language", "") or "").strip()
    study_format = str(programme.get("study_format", "") or "").strip()
    url = str(programme.get("url", "") or "").strip()
    application_url = str(programme.get("application_url", "") or "").strip()

    alias_text = ", ".join(str(a) for a in aliases if a)

    lines = [
        "Known programme context from local programme catalogue:",
        f"- Programme name: {program_name}",
    ]

    if alias_text:
        lines.append(f"- Known aliases: {alias_text}")

    if degree:
        lines.append(f"- Degree level: {degree}")

    if language:
        lines.append(f"- Catalogue language hint: {language}")

    if study_format:
        lines.append(f"- Catalogue study format hint: {study_format}")

    if url:
        lines.append(f"- Programme URL for staff verification: {url}")

    if application_url:
        lines.append(f"- Programme application URL for staff verification: {application_url}")

    lines.extend(
        [
            "",
            "Important instruction:",
            "- Use the programme name above when retrieving and drafting.",
            "- If the retrieved documents do not confirm a programme-specific fact, do not ask for the exact programme title again.",
            "- Instead, mention that the specific point should be reviewed by staff and include the programme URL for verification.",
        ]
    )

    return {
        "matched": True,
        "program_name": program_name,
        "context_block": "\n".join(lines),
        "url": url,
        "application_url": application_url,
    }


def enrich_email_text_with_programme_context(email_text: str, subject: str = "") -> Dict[str, Any]:
    """
    Return enriched email text and programme metadata.

    Use enriched_email_text instead of the original email_text in the existing flow.
    """
    context = build_programme_context_block(email_text=email_text, subject=subject)

    if not context["matched"]:
        return {
            **context,
            "enriched_email_text": email_text,
        }

    enriched_email_text = (
        context["context_block"]
        + "\n\nOriginal student email:\n"
        + str(email_text or "")
    )

    return {
        **context,
        "enriched_email_text": enriched_email_text,
    }