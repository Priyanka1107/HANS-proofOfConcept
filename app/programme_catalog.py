# app/programme_catalog.py
"""
Programme catalogue utilities for the HANS PoC.

Purpose:
- Avoid hardcoding programme names in the email assistant.
- Load programme names and URLs from data/programme_catalog.json.
- Match the programme mentioned in an incoming email using exact, alias and fuzzy matching.

The catalogue is built by scripts/build_programme_catalog.py from the scraped data.
If the catalogue is missing, this module safely returns no match and the PoC continues
with the existing general retrieval behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import difflib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "programme_catalog.json"


@dataclass
class ProgrammeMatch:
    name: str
    aliases: List[str]
    degree: Optional[str] = None
    language: Optional[str] = None
    study_format: Optional[str] = None
    url: Optional[str] = None
    application_url: Optional[str] = None
    score: float = 0.0
    source: str = "catalog"

    def to_context_fields(self) -> Dict[str, Optional[str]]:
        return {
            "target_program": self.name,
            "target_program_url": self.url,
            "target_program_application_url": self.application_url,
            "target_program_match_score": f"{self.score:.2f}",
            "target_program_source": self.source,
            "catalog_degree": self.degree,
            "catalog_language": self.language,
            "catalog_study_format": self.study_format,
        }


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9äöüß]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_programme_catalog(path: Optional[str] = None) -> List[Dict[str, Any]]:
    catalog_path = Path(path) if path else DEFAULT_CATALOG_PATH
    if not catalog_path.exists():
        return []
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("programmes", []) or []
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def match_programme_from_catalog(text: str, *, min_score: float = 0.78) -> Optional[ProgrammeMatch]:
    """
    Match a programme name mentioned in free text.

    Matching order:
    1. exact alias/name phrase match in the email text
    2. fuzzy match against longer phrases in the email

    This is intentionally conservative. If the score is weak, return None and
    let the answer be marked for review instead of forcing a wrong programme.
    """
    raw = text or ""
    norm_text = _normalize(raw)
    if not norm_text:
        return None

    catalog = load_programme_catalog()
    best: Optional[ProgrammeMatch] = None

    for item in catalog:
        name = item.get("program_name") or item.get("name") or ""
        if not name:
            continue
        aliases = list(dict.fromkeys([name] + item.get("aliases", [])))
        for alias in aliases:
            alias_norm = _normalize(alias)
            if not alias_norm:
                continue
            # Exact phrase/abbreviation match.
            if re.search(rf"\b{re.escape(alias_norm)}\b", norm_text):
                score = 1.0 if len(alias_norm) > 4 else 0.92
                candidate = ProgrammeMatch(
                    name=name,
                    aliases=aliases,
                    degree=item.get("degree"),
                    language=item.get("language"),
                    study_format=item.get("study_format"),
                    url=item.get("url"),
                    application_url=item.get("application_url"),
                    score=score,
                    source=item.get("source", "catalog"),
                )
                if best is None or candidate.score > best.score:
                    best = candidate

    if best:
        return best

    # Fuzzy fallback for non-exact spelling. Use only for multi-word programme names.
    tokens = norm_text.split()
    windows: List[str] = []
    for size in range(3, min(9, len(tokens)) + 1):
        for i in range(0, len(tokens) - size + 1):
            windows.append(" ".join(tokens[i:i+size]))

    for item in catalog:
        name = item.get("program_name") or item.get("name") or ""
        aliases = list(dict.fromkeys([name] + item.get("aliases", [])))
        for alias in aliases:
            alias_norm = _normalize(alias)
            if len(alias_norm.split()) < 2:
                continue
            for window in windows:
                score = difflib.SequenceMatcher(None, alias_norm, window).ratio()
                if score >= min_score and (best is None or score > best.score):
                    best = ProgrammeMatch(
                        name=name,
                        aliases=aliases,
                        degree=item.get("degree"),
                        language=item.get("language"),
                        study_format=item.get("study_format"),
                        url=item.get("url"),
                        application_url=item.get("application_url"),
                        score=score,
                        source=item.get("source", "catalog_fuzzy"),
                    )

    return best


def programme_reference_lines(context: Dict[str, Optional[str]]) -> List[str]:
    """Return staff verification links for the matched programme."""
    lines: List[str] = []
    program = context.get("target_program")
    url = context.get("target_program_url")
    app_url = context.get("target_program_application_url")

    if program and url:
        lines.append(f"- Programme page ({program}): {url}")
    if program and app_url and app_url != url:
        lines.append(f"- Programme application page ({program}): {app_url}")
    return lines


def programme_query_terms(context: Dict[str, Optional[str]]) -> str:
    """Build retrieval-enrichment text for the matched programme."""
    parts: List[str] = []
    if context.get("target_program"):
        parts.append(str(context["target_program"]))
    if context.get("target_program_url"):
        parts.append(str(context["target_program_url"]))
    if context.get("target_program_application_url"):
        parts.append(str(context["target_program_application_url"]))
    if context.get("catalog_degree"):
        parts.append(f"catalogue degree {context['catalog_degree']}")
    if context.get("catalog_language"):
        parts.append(f"catalogue language {context['catalog_language']}")
    if context.get("catalog_study_format"):
        parts.append(f"catalogue study format {context['catalog_study_format']}")
    return ". ".join(parts)
