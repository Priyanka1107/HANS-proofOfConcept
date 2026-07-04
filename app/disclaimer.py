"""
Disclaimer handling for HANS staff-facing email drafts.

The disclaimer text is stored outside the prompt and outside the generation
logic. This allows the wording to be changed without modifying generation code.

Default files:
    config/disclaimer.md
    config/disclaimer_de.md   (optional)

Optional environment overrides:
    DISCLAIMER_PATH=config/disclaimer.md
    DISCLAIMER_DE_PATH=config/disclaimer_de.md
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCLAIMER_PATH = PROJECT_ROOT / "config" / "disclaimer.md"
DEFAULT_DISCLAIMER_DE_PATH = PROJECT_ROOT / "config" / "disclaimer_de.md"


DEFAULT_GERMAN_DISCLAIMER = (
    "---\n\n"
    "Dieser Antwortentwurf wurde mit Unterstützung des HTW AI Navigation System erstellt "
    "und ist für die Prüfung durch Mitarbeitende vorbereitet. Er muss vor dem Versand "
    "von einer zuständigen Person geprüft und freigegeben werden. HANS-generierte Inhalte "
    "basieren auf offiziellen HTW-Berlin-Quellen zum Zeitpunkt der letzten Aktualisierung "
    "der Wissensbasis. Sie stellen keine rechtsverbindliche Auskunft dar und müssen bei "
    "Bedarf mit den aktuellen Programmseiten abgeglichen werden."
)


def get_disclaimer_path(reply_language: Optional[str] = None) -> Path:
    """
    Return the configured disclaimer file path.

    If reply_language starts with 'de', the German disclaimer path is used when
    configured or available. Otherwise the default English disclaimer path is used.

    If an environment path is relative, it is resolved from the project root.
    """
    lang = str(reply_language or "").lower()

    if lang.startswith("de"):
        configured_path = os.getenv("DISCLAIMER_DE_PATH", "").strip()
        if configured_path:
            path = Path(configured_path)
            return path if path.is_absolute() else PROJECT_ROOT / path

        if DEFAULT_DISCLAIMER_DE_PATH.exists():
            return DEFAULT_DISCLAIMER_DE_PATH

    configured_path = os.getenv("DISCLAIMER_PATH", "").strip()

    if not configured_path:
        return DEFAULT_DISCLAIMER_PATH

    path = Path(configured_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_disclaimer_text(reply_language: Optional[str] = None) -> str:
    """
    Load disclaimer text from a markdown or text file.

    Returns an empty string if the English disclaimer file is missing.
    For German output, returns config/disclaimer_de.md when available; otherwise
    falls back to a built-in German disclaimer.
    """
    path = get_disclaimer_path(reply_language)

    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    if str(reply_language or "").lower().startswith("de"):
        return DEFAULT_GERMAN_DISCLAIMER

    return ""


def append_disclaimer_to_draft(
    draft: str,
    disclaimer_text: Optional[str] = None,
    reply_language: Optional[str] = None,
) -> str:
    """
    Append the configured disclaimer to a generated staff draft.

    The function avoids duplicate insertion if the disclaimer is already present.
    """
    draft = (draft or "").rstrip()

    if disclaimer_text is None:
        disclaimer_text = load_disclaimer_text(reply_language=reply_language)

    disclaimer_text = (disclaimer_text or "").strip()

    if not disclaimer_text:
        return draft

    if disclaimer_text in draft:
        return draft

    return f"{draft}\n\n{disclaimer_text}"
