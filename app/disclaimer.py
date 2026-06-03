"""
Disclaimer handling for HANS staff-facing email drafts.

The disclaimer text is stored outside the prompt and outside the generation
logic. This allows the wording to be changed without modifying Python code.

Default file:
    config/disclaimer.md

Optional environment override:
    DISCLAIMER_PATH=config/disclaimer.md
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCLAIMER_PATH = PROJECT_ROOT / "config" / "disclaimer.md"


def get_disclaimer_path() -> Path:
    """
    Return the configured disclaimer file path.

    If DISCLAIMER_PATH is relative, it is resolved from the project root.
    """
    configured_path = os.getenv("DISCLAIMER_PATH", "").strip()

    if not configured_path:
        return DEFAULT_DISCLAIMER_PATH

    path = Path(configured_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_disclaimer_text() -> str:
    """
    Load disclaimer text from a markdown or text file.

    Returns an empty string if the file is missing or cannot be read.
    This keeps the PoC running during local testing.
    """
    path = get_disclaimer_path()

    if not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def append_disclaimer_to_draft(
    draft: str,
    disclaimer_text: Optional[str] = None,
) -> str:
    """
    Append the configured disclaimer to a generated staff draft.

    The function avoids duplicate insertion if the disclaimer is already present.
    """
    draft = (draft or "").rstrip()

    if disclaimer_text is None:
        disclaimer_text = load_disclaimer_text()

    disclaimer_text = (disclaimer_text or "").strip()

    if not disclaimer_text:
        return draft

    if disclaimer_text in draft:
        return draft

    return f"{draft}\n\n{disclaimer_text}"