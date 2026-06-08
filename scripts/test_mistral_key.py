"""
Small local test for the Mistral API key.

Run:
    python scripts/test_mistral_key.py

This script only checks whether the Mistral key and model can produce a short response.
It does not use student data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> int:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    model = os.getenv("GENERATION_MODEL", "mistral-small-latest").strip()

    if not api_key:
        print("ERROR: MISTRAL_API_KEY is missing in .env")
        return 1

    print(f"Testing Mistral model: {model}")

    client = Mistral(api_key=api_key)

    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a short test assistant.",
            },
            {
                "role": "user",
                "content": "Reply with exactly: Mistral key works.",
            },
        ],
        temperature=0.0,
        max_tokens=30,
    )

    content = response.choices[0].message.content
    print("Response:")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())