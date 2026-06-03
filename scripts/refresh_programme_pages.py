"""
Refresh official programme page cache for HANS PoC.

Creates:
    data/programme_official_pages.json

Why this exists:
    Some HTW programmes have programme-specific rules, especially fees.
    General HTW pages are not enough for questions about paid programmes
    such as PROITD or MPMD.

Run from project root:
    python scripts/refresh_programme_pages.py

Optional:
    python scripts/refresh_programme_pages.py --limit 10
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CATALOGUE_PATH = DATA_DIR / "programme_catalog.json"
OUTPUT_PATH = DATA_DIR / "programme_official_pages.json"


DEFAULT_EXTRA_PATHS = [
    "",
    "/en/",
    "/applying/",
    "/en/applying/",
    "/applying/fees-financing/",
    "/en/applying/fees-financing/",
    "/applying/faq/",
    "/en/applying/faq/",
    "/studying/everything-at-a-glance/",
    "/en/studying/everything-at-a-glance/",
    "/welcome/organise-your-finances/",
    "/en/welcome/organise-your-finances/",
    "/faq/finances-and-scholarships/",
    "/en/faq/finances-and-scholarships/",
]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def strip_html(raw: str) -> str:
    raw = html.unescape(raw or "")
    raw = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style\b[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<nav\b[^>]*>.*?</nav>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<footer\b[^>]*>.*?</footer>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def extract_title(raw: str) -> str:
    patterns = [
        r"<title[^>]*>(.*?)</title>",
        r"<h1[^>]*>(.*?)</h1>",
        r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']+)[\"']",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.I | re.S)
        if match:
            title = strip_html(match.group(1))
            title = re.sub(r"\s+", " ", title).strip()
            return title[:180]

    return ""


def clean_base_url(url: str) -> Optional[str]:
    if not url:
        return None

    parsed = urlparse(url)
    if not parsed.netloc.endswith("htw-berlin.de"):
        return None

    scheme = parsed.scheme or "https"
    host = parsed.netloc

    # Programme subdomain base.
    return f"{scheme}://{host}/"


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        item = str(item or "").strip()
        if not item:
            continue

        item = item.rstrip("/")

        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def candidate_urls_for_programme(programme: Dict[str, Any]) -> List[str]:
    urls: List[str] = []

    for key in ["url", "application_url"]:
        value = programme.get(key)
        if value:
            urls.append(str(value))

    base = clean_base_url(str(programme.get("url") or ""))
    if base:
        for path in DEFAULT_EXTRA_PATHS:
            urls.append(urljoin(base, path))

    # Some programme pages have known finance paths.
    name = str(programme.get("program_name", "")).lower()
    aliases = " ".join(str(a).lower() for a in programme.get("aliases", []) or [])

    if "project management and data science" in name or "mpmd" in aliases:
        urls.extend(
            [
                "https://mpmd.htw-berlin.de/",
                "https://mpmd.htw-berlin.de/applying/",
                "https://mpmd.htw-berlin.de/faq/finances-and-scholarships/",
            ]
        )

    if "professional it" in name or "proitd" in aliases:
        urls.extend(
            [
                "https://proitd.htw-berlin.de/",
                "https://proitd.htw-berlin.de/en/",
                "https://proitd.htw-berlin.de/applying/",
                "https://proitd.htw-berlin.de/en/applying/",
                "https://proitd.htw-berlin.de/applying/fees-financing/",
                "https://proitd.htw-berlin.de/applying/faq/",
                "https://proitd.htw-berlin.de/studying/everything-at-a-glance/",
                "https://proitd.htw-berlin.de/welcome/organise-your-finances/",
            ]
        )

    return unique_keep_order(urls)


def fetch_page(url: str) -> Optional[Dict[str, Any]]:
    headers = {
        "User-Agent": "HANS-PoC-Thesis-Research/1.0 (+local educational prototype)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
    except Exception:
        return None

    if response.status_code >= 400:
        return None

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return None

    raw = response.text
    text = strip_html(raw)

    if len(text) < 300:
        return None

    return {
        "url": url,
        "final_url": response.url,
        "status_code": response.status_code,
        "title": extract_title(raw),
        "text": text[:30000],
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    catalogue = read_json(CATALOGUE_PATH)

    if not isinstance(catalogue, list):
        raise FileNotFoundError(f"Programme catalogue not found or invalid: {CATALOGUE_PATH}")

    output: List[Dict[str, Any]] = []
    total_urls = 0

    for index, programme in enumerate(catalogue, start=1):
        if args.limit and index > args.limit:
            break

        program_name = str(programme.get("program_name", "") or "").strip()
        aliases = programme.get("aliases", []) or []
        degree = programme.get("degree")
        language = programme.get("language")
        study_format = programme.get("study_format")

        candidate_urls = candidate_urls_for_programme(programme)
        total_urls += len(candidate_urls)

        print(f"\n[{index}/{len(catalogue)}] {program_name}")
        print(f"Candidate URLs: {len(candidate_urls)}")

        for url in candidate_urls:
            print(f"  Fetching: {url}")
            page = fetch_page(url)

            if not page:
                print("    skipped")
                continue

            page.update(
                {
                    "program_name": program_name,
                    "aliases": aliases,
                    "degree": degree,
                    "language": language,
                    "study_format": study_format,
                    "source": "official_programme_page_cache",
                }
            )

            output.append(page)
            print(f"    ok: {len(page['text'])} chars")

            time.sleep(args.sleep)

    write_json(OUTPUT_PATH, output)

    print("\n" + "=" * 80)
    print("Programme official page cache refreshed")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Programmes in catalogue: {len(catalogue)}")
    print(f"Candidate URLs attempted: {total_urls}")
    print(f"Pages saved: {len(output)}")
    print("=" * 80)


if __name__ == "__main__":
    main()