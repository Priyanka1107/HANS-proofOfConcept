"""
Build a clean HTW programme catalogue from scraped data.

Creates:
    data/programme_catalog.json

Purpose:
    The catalogue helps the HANS PoC recognise programme names dynamically
    from scraped data instead of hardcoding every programme inside the
    retrieval and generation logic.

Important design decision:
    This script does NOT hardcode programme facts such as deadlines,
    documents, fees, language proof, or admission rules.

    It only normalises programme names and aliases, for example:
        MPMD  -> Project Management and Data Science
        PROITD -> Professional IT Business and Digitalization

    Actual answers must still come from retrieved documents.

Sources used:
    1. Structured programme objects:
        data/objects/degree_program-*.json

    2. Programme subdomain pages:
        mpmd.htw-berlin.de
        proitd.htw-berlin.de
        conrem.htw-berlin.de
        cyber-security-business.htw-berlin.de
        etc.

The script avoids broad free-text extraction from full raw pages because that
created noisy entries in earlier versions.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OBJECTS_DIR = DATA_DIR / "objects"
SNAPSHOTS_DIR = DATA_DIR / "snapshots_raw"
OUTPUT_PATH = DATA_DIR / "programme_catalog.json"


BLOCKED_SUBDOMAINS = {
    "www",
    "rz",
    "wiki",
    "lsf",
    "moodle",
    "campus",
    "mail",
    "webmail",
    "idp",
    "bibliothek",
    "library",
    "mediathek",
    "chatbot",
    "webseiten",
    "events",
    "datenschutz",
    "kiwerkstatt",
    "projekte",
    "corporatedesign",
    "computermuseum",
    "hochschulsport",
    "gateway-webservices",
    "portal",
    "anleitungen",
    "bewerbung",
    "berndcms",
    "bitrix2",
    "stellenticket",
    "sport",
    "osa",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
}


BLOCKED_PATH_FRAGMENTS = {
    "/hochschule/personen",
    "/person/",
    "/personen/",
    "/international/pathways-abroad",
    "/international/wege-an-die-htw",
    "/studying-abroad",
    "/nominierung-und-bewerbung",
    "/hochschule/",
    "/campus/",
    "/service/",
    "/news/",
    "/veranstaltungen/",
    "/events/",
    "/wiki",
    "confluence",
    "/datenschutz",
    "/privacy",
    "/data-privacy",
    "/impressum",
    "/legal",
    "/kontakt",
    "/contact",
    "/login",
}


BAD_ALIASES = {
    "CAMPUS-STORIES",
    "HOCHSCHULSPORT",
    "MEDIATHEK",
    "STAT",
    "WIKI",
    "HTW-BERLIN",
    "CHATBOT",
    "EVENTS",
    "DATENSCHUTZ",
    "PROJEKTE",
    "WEBSEITEN",
    "RZ",
    "BITRIX2",
    "BERNDCMS",
    "ACCOUNT",
    "API",
    "ANLEITUNGEN",
    "BEWERBUNG",
    "PORTAL",
    "SPORT",
    "STELLENTICKET",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
}


# Entity-name normalisation only.
# Do not add deadlines, fees, admission rules, documents, or language proof here.
SUBDOMAIN_NAME_OVERRIDES = {
    "mpmd": "Project Management and Data Science",
    "proitd": "Professional IT Business and Digitalization",
    "conrem": "Construction and Real Estate Management",
    "cyber-security-business": "Cyber Security and Business",
    "quantitative-finance-data-science": "Quantitative Finance and Data Science",
    "international-business": "International Business",
    "mib": "International Business",
    "mbae": "Business Administration and Engineering",
    "ai-master": "Applied Computer Science",
    "ai": "Applied Computer Science",
}


SUBDOMAIN_ALIAS_OVERRIDES = {
    "mpmd": ["MPMD", "Project Management and Data Science"],
    "proitd": [
        "PROITD",
        "Pro ITD",
        "Professional IT and Digitalization",
        "Professional IT Business and Digitalization",
    ],
    "conrem": ["CONREM", "Construction and Real Estate Management"],
    "cyber-security-business": ["Cyber Security and Business", "Cybersecurity and Business"],
    "quantitative-finance-data-science": [
        "Quantitative Finance and Data Science",
        "Quantitative Finance Data Science",
    ],
    "international-business": ["International Business"],
    "mib": ["MIB", "International Business"],
    "mbae": ["MBAE", "Business Administration and Engineering"],
    "ai-master": ["AI-MASTER", "Applied Computer Science"],
    "ai": ["AI", "Applied Computer Science"],
}


PROGRAMME_KEYWORDS = {
    "accounting",
    "administration",
    "aeco",
    "apparel",
    "architecture",
    "artificial",
    "automotive",
    "building",
    "business",
    "civil",
    "clothing",
    "communication",
    "computer",
    "construction",
    "culture",
    "cyber",
    "data",
    "design",
    "digital",
    "digitalization",
    "digitalisation",
    "electrical",
    "engineering",
    "environment",
    "environmental",
    "estate",
    "finance",
    "game",
    "general",
    "health",
    "industrial",
    "informatics",
    "information",
    "international",
    "management",
    "mechanical",
    "media",
    "museum",
    "professional",
    "project",
    "public",
    "quantitative",
    "real",
    "renewable",
    "science",
    "security",
    "smart",
    "software",
    "sustainability",
    "systems",
    "technology",
}


BAD_NAME_PATTERNS = [
    r"\bdata privacy\b",
    r"\bprivacy notice\b",
    r"\bgeneral data protection\b",
    r"\badministration\b.*\bhuman resources\b",
    r"\bcareer international information\b",
    r"\bprospective students\b",
    r"\bapplication overview\b",
    r"\bclearing process\b",
    r"\bselection procedure\b",
    r"\bmore information\b",
    r"\blogin data\b",
    r"\bstudent service centre\b",
    r"\bcampus management\b",
    r"\binformation centre\b",
    r"\binternational office\b",
    r"\bleave of absence\b",
    r"\bhealth insurance\b",
    r"\btable of contents\b",
    r"\bplease note\b",
    r"\bthis does not apply\b",
    r"\btransfer\b.*\blegal section\b",
    r"\bpre-study internship\b",
    r"\bvirtual exchange\b",
    r"\badvanced master'?s programmes\b",
]


# Subdomain-derived names that are too short or not useful as main catalogue entries.
# These may be internal programme abbreviations, but without a clean full title they
# are unsafe to use as main programme names in staff drafts.
BAD_SUBDOMAIN_DERIVED_NAME_PATTERNS = [
    r"^[A-Z][a-z]{1,4} Bachelor$",
    r"^[A-Z][a-z]{1,4} Master$",
    r"^Bau Bachelor$",
    r"^Bau Master$",
    r"^Btk Bachelor$",
    r"^Btk Master$",
    r"^Ce Bachelor$",
    r"^Ce Master$",
    r"^Et Bachelor$",
    r"^Et Master$",
    r"^Fm Master$",
    r"^Gd Bachelor$",
    r"^Ge Bachelor$",
    r"^Geit Master$",
    r"^Ikt Bachelor$",
    r"^Ikt Master$",
    r"^Imi Bachelor$",
    r"^Imi Master$",
    r"^Lse Bachelor$",
    r"^Lse Master$",
    r"^Mb Bachelor$",
    r"^Mb Master$",
    r"^Md Bachelor$",
    r"^Md Master$",
    r"^Re Bachelor$",
    r"^Re Master$",
    r"^Wiko Bachelor$",
    r"^Wiko Master$",
    r"^Wiw Bachelor$",
    r"^Wiw Master$",
]


BAD_FINAL_PROGRAM_NAMES = {
    "Advanced Master's programmes",
    "Game Changer",
    "Health",
    "Islamic Culture",
    "Pre-study internship for Bachelor’s degree programmes",
    "Virtual Exchange in International Teaching",
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""


def read_json(path: Path) -> Any:
    raw = read_text(path)
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []

    for item in items:
        value = str(item).strip()
        if not value:
            continue

        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def strip_html(raw: str) -> str:
    raw = html.unescape(raw)
    raw = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style\b[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def extract_urls(raw: str) -> List[str]:
    urls = re.findall(r"https?://[^\s\"'<>]+", raw, flags=re.I)
    cleaned: List[str] = []

    for url in urls:
        url = html.unescape(url).strip().rstrip(").,;'\"")
        if "htw-berlin.de" in url:
            cleaned.append(url)

    return unique_keep_order(cleaned)


def host_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def subdomain_from_url(url: Optional[str]) -> str:
    host = host_from_url(url)
    if not host.endswith("htw-berlin.de"):
        return ""
    return host.split(".")[0].lower().strip()


def is_blocked_url(url: Optional[str]) -> bool:
    if not url:
        return False

    lower = url.lower()
    subdomain = subdomain_from_url(url)

    if subdomain in BLOCKED_SUBDOMAINS:
        return True

    return any(fragment in lower for fragment in BLOCKED_PATH_FRAGMENTS)


def is_programme_subdomain_url(url: Optional[str]) -> bool:
    if not url:
        return False

    host = host_from_url(url)
    if not host.endswith("htw-berlin.de"):
        return False

    subdomain = subdomain_from_url(url)
    if not subdomain:
        return False

    if subdomain in BLOCKED_SUBDOMAINS:
        return False

    if is_blocked_url(url):
        return False

    return True


def slug_to_title(slug: str) -> str:
    words = [w for w in re.split(r"[-_]+", slug) if w]
    title_words: List[str] = []

    for word in words:
        if word.lower() in {"it", "ai"}:
            title_words.append(word.upper())
        else:
            title_words.append(word.capitalize())

    return " ".join(title_words)


def clean_name(name: str) -> str:
    name = html.unescape(str(name))
    name = re.sub(r"\s+", " ", name).strip()

    replacements = [
        r"\bHTW Berlin\b",
        r"\bUniversity of Applied Sciences\b",
        r"\bBachelor'?s degree programme\b",
        r"\bMaster'?s degree programme\b",
        r"\bBachelor'?s programme\b",
        r"\bMaster'?s programme\b",
        r"\bDegree programme\b",
        r"\bStudy programme\b",
    ]

    for pattern in replacements:
        name = re.sub(pattern, "", name, flags=re.I).strip()

    for sep in [" | ", " – ", " — ", " - "]:
        if sep in name:
            parts = [p.strip() for p in name.split(sep) if p.strip()]
            good_parts = [p for p in parts if looks_like_programme_name_basic(p)]
            name = good_parts[0] if good_parts else parts[0]

    name = name.strip(" -–—|:;,.")
    name = re.sub(r"\s+", " ", name).strip()

    return name


def looks_like_programme_name_basic(name: str) -> bool:
    if not name:
        return False

    lower = name.lower().strip()

    if len(name) < 4 or len(name) > 90:
        return False

    for pattern in BAD_NAME_PATTERNS:
        if re.search(pattern, lower, flags=re.I):
            return False

    if lower in {
        "bachelor",
        "master",
        "application",
        "admission",
        "study programmes",
        "studies",
        "programme",
        "programmes",
        "overview",
        "contact",
        "health",
    }:
        return False

    if len(name.split()) > 8:
        return False

    for pattern in BAD_SUBDOMAIN_DERIVED_NAME_PATTERNS:
        if re.match(pattern, name):
            return False

    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", lower)
    if not words:
        return False

    has_keyword = any(word in PROGRAMME_KEYWORDS for word in words)
    has_title_shape = len(re.findall(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]+\b", name)) >= 1

    return has_keyword and has_title_shape


def looks_like_programme_name(name: str) -> bool:
    cleaned = clean_name(name)
    return looks_like_programme_name_basic(cleaned)


def normalise_key(name: str) -> str:
    value = clean_name(name).lower()
    value = re.sub(r"[^a-z0-9äöüß]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def make_acronym(name: str) -> Optional[str]:
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", name)
    ignore = {"and", "of", "in", "for", "the", "und", "der", "die", "das"}
    letters = [w[0].upper() for w in words if w.lower() not in ignore]
    acronym = "".join(letters)

    if 3 <= len(acronym) <= 10:
        return acronym

    return None


def clean_alias(alias: str) -> Optional[str]:
    alias = html.unescape(str(alias)).strip()
    alias = re.sub(r"\s+", " ", alias).strip()

    if not alias:
        return None

    if len(alias) > 90:
        return None

    if alias.upper() in BAD_ALIASES:
        return None

    return alias


def guess_degree_from_text(text: str, filename: str = "") -> Optional[str]:
    lower = f"{text} {filename}".lower()

    has_master = any(x in lower for x in ["master", "master's", "m.sc", "msc", "master of"])
    has_bachelor = any(x in lower for x in ["bachelor", "b.sc", "bsc", "bachelor of"])

    if has_master and not has_bachelor:
        return "Master"

    if has_bachelor and not has_master:
        return "Bachelor"

    if has_master and has_bachelor:
        return "Unknown"

    return None


def guess_language_from_text(text: str) -> Optional[str]:
    lower = text.lower()

    if any(x in lower for x in [
        "taught entirely in english",
        "taught in english",
        "english-taught",
        "english-language degree",
        "language of instruction english",
        "language: english",
    ]):
        return "English"

    if any(x in lower for x in [
        "taught in german",
        "german-taught",
        "german-language degree",
        "language of instruction german",
        "language: german",
    ]):
        return "German"

    return None


def guess_study_format_from_text(text: str) -> Optional[str]:
    lower = text.lower()

    if any(x in lower for x in ["distance learning", "online programme", "online study"]):
        return "Distance learning"

    if any(x in lower for x in ["on-campus", "on campus", "attendance programme", "full-time attendance"]):
        return "On-campus"

    return None


def extract_structured_strings(obj: Any) -> Dict[str, List[str]]:
    result = {
        "title_values": [],
        "all_text": [],
        "urls": [],
    }

    title_keys = {
        "title",
        "name",
        "program_name",
        "programme",
        "program",
        "degree_program",
        "degreeprogramme",
        "headline",
        "h1",
        "label",
    }

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k).lower())
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif isinstance(value, str):
            s = value.strip()
            if not s:
                return

            result["all_text"].append(s)

            if "http://" in s or "https://" in s:
                result["urls"].extend(extract_urls(s))

            if key in title_keys:
                result["title_values"].append(s)

    walk(obj)

    result["title_values"] = unique_keep_order(result["title_values"])
    result["all_text"] = unique_keep_order(result["all_text"])
    result["urls"] = unique_keep_order(result["urls"])

    return result


def extract_title_from_html(raw: str) -> Optional[str]:
    patterns = [
        r"<title[^>]*>(.*?)</title>",
        r"<h1[^>]*>(.*?)</h1>",
        r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<meta[^>]+name=[\"']title[\"'][^>]+content=[\"']([^\"']+)[\"']",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.I | re.S)
        if not match:
            continue

        value = strip_html(match.group(1))
        value = clean_name(value)

        if looks_like_programme_name(value):
            return value

    return None


def choose_best_url(urls: List[str]) -> Optional[str]:
    for url in urls:
        if is_programme_subdomain_url(url):
            return url.rstrip("/")

    for url in urls:
        if not is_blocked_url(url):
            return url.rstrip("/")

    return None


def choose_application_url(urls: List[str]) -> Optional[str]:
    for url in urls:
        lower = url.lower()
        if is_blocked_url(url):
            continue
        if any(x in lower for x in ["applying", "application", "bewerbung", "admission"]):
            return url.rstrip("/")
    return None


def is_final_entry_allowed(program_name: str, url: Optional[str]) -> bool:
    name = clean_name(program_name)

    if name in BAD_FINAL_PROGRAM_NAMES:
        return False

    if not looks_like_programme_name(name):
        return False

    for pattern in BAD_SUBDOMAIN_DERIVED_NAME_PATTERNS:
        if re.match(pattern, name):
            return False

    # Remove very broad category labels.
    if name in {"Design", "Health"}:
        return False

    # Keep strong programme-subdomain override entries.
    subdomain = subdomain_from_url(url) if url else ""
    if subdomain in SUBDOMAIN_NAME_OVERRIDES:
        return True

    return True


def add_entry(
    entries: Dict[str, Dict[str, Any]],
    *,
    program_name: str,
    aliases: Optional[Iterable[str]] = None,
    degree: Optional[str] = None,
    language: Optional[str] = None,
    study_format: Optional[str] = None,
    url: Optional[str] = None,
    application_url: Optional[str] = None,
    source_file: Optional[Path] = None,
    source_priority: int = 1,
) -> None:
    program_name = clean_name(program_name)

    if not is_final_entry_allowed(program_name, url):
        return

    key = normalise_key(program_name)
    if not key:
        return

    existing = entries.get(key)

    if existing is None:
        existing = {
            "program_name": program_name,
            "aliases": set(),
            "degree": None,
            "language": None,
            "study_format": None,
            "url": None,
            "application_url": None,
            "source": "scraped_programme_catalogue",
            "source_files": set(),
            "_priority": source_priority,
        }
        entries[key] = existing

    old_priority = int(existing.get("_priority", 0))
    stronger = source_priority >= old_priority

    if stronger:
        existing["_priority"] = source_priority

    alias_values: Set[str] = set()
    alias_values.add(program_name)

    if aliases:
        alias_values.update(str(a) for a in aliases if a)

    acronym = make_acronym(program_name)
    if acronym:
        alias_values.add(acronym)

    if url:
        subdomain = subdomain_from_url(url)
        if subdomain:
            alias_values.add(subdomain.upper())
            alias_values.update(SUBDOMAIN_ALIAS_OVERRIDES.get(subdomain, []))

    for alias in alias_values:
        cleaned = clean_alias(alias)
        if cleaned:
            existing["aliases"].add(cleaned)

    if degree and not existing["degree"]:
        existing["degree"] = degree

    if language and not existing["language"]:
        existing["language"] = language

    if study_format and not existing["study_format"]:
        existing["study_format"] = study_format

    if url and not is_blocked_url(url):
        if stronger or not existing["url"]:
            existing["url"] = url.rstrip("/")

    if application_url and not is_blocked_url(application_url):
        if stronger or not existing["application_url"]:
            existing["application_url"] = application_url.rstrip("/")

    if source_file:
        try:
            existing["source_files"].add(str(source_file.relative_to(PROJECT_ROOT)))
        except Exception:
            existing["source_files"].add(str(source_file))


def build_from_degree_program_objects(entries: Dict[str, Dict[str, Any]]) -> None:
    if not OBJECTS_DIR.exists():
        return

    for path in sorted(OBJECTS_DIR.glob("degree_program-*.json")):
        obj = read_json(path)
        if obj is None:
            continue

        extracted = extract_structured_strings(obj)
        all_text = " ".join(extracted["all_text"])
        urls = extracted["urls"]

        candidate_names: List[str] = []

        for value in extracted["title_values"]:
            cleaned = clean_name(value)
            if is_final_entry_allowed(cleaned, None):
                candidate_names.append(cleaned)

        stem = path.stem.replace("degree_program-", "")
        stem = re.sub(r"-(bachelor|master)$", "", stem, flags=re.I)
        filename_name = slug_to_title(stem)

        if is_final_entry_allowed(filename_name, None):
            candidate_names.append(filename_name)

        candidate_names = unique_keep_order(candidate_names)

        best_url = choose_best_url(urls)
        app_url = choose_application_url(urls)
        degree = guess_degree_from_text(all_text, path.name)
        language = guess_language_from_text(all_text)
        study_format = guess_study_format_from_text(all_text)

        for name in candidate_names:
            add_entry(
                entries,
                program_name=name,
                aliases=[],
                degree=degree,
                language=language,
                study_format=study_format,
                url=best_url,
                application_url=app_url,
                source_file=path,
                source_priority=10,
            )


def build_from_programme_subdomains(entries: Dict[str, Dict[str, Any]]) -> None:
    if not SNAPSHOTS_DIR.exists():
        return

    grouped: Dict[str, Dict[str, Any]] = {}

    for path in sorted(SNAPSHOTS_DIR.glob("*.html")):
        raw = read_text(path)
        if not raw.strip():
            continue

        urls = extract_urls(raw)
        programme_urls = [url for url in urls if is_programme_subdomain_url(url)]

        domains = re.findall(r"([a-z0-9-]+\.htw-berlin\.de)", raw, flags=re.I)
        for domain in domains:
            candidate_url = f"https://{domain.lower()}"
            if is_programme_subdomain_url(candidate_url):
                programme_urls.append(candidate_url)

        programme_urls = unique_keep_order(programme_urls)

        for url in programme_urls:
            subdomain = subdomain_from_url(url)
            if not subdomain:
                continue

            group = grouped.setdefault(
                subdomain,
                {
                    "urls": set(),
                    "titles": [],
                    "texts": [],
                    "source_files": set(),
                },
            )

            group["urls"].add(url.rstrip("/"))

            title = extract_title_from_html(raw)
            if title:
                group["titles"].append(title)

            group["texts"].append(strip_html(raw[:8000]))

            try:
                group["source_files"].add(path.relative_to(PROJECT_ROOT))
            except Exception:
                group["source_files"].add(path)

    for subdomain, group in grouped.items():
        urls = sorted(group["urls"])
        texts = " ".join(group["texts"])
        source_files = sorted(group["source_files"], key=lambda x: str(x))

        best_url = choose_best_url(urls)
        app_url = choose_application_url(urls)

        if subdomain in SUBDOMAIN_NAME_OVERRIDES:
            program_name = SUBDOMAIN_NAME_OVERRIDES[subdomain]
        else:
            valid_titles = [
                clean_name(t)
                for t in group["titles"]
                if is_final_entry_allowed(t, best_url)
            ]

            if valid_titles:
                program_name = valid_titles[0]
            else:
                program_name = slug_to_title(subdomain)

        if not is_final_entry_allowed(program_name, best_url):
            continue

        aliases = SUBDOMAIN_ALIAS_OVERRIDES.get(subdomain, [])
        aliases = list(aliases) + [subdomain.upper()]

        degree = guess_degree_from_text(texts)
        language = guess_language_from_text(texts)
        study_format = guess_study_format_from_text(texts)

        add_entry(
            entries,
            program_name=program_name,
            aliases=aliases,
            degree=degree,
            language=language,
            study_format=study_format,
            url=best_url,
            application_url=app_url,
            source_file=Path(str(source_files[0])) if source_files else None,
            source_priority=8,
        )


def finalise_entries(entries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    final: List[Dict[str, Any]] = []

    for entry in entries.values():
        name = clean_name(str(entry.get("program_name", "")))
        url = entry.get("url")

        if not is_final_entry_allowed(name, url):
            continue

        aliases = []
        for alias in sorted(entry.get("aliases", set()), key=lambda x: str(x).lower()):
            cleaned = clean_alias(str(alias))
            if not cleaned:
                continue

            if cleaned == name:
                aliases.append(cleaned)
            elif re.fullmatch(r"[A-Z0-9-]{2,20}", cleaned):
                aliases.append(cleaned)
            elif looks_like_programme_name(cleaned):
                aliases.append(cleaned)

        aliases = unique_keep_order(aliases)

        if name not in aliases:
            aliases.insert(0, name)

        final.append(
            {
                "program_name": name,
                "aliases": aliases,
                "degree": entry.get("degree"),
                "language": entry.get("language"),
                "study_format": entry.get("study_format"),
                "url": entry.get("url"),
                "application_url": entry.get("application_url"),
                "source": "scraped_programme_catalogue",
                "source_files": sorted(str(s) for s in entry.get("source_files", set()))[:10],
            }
        )

    merged: Dict[str, Dict[str, Any]] = {}

    for item in final:
        key = normalise_key(str(item["program_name"]))

        if key not in merged:
            merged[key] = item
            continue

        existing = merged[key]

        existing["aliases"] = unique_keep_order(
            list(existing.get("aliases", [])) + list(item.get("aliases", []))
        )

        for field in ["degree", "language", "study_format", "url", "application_url"]:
            if not existing.get(field) and item.get(field):
                existing[field] = item.get(field)

        existing["source_files"] = unique_keep_order(
            list(existing.get("source_files", [])) + list(item.get("source_files", []))
        )[:10]

    output = list(merged.values())
    output.sort(key=lambda x: str(x["program_name"]).lower())

    return output


def build_catalogue() -> List[Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}

    build_from_degree_program_objects(entries)
    build_from_programme_subdomains(entries)

    return finalise_entries(entries)


def print_check(catalogue: List[Dict[str, Any]], term: str) -> None:
    term_lower = term.lower()
    matches = []

    for item in catalogue:
        name = str(item.get("program_name", ""))
        aliases = [str(a) for a in item.get("aliases", [])]

        if term_lower in name.lower():
            matches.append(item)
            continue

        if any(term_lower == alias.lower() for alias in aliases):
            matches.append(item)
            continue

        if any(term_lower in alias.lower() for alias in aliases):
            matches.append(item)

    print()
    print(f"Check: {term}")

    if not matches:
        print("  Not found")
        return

    for item in matches[:8]:
        print(f"  Found: {item.get('program_name')}")
        print(f"  Aliases: {', '.join(item.get('aliases', [])[:10])}")
        print(f"  Degree: {item.get('degree')}")
        print(f"  Language: {item.get('language')}")
        print(f"  Study format: {item.get('study_format')}")
        print(f"  URL: {item.get('url')}")
        print(f"  Application URL: {item.get('application_url')}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    catalogue = build_catalogue()
    write_json(OUTPUT_PATH, catalogue)

    print("=" * 80)
    print("Clean programme catalogue built")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Entries: {len(catalogue)}")
    print("=" * 80)

    checks = [
        "Project Management and Data Science",
        "MPMD",
        "Professional IT Business and Digitalization",
        "Professional IT and Digitalization",
        "PROITD",
        "Construction and Real Estate Management",
        "CONREM",
        "Quantitative Finance and Data Science",
        "Cyber Security and Business",
        "Cybersecurity and Business",
        "International Business",
        "Health",
        "Bau Bachelor",
        "Btk Master",
        "Pre-study internship",
        "Virtual Exchange",
    ]

    for term in checks:
        print_check(catalogue, term)


if __name__ == "__main__":
    main()