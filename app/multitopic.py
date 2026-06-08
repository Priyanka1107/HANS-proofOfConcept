# app/multitopic.py
"""
V4.6 Email Assistant utilities for the HANS PoC.

Design decision:
- Do NOT generate one full answer per detected topic.
- Instead:
    1. understand the email,
    2. detect the real topics,
    3. retrieve evidence per topic,
    4. generate ONE final staff-ready email draft.

This keeps the good fluency of the baseline while still measuring
multi-topic coverage for the thesis evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from app.config import config
from app.programme_catalog import match_programme_from_catalog, programme_query_terms, programme_reference_lines


# ---------------------------------------------------------------------
# Official external reference links used in staff-facing drafts
# ---------------------------------------------------------------------

# These links are added only as verification/reference links in the staff-facing
# draft. They do not replace retrieval from the HTW knowledge base.
UNI_ASSIST_HANDLING_FEES_URL = "https://www.uni-assist.de/en/how-to-apply/pay-all-fees/handling-fees/"
HOCHSCHULSTART_URL = "https://www.hochschulstart.de/"
ANABIN_URL = "https://anabin.kmk.org/"
DAAD_ADMISSIONS_DATABASE_URL = "https://www.daad.de/en/studying-in-germany/requirements/admission-database/"
HTW_APPLICATION_PORTAL_URL = "https://bewerbung.htw-berlin.de/"
HTW_ADMISSION_REQUIREMENTS_URL = "https://www.htw-berlin.de/en/studies/applications/admission-requirements/"


# ---------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------

FOLLOWUP_REVIEW_PATTERNS = [
    r"\byour previous (answer|reply|response)\b",
    r"\byou (said|told me|mentioned|wrote|explained)\b",
    r"\bi still (do not|don't) understand\b",
    r"\bi am still confused\b",
    r"\bthis does not answer\b",
    r"\bthis is not clear\b",
    r"\bthe answer was unclear\b",
    r"\bregarding your (answer|reply|response)\b",
    r"\bfollowing up\b",
]


def is_followup_email(text: str) -> bool:
    """
    Return True only when the email clearly refers to a previous HANS/staff answer.
    We do not flag every email with a session_id, because a related new question
    can still be drafted for staff review.
    """
    text = (text or "").lower()
    return any(re.search(p, text) for p in FOLLOWUP_REVIEW_PATTERNS)


def build_followup_flag_message(email_text: str) -> str:
    return (
        "This message appears to be a follow-up to a previous response.\n\n"
        "Please review the previous communication before replying. "
        "A new automatic draft was not generated because the student may be asking "
        "for clarification or correction of an earlier answer.\n\n"
        f"Message preview:\n{(email_text or '')[:500]}"
    )


# ---------------------------------------------------------------------
# Email context extraction
# ---------------------------------------------------------------------

# Programme names are now matched from data/programme_catalog.json.
# This avoids hardcoding every HTW course name in the code.
# A small fallback list is kept only for backward compatibility when the catalogue
# has not been built yet. Do not add new course names here unless absolutely needed.
PROGRAM_PATTERNS = {
    "International Business": [r"\binternational business\b"],
    "Cybersecurity and Business": [r"\bcybersecurity and business\b", r"\bcyber security and business\b"],
}


COUNTRY_PATTERNS = {
    "India": r"\bindia\b",
    "Pakistan": r"\bpakistan\b",
    "Turkey": r"\bturkey\b",
    "Brazil": r"\bbrazil\b",
    "Portugal": r"\bportugal\b",
    "South Korea": r"\bsouth korea\b",
    "Morocco": r"\bmorocco\b",
    "France": r"\bfrench\b|\bfrance\b",
    "Spain": r"\bspain\b|\bspanish\b",
}
EU_CITIZENSHIP_PATTERNS = [
    r"\beu citizen\b",
    r"\beu national\b",
    r"\beea citizen\b",
    r"\beea national\b",
    r"\bcitizen of (an )?eu\b",
    r"\bcitizen of (an )?eea\b",

    # Common EU nationality wording
    r"\bfrench citizen\b",
    r"\bfrench national\b",
    r"\bcitizen of france\b",
    r"\bdual citizen.*french\b",
    r"\bfrench.*dual citizen\b",
    r"\bfrench and moroccan\b",
    r"\bmoroccan and french\b",
]

NON_EU_CITIZENSHIP_PATTERNS = [
    r"\bnon[- ]eu\b",
    r"\bnon eu\b",
    r"\bnot (an )?eu citizen\b",
    r"\bnot (an )?eea citizen\b",
]


def _find_first(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_email_context(email_text: str) -> Dict[str, Optional[str]]:
    """
    Separate background profile from the target study goal.
    This prevents:
        "I completed my Bachelor's degree"
    from being interpreted as:
        "I am applying for a Bachelor programme".
    """
    text = email_text or ""
    lower = text.lower()

    context: Dict[str, Optional[str]] = {
        "student_name": None,
        "previous_degree": None,
        "target_degree": None,
        "target_program": None,
        "country": None,
        "citizenship_group": None,  # EU/EEA, non-EU, or unknown
        "residence_country": None,
        "target_program_url": None,
        "target_program_application_url": None,
        "target_program_match_score": None,
        "target_program_source": None,
        "catalog_degree": None,
        "catalog_language": None,
        "catalog_study_format": None,
    }

    # Name: only simple first-name extraction for email greeting.
    name_match = re.search(r"\bmy name is\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)", text)
    if name_match:
        context["student_name"] = name_match.group(1)

    # Previous/current education
    if (
        re.search(r"(completed|completing|finishing|have)\s+my\s+bachelor", lower)
        or "bachelor's degree" in lower
        or "bachelors degree" in lower
        or re.search(r"\bbba\b", lower)
        or re.search(r"\bfinal year in bba\b", lower)
    ):
        context["previous_degree"] = "Bachelor"
    if re.search(r"(completed|completing|finishing|have)\s+my\s+master", lower) or "master's degree" in lower:
        context["previous_degree"] = "Master"
    if "high school" in lower or "baccalaureate" in lower or "school certificate" in lower:
        context["previous_degree"] = "School leaving certificate"

    # Target degree. Prefer explicit "apply/interested in ... Master's/Bachelor's programme".
    if re.search(r"(apply|applying|interested|want|would like).{0,80}(master|master's|master’s)", lower):
        context["target_degree"] = "Master"
    elif re.search(r"(master|master's|master’s).{0,80}(programme|program|degree)", lower):
        context["target_degree"] = "Master"

    if re.search(r"(apply|applying|interested|want|would like).{0,80}(bachelor|bachelor's|bachelor’s)", lower):
        context["target_degree"] = "Bachelor"
    elif re.search(r"(bachelor|bachelor's|bachelor’s).{0,80}(programme|program)", lower):
        # Only use this as target degree if the sentence talks about a programme,
        # not just a completed previous Bachelor's degree.
        if not re.search(r"(completed|completing|finishing|have).{0,40}(bachelor|bachelor's|bachelor’s)\s+degree", lower):
            context["target_degree"] = "Bachelor"

    # Programme: first use the dynamic programme catalogue built from scraped data.
    # This is more scalable than adding course names manually in code.
    programme_match = match_programme_from_catalog(text)
    if programme_match:
        context.update(programme_match.to_context_fields())
        # If the email did not clearly state Bachelor/Master but the catalogue knows it,
        # use the catalogue degree as a soft target degree.
        if not context.get("target_degree") and programme_match.degree in {"Bachelor", "Master"}:
            context["target_degree"] = programme_match.degree
    else:
        # Backward-compatible fallback for very common programmes if the catalogue
        # has not yet been generated.
        for program, patterns in PROGRAM_PATTERNS.items():
            if _find_first(patterns, lower):
                context["target_program"] = program
                context["target_program_source"] = "fallback_pattern"
                context["target_program_match_score"] = "0.70"
                break

    # Country / background. Keep this separate from citizenship when possible.
    for country, pattern in COUNTRY_PATTERNS.items():
        if re.search(pattern, lower):
            context["country"] = country
            break

    # Citizenship category for application route. This matters especially for
    # Bachelor first-semester applications where uni-assist, Hochschulstart and
    # the HTW portal can depend on applicant category.
    # Citizenship category for application route.
    # Important: citizenship is not the same as residence country.
    # Example: a French citizen residing in Morocco is still an EU/EEA citizen.
    if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in EU_CITIZENSHIP_PATTERNS):
        context["citizenship_group"] = "EU/EEA"
    elif any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in NON_EU_CITIZENSHIP_PATTERNS):
        context["citizenship_group"] = "non-EU"

    residence_match = re.search(
        r"\b(?:living|residing|currently living|currently residing)\s+in\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\- ]+)",
        text,
        flags=re.IGNORECASE,
    )
    if residence_match:
        context["residence_country"] = residence_match.group(1).strip()

    return context


# ---------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------

TOPIC_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "application_before_graduation": {
        "label": "Application before graduation",
                "patterns": [
            r"\b(final|provisional|pending).{0,60}(transcript|certificate|degree|results|grades)",
            r"\b(transcript|certificate|degree|results|grades).{0,60}(pending|not yet available|not available yet|delayed)",
            r"\bbefore (receiving|graduation|graduating|getting).{0,40}(final|transcript|certificate|degree|results|grades)?",
            r"\bapply before (graduation|graduating|receiving my final|getting my final)",
            r"\bstill waiting for my final",
            r"\bfinal year\b",
            r"\bfinal semester results\b",
            r"\bfinal semester grades\b",
            r"\bmid[- ]july\b",
            r"\bpending transcript\b",
            r"\bpending transcripts\b",
            r"\bpending final transcript\b",
            r"\bpending final transcripts\b",
            r"\bhow pending transcripts are handled\b",
            r"\btranscripts are handled during the application review\b",
            r"\bresults.*mid[- ]july\b",
        ],
        "query": "Can an applicant apply before receiving the final transcript or final degree certificate?",
    },
    "english_language_requirements": {
        "label": "English language requirements",
        "patterns": [
            r"\benglish (language )?(proof|requirement|requirements|certificate|proficiency|test)",
            r"\bielts\b|\btoefl\b|\btoeic\b|\bpte academic\b",
            r"\bdegree was taught.*english\b",
            r"\breplace an english test\b",
            r"\blanguage requirements\b",
            r"\blanguage requirement\b",
            r"\blanguage proof\b",
            r"\bwhat language proof\b",
            r"\bwhich language proof\b",
            r"\bproof of language\b",
            r"\bproof of english\b",
            r"\benglish proof\b",
            r"\bis language proof required\b",
            r"\bwhat language proof is required\b",
        ],
        "query": "What English language proof is required for the programme?",
    },
    "german_language_requirements": {
        "label": "German language requirements",
        "patterns": [
            r"\bgerman (language )?(proof|requirement|requirements|certificate)",
            r"\bdo i need german\b",
        ],
        "query": "Is German language proof required for the programme?",
    },
    "language_of_instruction": {
        "label": "Language of instruction",
        "patterns": [
            r"\blanguage of instruction\b",
            r"\btaught in english\b",
            r"\bentirely in english\b",
            r"\bfully in english\b",
            r"\bis (the )?(course|programme|program) in english\b",
            r"\bwhat language (is|are).{0,60}(course|programme|program|lectures|classes)",
            r"\bis (the )?(course|programme|program) taught in english\b",
            r"\bis it taught in english\b",
            r"\bare lectures in english\b",
            r"\bare classes in english\b",
        ],
        "query": "What is the language of instruction for the programme?",
    },
    "study_format": {
        "label": "Study format",
        "patterns": [
            r"\bstudy format\b",
            r"\bon[- ]campus\b",
            r"\bon campus course\b",
            r"\bdistance learning\b",
            r"\bonline course\b",
            r"\bpart[- ]time\b",
            r"\bfull[- ]time\b",
        ],
        "query": "What is the study format for the programme, such as on-campus, online, distance learning, full-time or part-time?",
    },
    "application_fee": {
        "label": "Application fee",
        "patterns": [
            r"\bapplication fee\b",
            r"\bapplication fees\b",
            r"\buni[- ]assist (fee|fees|processing fee|processing costs|costs)",
            r"\bprocessing fee\b",
            r"\bprocessing costs\b",
        ],
        "query": "Are there application fees or uni-assist processing fees?",
    },
    "tuition_fees": {
        "label": "Tuition fees",
        "patterns": [
            r"\btuition fee\b",
            r"\btuition fees\b",
        ],
        "query": "Are there tuition fees for studying at HTW Berlin?",
    },
    "semester_contribution": {
        "label": "Semester contribution",
        "patterns": [
            r"\bsemester contribution\b",
            r"\bsemester fee\b",
            r"\bsemester fees\b",
        ],
        "query": "What is the semester contribution or semester fee?",
    },
    "application_route": {
        "label": "Application route / uni-assist",
        "patterns": [
            r"\buni[- ]assist\b",
            r"\bhochschulstart\b",
            r"\bdosv\b",
            r"\bapply through\b",
            r"\bapplication route\b",
            r"\bapply via\b",
            r"\bapply using\b",
            r"\beu or international\b",
            r"\beu applicant\b",
            r"\binternational applicant\b",
            r"\bhtw berlin application portal\b",

            # Common student wording
            r"\bhow should i apply\b",
            r"\bhow i should apply\b",
            r"\bhow do i apply\b",
            r"\bhow can i apply\b",
            r"\bhow to apply\b",
            r"\bwhere should i apply\b",
            r"\bwhere do i apply\b",
            r"\bwhere can i apply\b",
            r"\bwhich portal\b",
            r"\bwhich application portal\b",
            r"\bonline application form\b",
            r"\bapplication form\b",
            r"\bfill out the online application\b",
            r"\bsubmit my application\b",
            r"\bsubmit the application\b",

            # General process wording
            r"\bapplication process\b",
            r"\bapply for this programme\b",
            r"\bapply for the programme\b",
            r"\bapplication procedure\b",
        ],
        "query": "Which application route or application process should the applicant use, including the programme application page, HTW application portal, Hochschulstart or uni-assist if relevant?",
    },
    "motivation_letter": {
        "label": "Motivation letter",
        "patterns": [
            r"\bmotivation letter\b",
            r"\bmotivation letters\b",
            r"\bletter of motivation\b",
        ],
        "query": "Is a motivation letter required for the programme application?",
    },
    "aps_certificate": {
        "label": "APS certificate",
        "patterns": [
            r"\baps\b",
            r"\bacademic test centre\b",
            r"\bacademic test center\b",
        ],
        "query": "Is an APS certificate required?",
    },
    "certified_translations": {
        "label": "Certified translations",
        "patterns": [
            r"\bcertified translation",
            r"\btranslated documents\b",
            r"\bofficial translations\b",
            r"\bdocuments are in portuguese\b",
            r"\blanguages other than german or english\b",
        ],
        "query": "Are certified translations required for application documents?",
    },
    "hard_copy_documents": {
        "label": "Hard-copy documents",
        "patterns": [
            r"\bhard[- ]copy\b",
            r"\bhard copies\b",
            r"\bby post\b",
            r"\boriginals be sent\b",
        ],
        "query": "Are hard-copy documents required or are digital uploads sufficient?",
    },
    "document_uploads": {
        "label": "Digital document upload",
        "patterns": [
            r"\bdigital upload",
            r"\bupload(ed)?\b",
            r"\bpdf\b",
        ],
        "query": "How should application documents be uploaded?",
    },
    "required_documents": {
        "label": "Required documents",
        "patterns": [
            r"\bwhat documents\b",
            r"\bwhich documents\b",
            r"\bdocuments (are )?needed\b",
            r"\bdocuments should i prepare\b",
            r"\bdocuments do i need\b",
            r"\brequired documents\b",
            r"\bdocument requirements\b",
            r"\bapplication documents\b",
            r"\bofficial transcripts\b",
            r"\btranscripts\b",
            r"\bcertificates\b",
        ],
        "query": "Which documents are required for the application?",
    },
    "credit_recognition": {
        "label": "Credit recognition",
        "patterns": [
            r"\bcredit recognition\b",
            r"\bcredits can be (recognised|recognized|transferred)\b",
            r"\bprevious credits\b",
            r"\btransfer credits\b",
        ],
        "query": "Can previous university credits be recognised or transferred?",
    },
    "grade_conversion": {
        "label": "Grade conversion",
        "patterns": [
            r"\bgrade conversion\b",
            r"\bdifferent grading system\b",
            r"\bhow is my grade converted\b",
        ],
        "query": "How are foreign grades converted during the application process?",
    },
    "qualification_recognition": {
        "label": "Qualification recognition",
        "patterns": [
            r"\bqualification recognition\b",
            r"\bdiploma fulfils\b",
            r"\bdiploma fulfills\b",
            r"\bdoes my .*diploma meet\b",
            r"\bdoes my .*qualification meet\b",
            r"\bgeneral admission requirements\b",
            r"\bhigher education entrance qualification\b",
            r"\bschool[- ]leaving certificate\b",
            r"\bforeign school qualification\b",
            r"\bvpd\b",
            r"\bib diploma\b",
            r"\binternational baccalaureate\b",
            r"\bfrench baccalaur",
            r"\banabin\b",
            r"\bdaad admission",
        ],
        "query": "How is an International Baccalaureate or other foreign school qualification recognised as a higher education entrance qualification for Bachelor admission?",
    },
    "conditional_enrolment": {
        "label": "Conditional enrolment / final certificate",
        "patterns": [
            r"\benrol(l)? conditionally\b",
            r"\bfinal submission deadline\b",
            r"\bfinal certificate is delayed\b",
            r"\badmission offer\b",
        ],
        "query": "Can the applicant enrol conditionally if the final certificate is delayed?",
    },
    "application_deadline": {
        "label": "Application deadline",
        "patterns": [
            r"\bapplication deadline\b",
            r"\bdeadlines\b",
            r"\bdeadline\b",
            r"\bapplication period\b",
            r"\bnext application period\b",
            r"\bmissed the .*deadline\b",
            r"\blate application\b",
            r"\bregular deadline\b",
            r"\bregular deadlines\b",
            r"\bwithin the regular deadlines\b",
        ],
        "query": "What is the application deadline or application period?",
    },
    "accommodation": {
        "label": "Accommodation",
        "patterns": [
            r"\baccommodation\b",
            r"\bhousing\b",
            r"\bdormitory\b",
            r"\bstudent residence\b",
        ],
        "query": "Is accommodation or housing support available for international students?",
    },
    "work_experience": {
        "label": "Work experience",
        "patterns": [
            r"\bwork experience\b",
            r"\bprofessional experience\b",
            r"\bqualified professional experience\b",
            r"\bprofessional practice\b",
            r"\bpractical experience\b",
            r"\bone year of experience\b",
            r"\b1 year of experience\b",
            r"\bexperience required\b",
        ],
        "query": "Is work experience or professional experience required for admission to the programme?",
    },
    "admission_requirements": {
        "label": "Admission requirements",
        "patterns": [
            r"\badmission requirements\b",
            r"\bprogramme[- ]specific requirements\b",
            r"\bspecial programme requirements\b",
            r"\bspecific admission requirements\b",
        ],
        "query": "What are the admission requirements for the programme?",
    },
    "application_process": {
        "label": "Application process",
        "patterns": [
            r"\bapplication process\b",
            r"\bapplication procedure\b",
            r"\bexplain the application process\b",
            r"\bhow should i apply\b",
            r"\bhow i should apply\b",
            r"\bhow do i apply\b",
            r"\bhow can i apply\b",
            r"\bhow to apply\b",
            r"\bwhere should i apply\b",
            r"\bwhere do i apply\b",
            r"\bwhich portal\b",
            r"\bonline application form\b",
            r"\bapplication form\b",
        ],
        "query": "What is the application process?",
    },
}


# Priority order controls the final order in the draft.
TOPIC_ORDER = [
    "application_route",
    "application_process",
    "qualification_recognition",
    "admission_requirements",
    "application_before_graduation",
    "conditional_enrolment",
    "application_deadline",
    "required_documents",
    "document_uploads",
    "hard_copy_documents",
    "certified_translations",
    "aps_certificate",
    "credit_recognition",
    "grade_conversion",
    "english_language_requirements",
    "german_language_requirements",
    "language_of_instruction",
    "study_format",
    "work_experience",
    "motivation_letter",
    "application_fee",
    "tuition_fees",
    "semester_contribution",
    "accommodation",
]


def detect_topics(email_text: str, context: Dict[str, Optional[str]], max_topics: int = 4) -> List[Dict[str, str]]:
    """
    Detect the real information needs in a student email.
    This function is intentionally conservative:
    - do not create generic topics just because a keyword appears in background text,
    - merge overlapping topics,
    - keep the topic list small so the draft stays readable.
    """
    text = (email_text or "").lower()
    found: List[str] = []

    for topic_id, spec in TOPIC_DEFINITIONS.items():
        for pattern in spec["patterns"]:
            if re.search(pattern, text, flags=re.IGNORECASE):
                found.append(topic_id)
                break
            
    # Extra safety for common student wording.
    # Example: "what language proof is required?"
    # This should not be missed just because the student did not write "English proof".
    if re.search(r"\b(language proof|proof of language|what language proof|which language proof|language requirement|language requirements)\b", text):
        if "english_language_requirements" not in found and "german_language_requirements" not in found:
            found.append("english_language_requirements")

    # Only answer language of instruction when the student asks it as a question.
    # Do not add this topic only because the programme name/description says "English-taught".
    if re.search(
        r"\b("
        r"what\s+language\s+(is|are).{0,60}(course|programme|program|lectures|classes)|"
        r"is\s+(the\s+)?(course|programme|program)\s+taught\s+in\s+english|"
        r"is\s+it\s+taught\s+in\s+english|"
        r"are\s+(lectures|classes)\s+in\s+english"
        r")\b",
        text,
        flags=re.IGNORECASE,
    ):
        if "language_of_instruction" not in found:
            found.append("language_of_instruction")

    # If the student asks about campus/online format, make sure study_format is included.
    if re.search(r"\b(on[- ]campus|on campus|online|distance learning|study format|full[- ]time|part[- ]time)\b", text):
        if "study_format" not in found:
            found.append("study_format")
    # Extra safety for application route / application process wording.
    # This catches natural wording such as:
    # "how I should apply", "where should I apply", "which portal should I use?"
    if re.search(
        r"\b("
        r"how\s+(i\s+)?(should|can|do)\s+apply|"
        r"how\s+to\s+apply|"
        r"where\s+(i\s+)?(should|can|do)\s+apply|"
        r"which\s+(application\s+)?portal|"
        r"online\s+application\s+form|"
        r"application\s+form|"
        r"application\s+process|"
        r"application\s+procedure|"
        r"submit\s+(my|the)\s+application"
        r")\b",
        text,
        flags=re.IGNORECASE,
    ):
        if "application_route" not in found:
            found.append("application_route")

    # Merge rules to avoid duplicated topics.
    found_set = set(found)

    # If explicit upload/hard copy/translation topics exist, avoid generic required_documents
    # unless "what documents/documents needed/required documents" was explicitly asked.
    if "required_documents" in found_set:
        explicit_required = re.search(r"\b(what documents|documents (are )?needed|required documents|document requirements)\b", text)
        if not explicit_required:
            found_set.discard("required_documents")

    # If application route is present, generic application process adds little value.
    if "application_route" in found_set:
        found_set.discard("application_process")

    # If precise cost topics are present, do not create generic fee topic. We do not have a generic fee topic,
    # but keep application_fee, tuition_fees and semester_contribution separate only when explicitly asked.
    if "application_fee" in found_set and "semester_contribution" not in found_set:
        # Do nothing. Application fee is separate from semester contribution.
        pass

    # For "English-taught Master's programme", avoid a target-degree mistake.
    # It may also imply English proof, but only if the email asks if proof is needed/replaced.
    if "english_language_requirements" in found_set:
        pass

    ordered = [tid for tid in TOPIC_ORDER if tid in found_set]

    # If no topic detected, use application_process only as safe fallback.
    if not ordered:
        ordered = ["application_process"]

    # Keep at most max_topics. This prevents V4.4-style over-splitting.
    # Cost topics are allowed to remain separate when specifically asked.
    ordered = ordered[:max_topics]

    topics: List[Dict[str, str]] = []
    for tid in ordered:
        spec = TOPIC_DEFINITIONS[tid]
        topics.append({
            "topic_id": tid,
            "label": spec["label"],
            "base_query": spec["query"],
            "query": build_evidence_query(tid, spec["query"], context),
        })

    return topics


def build_evidence_query(topic_id: str, base_query: str, context: Dict[str, Optional[str]]) -> str:
    """
    Enrich a topic query with the target programme and target degree.
    This makes retrieval more precise but avoids mixing previous degree with target degree.
    """
    parts = [base_query]

    if context.get("target_program"):
        parts.append(f"Target programme: {context['target_program']}")
        catalog_terms = programme_query_terms(context)
        if catalog_terms:
            parts.append(f"Programme catalogue match: {catalog_terms}")

    if context.get("target_degree"):
        parts.append(f"Target degree: {context['target_degree']}")

    if context.get("country") and topic_id in {
        "application_route",
        "qualification_recognition",
        "aps_certificate",
        "certified_translations",
        "application_fee",
    }:
        parts.append(f"Applicant country/background: {context['country']}")

    if context.get("citizenship_group") and topic_id == "application_route":
        parts.append(f"Citizenship category: {context['citizenship_group']}")

    if context.get("residence_country") and topic_id == "application_route":
        parts.append(f"Residence country: {context['residence_country']}")

    # Topic-specific retrieval hints. These are not additional facts; they guide
    # retrieval toward the right HTW/application pages.
    if topic_id == "application_route":
        parts.append("Include application route, Hochschulstart, DoSV, HTW application portal, EU/EEA and uni-assist rules.")
    elif topic_id == "qualification_recognition":
        parts.append("Include International Baccalaureate, IB diploma, foreign school leaving certificate, higher education entrance qualification, anabin and DAAD admission database.")
    elif topic_id == "motivation_letter":
        parts.append("Check whether motivation letter is listed as a programme-specific required document.")
    elif topic_id == "english_language_requirements":
        parts.append("Include CEFR level, IELTS, TOEFL, TOEIC or accepted English proof if listed.")
    elif topic_id == "language_of_instruction":
        parts.append("Use the programme page if available. Include whether the programme is taught in English or German.")
    elif topic_id == "study_format":
        parts.append("Use the programme page if available. Include whether the programme is on-campus, online, distance learning, full-time or part-time.")
    elif topic_id == "application_deadline":
        parts.append("Prefer programme-specific applying page or programme deadline page when a programme is detected. Use general HTW deadline rules only as backup.")
    elif topic_id == "application_fee":
        parts.append(
            "Focus on application processing fees, uni-assist handling fees, payment by the application deadline, "
            "and whether the application route uses uni-assist. Do not focus on tuition fees."
        )
    elif topic_id == "application_before_graduation":
        parts.append(
            "Focus on applying before graduation, pending final transcript, provisional transcript, final certificate, "
            "final semester results, conditional admission or enrolment, and the deadline for submitting the final certificate. "
            "Prefer programme-specific application pages when a programme is detected."
        )

    return ". ".join(parts)


# ---------------------------------------------------------------------
# Draft generation
# ---------------------------------------------------------------------

def _build_context(docs: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        title = d.get("title", "") or ""
        url = d.get("source_url", "") or d.get("url", "") or ""
        content = d.get("content", "") or d.get("chunk_text", "") or ""
        updated = d.get("last_updated", "") or ""
        blocks.append(
            f"[Doc {i}] {title}\nURL: {url}\nLast updated: {updated}\nCONTENT:\n{content[:1400]}"
        )
    return "\n\n---\n\n".join(blocks)


def _student_greeting(context: Dict[str, Optional[str]]) -> str:
    name = context.get("student_name")
    if name:
        return f"Dear {name},"
    return "Dear applicant,"

def _topic_ids(topics: List[Dict[str, str]]) -> set:
    return {str(t.get("topic_id", "") or "") for t in topics}


def _find_best_application_fee_citation(docs: List[Dict[str, Any]]) -> str:
    """
    Find the best available Doc citation for application processing fees.

    Preference:
    - HTW uni-assist application pages
    - any source mentioning uni-assist / processing fee / application fee
    """
    for index, doc in enumerate(docs or [], start=1):
        combined = " ".join(
            str(doc.get(key, "") or "")
            for key in ["title", "source_url", "url", "content", "chunk_text", "object_type"]
        ).lower()

        if (
            "uni-assist" in combined
            or "uni assist" in combined
            or "processing fee" in combined
            or "processing costs" in combined
            or "application fee" in combined
            or "application fees" in combined
            or "handling fees" in combined
        ):
            return f"[Doc {index}]"

    return ""


def _application_fee_guidance_for_prompt(
    topics: List[Dict[str, str]],
    context: Dict[str, Optional[str]],
) -> str:
    """
    Add explicit instruction for application-fee cases.

    This is not a final answer. It only tells the model how to interpret the
    student's fee question correctly.
    """
    topic_ids = _topic_ids(topics)

    if "application_fee" not in topic_ids:
        return ""

    tuition_asked = "tuition_fees" in topic_ids
    semester_asked = "semester_contribution" in topic_ids

    country = str(context.get("country") or "").strip()
    previous_degree = str(context.get("previous_degree") or "").strip()

    lines = [
        "APPLICATION FEE INTERPRETATION:",
        "- The student asked about application fees.",
        "- Treat this as application processing fees or uni-assist handling fees.",
        "- Do not answer this question with tuition fees or semester contribution.",
        "- Do not write that the programme is tuition-free as the main answer to the application fee question.",
    ]

    if tuition_asked or semester_asked:
        lines.append(
            "- Tuition fees or semester contribution may be answered only because they were asked as separate topics."
        )
    else:
        lines.append(
            "- Since tuition fees and semester contribution were not asked as separate topics, do not include them in the application fee paragraph."
        )

    if country:
        lines.append(f"- Applicant background/country detected: {country}.")
    if previous_degree:
        lines.append(f"- Previous/current education detected: {previous_degree}.")

    lines.append(
        "- If the applicant must use uni-assist according to the evidence, say that uni-assist processing costs/handling fees apply and must be paid by the deadline."
    )
    lines.append(
        "- If the exact amount is not in the evidence, do not invent it. Refer to the official uni-assist handling fee page in the staff verification links."
    )

    return "\n".join(lines)

def generate_staff_email_draft(
    original_email: str,
    context: Dict[str, Optional[str]],
    topics: List[Dict[str, str]],
    docs: List[Dict[str, Any]],
) -> str:
    """
    Generate ONE final email draft using all retrieved evidence.
    This restores the fluency of the baseline while keeping topic-level retrieval.
    """
    if not docs:
        return (
            f"{_student_greeting(context)}\n\n"
            "Thank you for your enquiry.\n\n"
            "I could not confirm the requested information from the available sources. "
            "Please contact Student Services directly so that your case can be checked.\n\n"
            "Kind regards,\n"
            "HTW Berlin Student Services"
        )

    topics_text = "\n".join(
        [f"- {t['label']}: {t['base_query']}" for t in topics]
    )

    profile_bits = []
    for key, label in [
        ("target_degree", "Target degree"),
        ("target_program", "Target programme"),
        ("target_program_url", "Programme page"),
        ("target_program_application_url", "Programme application page"),
        ("catalog_degree", "Catalogue degree hint"),
        ("catalog_language", "Catalogue language hint"),
        ("catalog_study_format", "Catalogue study format hint"),
        ("previous_degree", "Previous/current education"),
        ("country", "Applicant background"),
        ("citizenship_group", "Citizenship category"),
        ("residence_country", "Residence country"),
    ]:
        if context.get(key):
            profile_bits.append(f"{label}: {context[key]}")

    profile = "\n".join(profile_bits) if profile_bits else "No clear profile information detected."

    evidence = _build_context(docs)
    application_fee_guidance = _application_fee_guidance_for_prompt(topics, context)

    system = (
        "You are drafting an email from HTW Berlin Student Services to a student. "
        "Write in simple, polite and professional English. "
        "Use only the evidence documents. "
        "Do not invent information. "
        "Write directly to the student using 'you', not 'the student'. "
        "Keep the email ready to paste and send after staff review. "
        "Be specific and useful, but stay cautious where formal checking is required."
    )

    user = (
        f"ORIGINAL STUDENT EMAIL:\n{original_email}\n\n"
        f"INTERPRETED STUDENT PROFILE:\n{profile}\n\n"
        f"TOPICS TO ANSWER ONLY:\n{topics_text}\n\n"
        f"EVIDENCE DOCUMENTS:\n{evidence}\n\n"
        f"{application_fee_guidance}\n\n"
        "DRAFTING RULES:\n"
        "1) Start with the greeting provided below.\n"
        f"GREETING: {_student_greeting(context)}\n"
        "After the greeting, add one short polite opening sentence. "
        "If a specific programme was detected, write: "
        "'Thank you for your interest in [programme name] at HTW Berlin.' "
        "If no specific programme was detected, write: "
        "'Thank you for your enquiry.'\n"
        "2) Answer only the topics asked in the student email or listed above. "
        "Do not add additional sections from the evidence, such as language requirements, deadlines, or fees, unless the student asked about them.\n"
        "3) Keep each topic to 1-3 short sentences.\n"
        "4) Do not use markdown headings, tables, or long bullet lists.\n"
        "5) Include citations like [Doc 1] after factual claims.\n"
        "6) For application route questions, consider citizenship, residence country, target degree and programme. "
        "Citizenship and residence country are different. "
        "If the interpreted profile says Citizenship category: EU/EEA, do not classify the applicant as non-EU only because they live outside the EU or because their school certificate was obtained outside Germany. "
        "For a first-semester Bachelor application, mention Hochschulstart, HTW portal, or uni-assist only if supported by the evidence.\n"
        "7) For International Baccalaureate or foreign school certificates, do not state final acceptance. "
        "Say that the exact subject combination/results must be checked during the application process.\n"
        "8) For motivation letters, if the evidence does not list a motivation letter as a required document, say it is not listed as a programme-specific required document, "
        "but the applicant should follow the application portal if it requests one.\n"
        "9) For application fee questions, answer only application processing fees or uni-assist handling fees. "
        "Never answer an application fee question by saying that the programme is tuition-free or that only a semester fee is paid. "
        "Tuition fees and semester contribution are different topics and may only be mentioned if the student explicitly asked about them as separate topics.\n"
        "10) If evidence is missing for one topic, do not write internal staff notes inside the student email. "
        "Do not write phrases such as 'This point should be checked by staff before the final reply is sent'. "
        "Instead, give the most specific confirmed information from the evidence. If a detail is not confirmed, write a normal student-facing sentence such as: "
        "'The programme page linked below provides the most specific details for this point.'\n"
        "11) Do not ask the student to repeat the programme name if a specific programme was detected in the interpreted profile.\n"
        "12) Do not say 'Thank you for your interest in HTW Berlin’s Master’s programmes' if a specific programme was detected. "
        "Say 'Thank you for your interest in [programme name] at HTW Berlin.'\n"
        "13) Keep the style close to a normal staff email. Avoid technical words such as evidence, grounding, retrieved documents, or staff review in the student-facing draft.\n"
        "14) For tuition fee questions, never answer only from the general rule that public universities in Berlin do not charge tuition fees. "
        "If programme-specific fee evidence is available, use that first. If no programme-specific fee evidence is available, say that the programme page linked below should be used to confirm programme-specific fees.\n"
        "15) For paid international programmes, mention tuition fees only if the amount is supported by the provided sources.\n"
        "16) For pending transcript or application-before-graduation questions, first look for evidence about provisional transcripts, final certificates, final results, conditional admission, or later submission deadlines. "
        "Do not answer only with a generic deadline paragraph if the student asks about pending final documents.\n"
        "17) End with:\nKind regards,\nHTW Berlin Student Services\n"
    )

    provider = (config.GENERATION_PROVIDER or "extractive").strip().lower()
    draft = ""

    if provider == "mistral" and getattr(config, "MISTRAL_API_KEY", ""):
        from mistralai import Mistral

        client = Mistral(api_key=config.MISTRAL_API_KEY)
        resp = client.chat.complete(
            model=getattr(config, "GENERATION_MODEL", "") or "mistral-small-latest",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        if resp.choices:
            draft = resp.choices[0].message.content or ""

    elif provider == "anthropic" and config.ANTHROPIC_API_KEY:
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=getattr(config, "GENERATION_MODEL", "") or "claude-3-haiku-20240307",
            max_tokens=1200,
            temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        draft = resp.content[0].text if resp.content else ""

    else:
        draft = _extractive_email_draft(context, topics, docs)

    # Safety fallback if the selected provider returns an empty draft.
    if not str(draft or "").strip():
        draft = _extractive_email_draft(context, topics, docs)

    # Clean the actual generated draft, then apply the fee guard and staff reference links.
    cleaned = clean_staff_draft(draft, context)
    cleaned = fix_application_fee_confusion(cleaned, topics, docs, context)
    return add_reference_links_to_draft(cleaned, docs, topics, context)

def _extractive_email_draft(
    context: Dict[str, Optional[str]],
    topics: List[Dict[str, str]],
    docs: List[Dict[str, Any]],
) -> str:
    lines = [
        _student_greeting(context),
        "",
        "Thank you for your enquiry.",
        "",
    ]
    for i, t in enumerate(topics[:4], start=1):
        lines.append(f"Regarding {t['label'].lower()}, please check the following information from the available HTW sources [Doc {min(i, len(docs))}].")
    lines.extend(["", "Kind regards,", "HTW Berlin Student Services"])
    return "\n".join(lines)


BAD_PHRASES = [
    # Not student-facing
    "the student",
    "selected programme",
    "programme programme",
    "based on the provided documents",
    "based on your profile",
    "evidence documents",
    "retrieved documents",
    "grounding",
    "grounded",
    "staff review",

    # Asking for information already present or sounding unhelpful
    "i need more information",
    "could you please specify",
    "which programme",
    "please let us know which degree programme",
    "contact us again",

    # Too vague or unsuitable
    "the evidence documents do not contain",
    "available information does not specify",
    "not available in our current documentation",
    "does not contain specific information",
    "do not contain specific information",
    "not specified in the available",
    "not specified in our",
    "cannot confirm",
    "could not confirm",
    "not confirm",
    "contact student services directly",
    "contact the admissions office",
    "this will determine your exact deadline",
    "which category applies to",

    # Over-general greeting when a programme is known
    "thank you for your interest in htw berlin's master's programmes",
    "thank you for your interest in htw berlin’s master’s programmes",
]

def fix_application_fee_confusion(
    draft: str,
    topics: List[Dict[str, str]],
    docs: List[Dict[str, Any]],
    context: Dict[str, Optional[str]],
) -> str:
    """
    Fix a common LLM failure:
    The student asks about application fees, but the draft answers tuition fees
    or semester contribution instead.

    This guard only runs when:
    - application_fee is a detected topic,
    - tuition_fees and semester_contribution are not detected as separate topics,
    - the draft contains a fee paragraph that wrongly says tuition-free or semester fee.
    """
    text = draft or ""
    topic_ids = _topic_ids(topics)

    if "application_fee" not in topic_ids:
        return text

    # If the student explicitly asked about tuition or semester contribution too,
    # do not rewrite the fee paragraph.
    if "tuition_fees" in topic_ids or "semester_contribution" in topic_ids:
        return text

    lower = text.lower()

    fee_confusion_markers = [
        "programme itself is tuition-free",
        "program itself is tuition-free",
        "master's programme itself is tuition-free",
        "master's program itself is tuition-free",
        "only pay a semester fee",
        "only pay a semester contribution",
        "tuition-free, and you only pay",
    ]

    if not any(marker in lower for marker in fee_confusion_markers):
        return text

    citation = _find_best_application_fee_citation(docs)
    citation_text = f" {citation}" if citation else ""

    country = str(context.get("country") or "").strip()
    previous_degree = str(context.get("previous_degree") or "").strip()

    if country or previous_degree:
        corrected_paragraph = (
            "Application fees: Since your previous education was completed outside Germany, "
            "you may need to apply via uni-assist depending on the application route confirmed for your case. "
            f"Applications via uni-assist are subject to processing or handling fees, which must be paid by the application deadline{citation_text}. "
            "Please check the official uni-assist handling fee page linked below for the current amount."
        )
    else:
        corrected_paragraph = (
            "Application fees: Application fees refer to application processing costs, for example uni-assist handling fees where uni-assist is used. "
            f"If your application route uses uni-assist, the processing or handling fee must be paid by the application deadline{citation_text}. "
            "Please check the official uni-assist handling fee page linked below for the current amount."
        )

    # Replace an existing "Application fees:" paragraph.
    pattern = re.compile(
        r"(Application fees:\s*)(.*?)(?=\n\n[A-Z][A-Za-z /-]+:|\n\nFor detailed|\n\nKind regards|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    if pattern.search(text):
        text = pattern.sub(corrected_paragraph, text, count=1)
        return text.strip()

    # If no explicit paragraph found, add a corrected paragraph before closing.
    if "Kind regards" in text:
        text = text.replace("Kind regards", corrected_paragraph + "\n\nKind regards", 1)
    else:
        text = text.rstrip() + "\n\n" + corrected_paragraph

    return text.strip()

def clean_staff_draft(draft: str, context: Dict[str, Optional[str]]) -> str:
    """Remove common model artefacts and make the text more student-facing."""
    text = draft or ""

    # Remove markdown formatting.
    text = re.sub(r"#+\s*", "", text)
    text = text.replace("**", "")
    text = text.replace("__", "")

    # Make it directly student-facing.
    text = re.sub(r"\bthe student\b", "you", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe applicant\b", "you", text, flags=re.IGNORECASE)
    
    # Remove internal staff-check wording from the student-facing draft.
    text = re.sub(
        r"This point should be checked by staff before the final reply is sent\.?",
        "The programme page linked below provides the most specific details for this point.",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"This point should be reviewed by staff before the final reply is sent\.?",
        "The programme page linked below provides the most specific details for this point.",
        text,
        flags=re.IGNORECASE,
    )

    # Fix common wording artefacts from source text or generation.
    text = re.sub(r"\byou body\b", "student body", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou organisation\b", "student organisation", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour body\b", "student body", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour organisation\b", "student organisation", text, flags=re.IGNORECASE)

    # Avoid repeated greeting if model adds extra notes.
    greeting = _student_greeting(context)
    if greeting in text:
        before, after = text.split(greeting, 1)
        text = greeting + after
    else:
        text = greeting + "\n\n" + text.strip()

    # Normalize endings.
    if "Kind regards" not in text:
        text = text.rstrip() + "\n\nKind regards,\nHTW Berlin Student Services"

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def _doc_url(doc: Dict[str, Any]) -> str:
    return (doc.get("source_url", "") or doc.get("url", "") or "").strip()

# ---------------------------------------------------------------------
# Programme-aware source filtering
# ---------------------------------------------------------------------

def _normalise_for_match(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9äöüß]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _doc_text_for_matching(doc: Dict[str, Any]) -> str:
    return " ".join(
        str(doc.get(key, "") or "")
        for key in [
            "title",
            "source_url",
            "url",
            "object_type",
            "object_id",
            "content",
            "chunk_text",
        ]
    )


def _programme_aliases_from_context(context: Dict[str, Optional[str]]) -> List[str]:
    """
    Build safe programme match terms from the detected programme context.

    This does not add programme facts. It only helps us avoid using the wrong
    programme page as evidence.
    """
    aliases: List[str] = []

    for key in [
        "target_program",
        "target_program_url",
        "target_program_application_url",
    ]:
        value = context.get(key)
        if value:
            aliases.append(str(value))

    program = str(context.get("target_program") or "").strip().lower()

    # Small alias support for programme identifiers already used in the catalogue.
    if "project management and data science" in program:
        aliases.extend(["mpmd", "project management and data science"])

    if "professional it" in program or "digitalization" in program:
        aliases.extend([
            "proitd",
            "professional it",
            "professional it business and digitalization",
            "professional it and digitalization",
        ])

    if "construction and real estate" in program:
        aliases.extend(["conrem", "construction and real estate management"])

    if "cyber security and business" in program or "cybersecurity and business" in program:
        aliases.extend(["cyber security and business", "cybersecurity and business"])

    if "international business" in program:
        aliases.extend(["international business", "mib"])

    # Add clean terms from the programme catalogue helper.
    catalogue_terms = programme_query_terms(context)
    if catalogue_terms:
        aliases.extend([part.strip() for part in catalogue_terms.split() if len(part.strip()) >= 4])

    # Deduplicate.
    seen = set()
    clean_aliases: List[str] = []
    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias:
            continue
        key = alias.lower()
        if key not in seen:
            seen.add(key)
            clean_aliases.append(alias)

    return clean_aliases


def _doc_programme_score(doc: Dict[str, Any], context: Dict[str, Optional[str]]) -> int:
    """
    Positive score means the document looks related to the detected programme.
    Negative score means it likely belongs to another programme or wrong section.
    """
    text = _normalise_for_match(_doc_text_for_matching(doc))
    url = str(doc.get("source_url", "") or doc.get("url", "") or "").lower()
    title = str(doc.get("title", "") or "").lower()

    aliases = _programme_aliases_from_context(context)
    score = 0

    for alias in aliases:
        alias_norm = _normalise_for_match(alias)
        if not alias_norm or len(alias_norm) < 4:
            continue

        if alias_norm in text:
            score += 4

        if alias_norm in _normalise_for_match(url):
            score += 6

        if alias_norm in _normalise_for_match(title):
            score += 5

    # Programme URL match is very strong.
    target_url = str(context.get("target_program_url") or "").lower().strip()
    if target_url:
        target_host_or_slug = target_url.replace("https://", "").replace("http://", "").strip("/")
        if target_host_or_slug and target_host_or_slug in url:
            score += 10

    # Downrank known wrong/general sections for programme-specific questions.
    wrong_or_weak_fragments = [
        "student-exchange-programmes",
        "nomination-and-application",
        "studying-abroad",
        "pathways-abroad",
        "wege-an-die-htw-berlin/student-exchange",
        "campus-stories",
    ]

    if any(fragment in url for fragment in wrong_or_weak_fragments):
        score -= 5

    # Avoid using another specific programme page when a target programme is known.
    target_program = _normalise_for_match(context.get("target_program") or "")
    other_programme_clues = [
        "information technology master",
        "international business bachelor",
        "construction and real estate management",
        "cyber security and business",
        "project management and data science",
        "professional it business and digitalization",
    ]

    for clue in other_programme_clues:
        clue_norm = _normalise_for_match(clue)
        if clue_norm and clue_norm in text and clue_norm not in target_program:
            score -= 6

    return score


def filter_docs_for_programme(
    docs: List[Dict[str, Any]],
    context: Dict[str, Optional[str]],
    topic_id: str = "",
    min_keep: int = 3,
) -> List[Dict[str, Any]]:
    """
    Prefer documents related to the detected programme for programme-specific topics.

    This is intentionally light-touch:
    - if programme-specific docs are found, prefer them;
    - if none are found, keep the original docs so the system can still answer
      using general HTW application information.
    """
    if not docs:
        return docs

    if not context.get("target_program"):
        return docs

    programme_specific_topics = {
        "application_deadline",
        "admission_requirements",
        "required_documents",
        "english_language_requirements",
        "german_language_requirements",
        "language_of_instruction",
        "study_format",
        "work_experience",
        "motivation_letter",
        "application_before_graduation",
    }

    # Application route and fees often need general HTW / uni-assist pages.
    # Do not filter them too strongly.
    if topic_id not in programme_specific_topics:
        return docs

    scored = [(doc, _doc_programme_score(doc, context)) for doc in docs]
    strong = [doc for doc, score in scored if score >= 3]

    if len(strong) >= min_keep:
        return strong[: max(min_keep, 5)]

    if strong:
        # Keep strong programme docs first, then add original docs as backup.
        result: List[Dict[str, Any]] = []
        seen = set()

        for doc in strong + docs:
            doc_id = str(doc.get("id", "")) or str(doc.get("source_url", "")) or str(doc.get("url", ""))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            result.append(doc)
            if len(result) >= max(min_keep, 5):
                break

        return result

    return docs

# ---------------------------------------------------------------------
# UI source ordering and citation remapping
# ---------------------------------------------------------------------

def _extract_cited_doc_numbers_from_draft(draft: str) -> List[int]:
    """
    Extract cited document numbers from the draft.

    Example:
        "Please see [Doc 1] and [Doc 6]" -> [1, 6]
    """
    numbers: List[int] = []

    for match in re.finditer(r"\[Doc\s*(\d+)\]", draft or "", flags=re.IGNORECASE):
        try:
            number = int(match.group(1))
            if number not in numbers:
                numbers.append(number)
        except Exception:
            continue

    return numbers


def _is_official_programme_cache_doc(doc: Dict[str, Any]) -> bool:
    """
    True if the document came from the official programme-page cache.
    These should be shown before general HTW backup pages.
    """
    object_type = str(doc.get("object_type", "") or "").lower()
    source = str(doc.get("source", "") or "").lower()
    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
    metadata_source = str(metadata.get("source", "") or "").lower()

    return (
        "official_programme_page_cache" in object_type
        or "official_programme_page_cache" in source
        or "official_programme_page_cache" in metadata_source
    )


def _is_general_or_weak_backup_doc(doc: Dict[str, Any]) -> bool:
    """
    True for sources that are useful as fallback but should not dominate
    programme-specific answers.
    """
    url = str(doc.get("source_url", "") or doc.get("url", "") or "").lower()
    title = str(doc.get("title", "") or "").lower()
    combined = f"{url} {title}"

    weak_fragments = [
        "student-exchange-programmes",
        "nomination-and-application",
        "studying-abroad",
        "pathways-abroad",
        "campus-stories",
        "application-via-uni-assist/faq",
        "advanced-masters-programmes/faq",
        "changing-study-programme",
    ]

    return any(fragment in combined for fragment in weak_fragments)


def _doc_unique_key(doc: Dict[str, Any]) -> str:
    """
    Stable key for deduplication.
    """
    return (
        str(doc.get("id", "") or "")
        or str(doc.get("source_url", "") or "")
        or str(doc.get("url", "") or "")
        or str(doc.get("title", "") or "")
        or str(doc.get("content", "") or "")[:120]
    )


def prepare_docs_for_staff_ui(
    *,
    draft: str,
    docs: List[Dict[str, Any]],
    context: Dict[str, Optional[str]],
    max_sources: int = 8,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Prepare final document list for UI display.

    Order:
        1. cited docs first,
        2. official programme-page cache docs,
        3. programme-matching docs,
        4. general HTW backup docs.

    Also remaps [Doc N] citations in the draft so that citation numbers still
    match the reordered source list shown in the UI.

    This is only a display/order cleanup. It does not change the answer content.
    """
    if not docs:
        return draft, docs

    cited_numbers = _extract_cited_doc_numbers_from_draft(draft)

    # Convert 1-based Doc numbers to 0-based indexes.
    cited_indexes = [
        number - 1
        for number in cited_numbers
        if 1 <= number <= len(docs)
    ]

    cited_docs: List[Dict[str, Any]] = [docs[index] for index in cited_indexes]

    cited_keys = {_doc_unique_key(doc) for doc in cited_docs}

    remaining_docs = [
        doc
        for doc in docs
        if _doc_unique_key(doc) not in cited_keys
    ]

    official_docs = [
        doc
        for doc in remaining_docs
        if _is_official_programme_cache_doc(doc)
    ]

    official_keys = {_doc_unique_key(doc) for doc in official_docs}

    remaining_docs = [
        doc
        for doc in remaining_docs
        if _doc_unique_key(doc) not in official_keys
    ]

    programme_docs = []
    general_docs = []

    for doc in remaining_docs:
        if _is_general_or_weak_backup_doc(doc):
            general_docs.append(doc)
        elif context.get("target_program") and _doc_programme_score(doc, context) >= 3:
            programme_docs.append(doc)
        else:
            general_docs.append(doc)

    ordered_docs_raw = cited_docs + official_docs + programme_docs + general_docs

    # Deduplicate while keeping order.
    ordered_docs: List[Dict[str, Any]] = []
    seen_keys = set()

    for doc in ordered_docs_raw:
        key = _doc_unique_key(doc)
        if key in seen_keys:
            continue

        seen_keys.add(key)
        ordered_docs.append(doc)

    # Limit visible sources, but keep all cited docs if possible.
    safe_limit = max(max_sources, len(cited_docs))
    ordered_docs = ordered_docs[:safe_limit]

    # Build old index -> new index map.
    old_to_new: Dict[int, int] = {}

    for old_index, old_doc in enumerate(docs):
        old_key = _doc_unique_key(old_doc)

        for new_index, new_doc in enumerate(ordered_docs):
            if _doc_unique_key(new_doc) == old_key:
                old_to_new[old_index + 1] = new_index + 1
                break

    def replace_doc_number(match: re.Match) -> str:
        old_number = int(match.group(1))
        new_number = old_to_new.get(old_number)

        if not new_number:
            return match.group(0)

        return f"[Doc {new_number}]"

    remapped_draft = re.sub(
        r"\[Doc\s*(\d+)\]",
        replace_doc_number,
        draft or "",
        flags=re.IGNORECASE,
    )

    return remapped_draft, ordered_docs

def add_reference_links_to_draft(
    draft: str,
    docs: List[Dict[str, Any]],
    topics: List[Dict[str, str]],
    context: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    """
    Add a short verification section for staff.

    The email body can still use [Doc 1] citation markers for grounding, but staff
    also need the actual URLs in the same draft view. This keeps the PoC
    staff-facing and easy to verify before sharing.
    """
    text = (draft or "").strip()

    # Avoid adding duplicate reference sections if the function is called twice.
    if "Reference links for staff verification:" in text:
        return text

    cited_numbers = []
    for m in re.finditer(r"\[(?:Doc\s*)?(\d+)\]", text):
        n = int(m.group(1))
        if n not in cited_numbers:
            cited_numbers.append(n)

    reference_lines: List[str] = []

    # Add matched programme URLs from the programme catalogue. This avoids saying
    # "check the programme website" without giving staff the exact URL.
    for line in programme_reference_lines(context or {}):
        reference_lines.append(line)

    # Add cited source URLs only. This keeps the draft short and avoids dumping all
    # retrieved sources into the student-ready part.
    for n in cited_numbers:
        if 1 <= n <= len(docs):
            doc = docs[n - 1]
            url = _doc_url(doc)
            if url:
                title = (doc.get("title", "") or doc.get("object_type", "") or "Source").strip()
                reference_lines.append(f"- [Doc {n}] {title}: {url}")

    # If the answer discusses uni-assist fees, include the official fee page as a
    # verification link. We do not hard-code the fee amount here unless the source
    # is indexed and retrieved.
    lower = text.lower()
    topic_ids = {t.get("topic_id", "") for t in topics}
    mentions_uni_assist_fee = (
        "application_fee" in topic_ids
        or "uni-assist" in lower and ("fee" in lower or "processing" in lower or "cost" in lower)
    )
    if mentions_uni_assist_fee and UNI_ASSIST_HANDLING_FEES_URL not in text:
        reference_lines.append(
            f"- Official uni-assist handling fees: {UNI_ASSIST_HANDLING_FEES_URL}"
        )

    # Add official verification links for common external checks. These help staff
    # verify the draft quickly without changing the retrieved HTW evidence.
    if "application_route" in topic_ids:
        reference_lines.append(f"- Hochschulstart: {HOCHSCHULSTART_URL}")
        reference_lines.append(f"- HTW Berlin application portal: {HTW_APPLICATION_PORTAL_URL}")

    if "qualification_recognition" in topic_ids:
        reference_lines.append(f"- anabin qualification database: {ANABIN_URL}")
        reference_lines.append(f"- DAAD admission database: {DAAD_ADMISSIONS_DATABASE_URL}")
        reference_lines.append(f"- HTW admission requirements: {HTW_ADMISSION_REQUIREMENTS_URL}")

    if not reference_lines:
        return text

    # Deduplicate while preserving order.
    seen = set()
    unique_lines = []
    for line in reference_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return text.rstrip() + "\n\nReference links for staff verification:\n" + "\n".join(unique_lines)


# ---------------------------------------------------------------------
# Metrics and quality
# ---------------------------------------------------------------------

DISCLAIMER_MARKERS = [
    "This draft was generated with AI support",
    "This mail was generated with AI support",
    "This email was generated with AI support",
]


def strip_disclaimer_for_metrics(text: str) -> str:
    """
    Remove the configurable AI disclaimer before quality phrase checks.

    The full draft can still include the disclaimer, but review metrics should
    evaluate only the actual generated answer body.
    """
    cleaned = text or ""
    for marker in DISCLAIMER_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
            break
    return cleaned

def extract_doc_citations(text: str) -> List[str]:
    seen = []
    for m in re.finditer(r"\[(?:Doc\s*)?(\d+)\]", text or ""):
        c = f"[Doc {m.group(1)}]"
        if c not in seen:
            seen.append(c)
    return seen


def has_bad_draft_phrase(draft: str) -> bool:
    body_only = strip_disclaimer_for_metrics(draft)
    lower = (body_only or "").lower()
    return any(p in lower for p in BAD_PHRASES)


def assess_email_quality(
    *,
    context: Dict[str, Optional[str]],
    topics: List[Dict[str, str]],
    docs: List[Dict[str, Any]],
    draft: str,
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Technical review signal.

    V5.3 adjustment:
    - keep the V4.6-style usable draft quality,
    - do not over-flag drafts just because they contain cautious wording,
    - still flag genuinely unsafe cases: no sources, no citations, low grounding,
      unknown programme for programme-specific questions, or wrong programme source.
    """
    reasons: List[str] = []
    citations = extract_doc_citations(draft)
    topic_ids = {t.get("topic_id", "") for t in topics}

    if not topics:
        reasons.append("No topics detected")

    if not docs:
        reasons.append("No sources retrieved")

    if not citations:
        reasons.append("No citations in draft")

    is_grounded = bool(validation.get("is_grounded", False)) if validation else False
    confidence = float(validation.get("confidence", 0.0)) if validation else 0.0

    if validation and not is_grounded:
        reasons.append("Draft not grounded according to validator")

    if confidence < 0.65:
        reasons.append("Low grounding confidence")

    bad_phrase = has_bad_draft_phrase(draft)

    # Bad phrase alone should not destroy the result if the draft is grounded
    # and cited. It should reduce the score, but not always force review.
    if bad_phrase and (not citations or not is_grounded):
        reasons.append("Draft contains uncertain or unsuitable wording")

    # Degree mismatch: target Master but draft appears to call the target programme
    # a Bachelor programme. Do not flag harmless phrases such as "completed a
    # Bachelor's degree".
    body_only = strip_disclaimer_for_metrics(draft)
    lower_draft = body_only.lower()
    
    if context.get("target_degree") == "Master":
        unsafe_bachelor_target = re.search(
            r"(interest(ed)? in|apply(ing)? for|admission to|the)\s+(a\s+)?bachelor('?s)?\s+programme",
            lower_draft,
        )
        if unsafe_bachelor_target:
            reasons.append("Possible degree mismatch")

    if len(topics) > 4:
        reasons.append("Too many detected topics")

    programme_specific_topics = {
        "application_deadline",
        "admission_requirements",
        "required_documents",
        "english_language_requirements",
        "german_language_requirements",
        "language_of_instruction",
        "study_format",
        "work_experience",
        "motivation_letter",
        "application_before_graduation",
    }

    match_score = context.get("target_program_match_score")
    try:
        match_score_float = float(match_score) if match_score else 0.0
    except Exception:
        match_score_float = 0.0

    if topic_ids & programme_specific_topics and not context.get("target_program"):
        reasons.append("Programme not confidently matched")
    elif topic_ids & programme_specific_topics and match_score_float and match_score_float < 0.70:
        reasons.append("Weak programme match")

    # Check whether programme-specific docs are likely from the detected programme.
    # This flags cases like PROITD answered using Information Technology Master pages.
    if context.get("target_program") and topic_ids & programme_specific_topics and docs:
        programme_scores = [_doc_programme_score(doc, context) for doc in docs]
        if max(programme_scores or [0]) < 3:
            reasons.append("No strong programme-specific source found")

    # Score is less aggressive than the previous latest version.
    score = 100

    if not topics:
        score -= 20

    if not docs:
        score -= 30

    if not citations:
        score -= 25

    if validation and not is_grounded:
        score -= 25

    if confidence < 0.65:
        score -= 15

    if bad_phrase:
        score -= 10

    if "Possible degree mismatch" in reasons:
        score -= 25

    if "Programme not confidently matched" in reasons:
        score -= 15

    if "Weak programme match" in reasons:
        score -= 10

    if "No strong programme-specific source found" in reasons:
        score -= 10

    if len(topics) > 4:
        score -= 10

    score = max(0, min(100, score))

    # Review only for real risk.
    hard_review_reasons = {
        "No topics detected",
        "No sources retrieved",
        "No citations in draft",
        "Draft not grounded according to validator",
        "Low grounding confidence",
        "Possible degree mismatch",
        "Programme not confidently matched",
    }

    review_required = any(reason in hard_review_reasons for reason in reasons)

    # Programme source weakness should normally be partial, not always hard review,
    # because general HTW application pages can still be useful.
    if "No strong programme-specific source found" in reasons and score < 80:
        review_required = True

    if score >= 85 and not review_required:
        label = "good"
    elif score >= 65:
        label = "partial"
    else:
        label = "review"

    return {
        "quality_score": score,
        "quality_label": label,
        "review_required": review_required,
        "review_reason": "; ".join(reasons),
        "citation_count": len(citations),
        "citations": citations,
        "bad_draft_phrase": bad_phrase,
    }