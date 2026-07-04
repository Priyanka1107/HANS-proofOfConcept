# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time
import logging
import re
from datetime import datetime

from app.programme_context import enrich_email_text_with_programme_context

from app.conversation import (
    get_or_create_session,
    detect_intent,
    extract_entities,
    reformulate_query_with_memory,
    update_session_memory,
    enhance_query_by_intent,
)

from app.retrieval import (
    generate_embedding,
    vector_search,
    keyword_search,
    reciprocal_rank_fusion,
    rerank_documents,
)

from app.generation import (
    generate_answer,
    validate_answer,
    extract_citations,
)

from app.config import config
from app.conflict import detect_and_resolve_conflicts
from app.fallback import (
    should_use_fallback,
    build_fallback_answer,
)
from app.disclaimer import append_disclaimer_to_draft

from app.logging_utils import (
    log_complete_query,
    add_to_regression_set,
    init_db_schema,
)

init_db_schema()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="HANS Optimized POC API",
    description="HTW AI Navigation System - Best-in-class RAG with grounding",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Shared API models
# ============================================================

class Source(BaseModel):
    id: str
    title: str
    url: str
    type: str
    last_updated: str
    excerpt: Optional[str] = None


class Validation(BaseModel):
    is_grounded: bool
    citations_valid: bool
    has_hallucinations: bool
    confidence: float
    failure_type: Optional[str] = None


# ============================================================
# Query / QA models
# ============================================================

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None
    top_k: int = 5
    session_id: Optional[str] = None

    # Supported values used by the evaluation runner:
    # - baseline_qa
    # - conversational_qa
    #
    # Older UI values are also accepted:
    # - baseline
    # - conversation
    # - conversational
    mode: Optional[str] = "conversation"


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    sources: List[Source]
    validation: Validation
    timing: Dict[str, float]
    had_conflicts: bool
    conflicts: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    standalone_query: Optional[str] = None
    detected_intent: Optional[str] = None
    enhanced_query: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    used_memory: bool = False
    mode: Optional[str] = None


class FeedbackRequest(BaseModel):
    query_id: int
    helpful: bool
    comment: Optional[str] = None
    add_to_regression: bool = False


# ============================================================
# Email Assistant models
# ============================================================

class EmailAssistantRequest(BaseModel):
    email_text: str
    session_id: Optional[str] = None
    language: Optional[str] = None
    top_k: int = 3

    # Supported values used by the evaluation runner:
    # - email_claude
    # - email_mistral
    mode: Optional[str] = "email_claude"
    test_id: Optional[str] = None
    evaluation_run_id: Optional[str] = None
    generation_provider: Optional[str] = None
    generation_model: Optional[str] = None

    # V5 email-thread simulation fields.
    # They allow context to be kept across emails without changing the V4.6.2 answer architecture.
    student_email: Optional[str] = None
    subject: Optional[str] = None
    thread_id: Optional[str] = None
    email_id: Optional[str] = None


class EmailTopic(BaseModel):
    topic_id: str
    label: str
    query: str
    intent: Optional[str] = None
    source_count: int = 0


class EmailQuality(BaseModel):
    quality_score: int
    quality_label: str
    review_required: bool
    review_reason: str
    citation_count: int
    citations: List[str]
    bad_draft_phrase: bool


class EmailAssistantResponse(BaseModel):
    is_followup: bool
    flagged_for_human: bool
    followup_type: Optional[str] = None
    thread_id: Optional[str] = None
    thread_context: Optional[Dict[str, Any]] = None
    email_context: Dict[str, Optional[str]]
    detected_topics: List[EmailTopic]
    staff_draft: str
    citations: List[str]
    sources: List[Source]
    validation: Validation
    quality: EmailQuality
    timing: Dict[str, float]
    session_id: Optional[str] = None
    mode: Optional[str] = None
    test_id: Optional[str] = None
    generation_provider: Optional[str] = None
    generation_model: Optional[str] = None


# ============================================================
# Health and metadata endpoints
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "HANS Optimized POC",
        "version": "1.0.0",
        "configuration": {
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_model": (
                config.COHERE_EMBED_MODEL
                if config.EMBEDDING_PROVIDER == "cohere"
                else "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "embedding_dimension": config.EMBEDDING_DIMENSION,
            "qdrant_collection": config.QDRANT_COLLECTION,
            "rerank_model": config.RERANK_MODEL,
            "generation_provider": config.GENERATION_PROVIDER,
            "generation_model": getattr(config, "GENERATION_MODEL", None),
        },
        "models": {
            "embeddings": (
                config.COHERE_EMBED_MODEL
                if config.EMBEDDING_PROVIDER == "cohere"
                else "local/sentence-transformers/all-MiniLM-L6-v2"
            ),
            "retrieval": "qdrant-dense",
            "reranking": config.RERANK_MODEL if config.COHERE_API_KEY else "disabled",
            "generation": (
                getattr(config, "GENERATION_MODEL", "extractive-fallback")
                if config.GENERATION_PROVIDER == "anthropic"
                else "extractive-fallback"
            ),
            "validation": "rule-based",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "generation_provider": config.GENERATION_PROVIDER,
        "components": {
            "api": "operational",
            "qdrant": "operational" if config.QDRANT_URL else "not_configured",
            "cohere": "operational" if config.COHERE_API_KEY else "not_configured",
            "embeddings": "operational",
            "generation": "operational",
        },
    }


# ============================================================
# Helper functions
# ============================================================

def _source_from_doc(doc: Dict[str, Any]) -> Source:
    return Source(
        id=str(doc.get("id", "")),
        title=doc.get("title", "") or "",
        url=doc.get("source_url", "") or doc.get("url", "") or "",
        type=doc.get("object_type", "") or doc.get("type", "") or "",
        last_updated=doc.get("last_updated", "") or "",
        excerpt=(doc.get("content", "") or doc.get("chunk_text", "") or "")[:250],
    )


def _deduplicate_docs(docs: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []

    for doc in docs:
        doc_id = str(doc.get("id", ""))
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc)

        if len(unique) >= limit:
            break

    return unique

def _is_german_reply(email_context: Dict[str, Any]) -> bool:
    return str(email_context.get("reply_language", "")).lower().startswith("de")


def _cleanup_staff_draft_footer(staff_draft: str, email_context: Dict[str, Any]) -> str:
    """
    Clean language-mismatched footer text before appending reference links.

    The disclaimer can stay in English because it is for staff review.
    This function only removes duplicated English closing lines from German drafts.
    """
    draft = (staff_draft or "").rstrip()

    if not _is_german_reply(email_context):
        return draft

    # Remove English closing that the model may still add after a German closing.
    english_closing_patterns = [
        r"\n\nKind regards,\s*\nHTW Berlin Student Services\s*$",
        r"\n\nBest regards,\s*\nHTW Berlin Student Services\s*$",
        r"\n\nSincerely,\s*\nHTW Berlin Student Services\s*$",
    ]

    for pattern in english_closing_patterns:
        draft = re.sub(pattern, "", draft, flags=re.IGNORECASE).rstrip()

    return draft


def _append_staff_reference_links(
    staff_draft: str,
    email_context: Dict[str, Any],
    matched_programme_url: str,
    matched_programme_application_url: str,
) -> str:
    """
    Append staff verification links using the reply language for the heading.

    The links are still only staff-facing support links.
    """
    draft = (staff_draft or "").rstrip()

    if not matched_programme_url:
        return draft

    already_has_links = (
        "Reference links for staff verification" in draft
        or "Referenzlinks zur Prüfung durch Mitarbeitende" in draft
        or "Programme page for staff verification" in draft
        or "Programmseite zur Prüfung" in draft
    )

    if already_has_links:
        return draft

    if _is_german_reply(email_context):
        lines = [
            "",
            "",
            "Referenzlinks zur Prüfung durch Mitarbeitende:",
            f"- Programmseite: {matched_programme_url}",
        ]
        if matched_programme_application_url:
            lines.append(f"- Bewerbungsseite des Studiengangs: {matched_programme_application_url}")
    else:
        lines = [
            "",
            "",
            "Reference links for staff verification:",
            f"- Programme page: {matched_programme_url}",
        ]
        if matched_programme_application_url:
            lines.append(f"- Programme application page: {matched_programme_application_url}")

    return draft + "\n".join(lines)

def _detect_input_language(text: str, requested_language: Optional[str] = None) -> str:
    """
    Detect the input language for UI/evaluation mode.

    This is used only to control the output language of staff-facing drafts.
    It does not replace multilingual retrieval.
    """
    if requested_language:
        return requested_language

    try:
        from langdetect import detect as lang_detect
        return lang_detect(text or "")
    except Exception:
        return "en"


def _reply_language_from_input(input_language: str) -> str:
    return "de" if str(input_language or "").lower().startswith("de") else "en"


# ============================================================
# Query / QA endpoint
# ============================================================

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    timing: Dict[str, float] = {}
    raw_query = req.query
    selected_mode = (req.mode or "conversation").strip().lower()
    conversation_mode = selected_mode in {"conversation", "conversational", "conversational_qa"}

    try:
        session_id = None
        memory = None
        detected_intent = None
        entities: Dict[str, Any] = {}
        standalone_query = raw_query
        used_memory = False

        if conversation_mode:
            session_id, memory = get_or_create_session(req.session_id)

            start = time.time()
            # Use memory only to reformulate the query. This keeps the existing
            # conversational behaviour unchanged.
            entities = extract_entities(raw_query)
            standalone_query = reformulate_query_with_memory(raw_query, memory)
            used_memory = standalone_query.strip() != raw_query.strip()
            timing["conversation_preprocessing"] = time.time() - start

        query_text = standalone_query

        # Detect intent after reformulation, so short follow-up questions such as
        # "And what is the deadline?" inherit the full context first.
        start = time.time()
        detected_intent = detect_intent(query_text)
        query_entities = extract_entities(query_text)

        # Merge entities found in raw and standalone queries; standalone wins when available.
        entities = {**entities, **{k: v for k, v in query_entities.items() if v}}

        enhanced_query = enhance_query_by_intent(query_text, detected_intent)
        timing["intent_detection"] = time.time() - start

        logger.info("Processing query: %s...", query_text[:100])
        logger.info("Detected intent: %s; enhanced query: %s...", detected_intent, enhanced_query[:140])

        # Step 1: Language detection
        start = time.time()
        if not req.language:
            from langdetect import detect
            language = detect(query_text)
        else:
            language = req.language
        timing["language_detection"] = time.time() - start

        # Step 2: Generate query embedding
        start = time.time()
        query_embedding = generate_embedding(enhanced_query)
        timing["embedding"] = time.time() - start
        logger.info("Generated embedding in %.3fs", timing["embedding"])

        # Step 3: Retrieval
        start = time.time()
        vector_results = vector_search(query_embedding, top_k=20)
        keyword_results = keyword_search(enhanced_query, language=language, top_k=20)
        timing["retrieval"] = time.time() - start
        logger.info("Retrieved %s vector + %s keyword results", len(vector_results), len(keyword_results))

        # Step 4: Fusion
        start = time.time()
        fused_results = reciprocal_rank_fusion(vector_results, keyword_results)
        timing["fusion"] = time.time() - start

        # Step 5: Reranking
        start = time.time()
        reranked = rerank_documents(enhanced_query, fused_results[:10], top_k=req.top_k)
        timing["reranking"] = time.time() - start
        logger.info("Reranked to top %s documents", len(reranked))

        # Step 6: Conflict handling
        start = time.time()
        final_docs, conflicts = detect_and_resolve_conflicts(reranked)
        timing["conflict_handling"] = time.time() - start

        if conflicts:
            logger.warning("Detected %s conflicts", len(conflicts))

        # Step 7: Fallback check + answer generation
        start = time.time()
        fallback_used, fallback_reason = should_use_fallback(
            query=query_text,
            docs=final_docs,
            intent=detected_intent or "general_info",
            entities=entities,
        )
        timing["fallback_check"] = time.time() - start

        start = time.time()
        if fallback_used:
            answer_data = {
                "answer": build_fallback_answer(
                    query=query_text,
                    intent=detected_intent or "general_info",
                    reason=fallback_reason,
                    entities=entities,
                ),
                "usage": {"fallback_used": True, "fallback_reason": fallback_reason},
            }
            logger.info("Fallback used: %s", fallback_reason)
        else:
            answer_data = generate_answer(query_text, final_docs, conflicts)
        timing["generation"] = time.time() - start
        logger.info("Generated answer in %.3fs", timing["generation"])

        # Update memory only in conversational mode.
        if conversation_mode and memory is not None:
            update_session_memory(
                memory=memory,
                raw_query=raw_query,
                standalone_query=standalone_query,
                intent=detected_intent,
                entities=entities,
                answer=answer_data["answer"],
            )

        # Step 8: Validate answer
        start = time.time()
        validation = validate_answer(
            answer_data["answer"],
            final_docs,
            query_text,
        )
        if fallback_used:
            # Fallback answers intentionally have no citations because they state
            # that the evidence was insufficient. Mark this explicitly so thesis
            # evaluation can distinguish safe fallback from hallucination.
            validation = {
                "is_grounded": False,
                "citations_valid": False,
                "has_hallucinations": False,
                "confidence": 0.6,
                "failure_type": fallback_reason or "fallback_used",
            }
        timing["validation"] = time.time() - start

        # Step 9: Logging
        total_time = sum(timing.values())
        log_entry = log_complete_query(
            query_text,
            language,
            vector_results,
            keyword_results,
            fused_results,
            final_docs,
            conflicts,
            answer_data,
            validation,
            timing,
            original_query=raw_query,
            standalone_query=standalone_query,
            enhanced_query=enhanced_query,
            detected_intent=detected_intent,
            mode=selected_mode,
            used_memory=used_memory,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

        if validation.get("failure_type"):
            logger.warning("Validation failed: %s", validation["failure_type"])
            add_to_regression_set(log_entry)

        response = QueryResponse(
            answer=answer_data["answer"],
            citations=extract_citations(answer_data["answer"]),
            sources=[
                Source(
                    id=doc["id"],
                    title=doc["title"],
                    url=doc["source_url"],
                    type=doc["object_type"],
                    last_updated=doc["last_updated"],
                    excerpt=doc["content"][:200],
                )
                for doc in final_docs
            ],
            validation=Validation(**validation),
            timing=timing,
            had_conflicts=len(conflicts) > 0,
            conflicts=conflicts if conflicts else None,
            session_id=session_id,
            standalone_query=standalone_query,
            detected_intent=detected_intent,
            enhanced_query=enhanced_query,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            used_memory=used_memory,
            mode=selected_mode,
        )

        logger.info("Query completed in %.3fs", total_time)
        return response

    except Exception as e:
        logger.error("Error processing query: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Email Assistant endpoint
# ============================================================

@app.post("/email", response_model=EmailAssistantResponse)
async def email_assistant_endpoint(req: EmailAssistantRequest):
    """
    Staff-facing email assistant endpoint.

    V5.1 design:
    - keep the existing V4.6/V5 email draft quality,
    - add programme catalogue context before topic detection and retrieval,
    - use the matched programme name and URL in the email context,
    - avoid asking for the exact programme title again when it was already detected.
    """
    from app.multitopic import (
        build_followup_flag_message,
        extract_email_context,
        detect_topics,
        generate_staff_email_draft,
        assess_email_quality,
        filter_docs_for_programme,
        prepare_docs_for_staff_ui,
    )
    from app.programme_official_evidence import get_official_programme_docs
    from app.email_thread_memory import (
        build_thread_key,
        load_thread_context,
        save_thread_context,
        classify_followup_email,
        merge_context_with_thread,
    )

    timing: Dict[str, float] = {}
    overall_start = time.time()

    selected_mode = (req.mode or "email_claude").strip().lower()

    if selected_mode in {"email_mistral", "mistral"}:
        generation_provider = req.generation_provider or "mistral"
        generation_model = req.generation_model or "mistral-small-latest"
    else:
        generation_provider = req.generation_provider or "anthropic"
        generation_model = req.generation_model or "claude-haiku-4-5"

    try:
        # Email Assistant does not use QA session memory.
        # It uses thread_id/email_thread_memory for email follow-up handling.
        session_id = req.session_id
        thread_key = build_thread_key(
            thread_id=req.thread_id,
            student_email=req.student_email,
            subject=req.subject,
        )
        previous_thread_context = load_thread_context(thread_key)

        # ------------------------------------------------------------
        # V5.1 programme catalogue bridge
        # ------------------------------------------------------------
        # The programme catalogue is built in data/programme_catalog.json.
        # This block makes the live email endpoint actually use that catalogue.
        #
        # It does not hardcode facts such as deadlines, language proof, fees,
        # or documents. It only adds programme identity and programme URL
        # context before topic detection and retrieval.
        # ------------------------------------------------------------
        start = time.time()
        programme_context = enrich_email_text_with_programme_context(
            email_text=req.email_text,
            subject=req.subject or "",
        )

        original_email_text = req.email_text
        detected_input_language = _detect_input_language(original_email_text, req.language)
        reply_language = _reply_language_from_input(detected_input_language)

        email_text_for_processing = programme_context.get(
            "enriched_email_text",
            req.email_text,
        )

        matched_programme = programme_context.get("program_name", "") or ""
        matched_programme_url = programme_context.get("url", "") or ""
        matched_programme_application_url = programme_context.get("application_url", "") or ""

        timing["programme_catalogue_match"] = time.time() - start

        logger.info(
            "Programme catalogue match: %s | URL: %s",
            matched_programme or "None",
            matched_programme_url or "None",
        )

        # Step 1: follow-up detection for email workflow.
        # Clarification/complaint follow-ups are sent to staff directly.
        # Normal follow-up topics can still be drafted using thread context.
        start = time.time()
        followup_type = classify_followup_email(
            original_email_text,
            has_thread_memory=bool(previous_thread_context),
        )
        followup = followup_type != "new_enquiry"
        timing["followup_detection"] = time.time() - start

        if followup_type == "clarification_or_complaint":
            draft = build_followup_flag_message(original_email_text)
            draft = append_disclaimer_to_draft(draft, reply_language=reply_language)

            validation = {
                "is_grounded": False,
                "citations_valid": False,
                "has_hallucinations": False,
                "confidence": 0.0,
                "failure_type": "followup_requires_human_review",
            }
            quality = {
                "quality_score": 0,
                "quality_label": "review",
                "review_required": True,
                "review_reason": "Follow-up email requires human review",
                "citation_count": 0,
                "citations": [],
                "bad_draft_phrase": False,
            }
            timing["total"] = time.time() - overall_start

            return EmailAssistantResponse(
                is_followup=True,
                flagged_for_human=True,
                followup_type=followup_type,
                thread_id=thread_key,
                thread_context=previous_thread_context,
                email_context={
                    "matched_programme": matched_programme,
                    "matched_programme_url": matched_programme_url,
                    "matched_programme_application_url": matched_programme_application_url,
                    "input_language": detected_input_language,
                    "reply_language": reply_language,
                },
                detected_topics=[],
                staff_draft=draft,
                citations=[],
                sources=[],
                validation=Validation(**validation),
                quality=EmailQuality(**quality),
                timing=timing,
                session_id=session_id,
                mode=selected_mode,
                test_id=req.test_id,
                generation_provider=generation_provider,
                generation_model=generation_model,
            )

        # Step 2: understand email context and topics.
        #
        # Important:
        # Use email_text_for_processing here, not only req.email_text.
        # This allows detect_topics() and extract_email_context() to see
        # the programme catalogue context.
        start = time.time()
        email_context = extract_email_context(email_text_for_processing)

        # Preserve the input language for staff-draft generation.
        # This controls output language; retrieval remains multilingual.
        email_context["input_language"] = detected_input_language
        email_context["reply_language"] = reply_language

        # Add programme catalogue result explicitly into the context.
        # This makes it visible to draft generation and thread memory.
        if matched_programme:
            email_context["matched_programme"] = matched_programme
            email_context["target_programme"] = matched_programme
            email_context["target_program"] = matched_programme

        if matched_programme_url:
            email_context["matched_programme_url"] = matched_programme_url
            email_context["programme_url"] = matched_programme_url
            email_context["target_program_url"] = matched_programme_url

        if matched_programme_application_url:
            email_context["matched_programme_application_url"] = matched_programme_application_url
            email_context["programme_application_url"] = matched_programme_application_url
            email_context["target_program_application_url"] = matched_programme_application_url

        if followup_type == "followup_new_topic":
            email_context = merge_context_with_thread(email_context, previous_thread_context)

        detected_topics = detect_topics(email_text_for_processing, email_context, max_topics=4)
        timing["email_understanding"] = time.time() - start

        logger.info("Email context: %s", email_context)
        logger.info("Detected topics: %s", [t["topic_id"] for t in detected_topics])

        # Step 3: retrieve evidence per topic, but do not generate per-topic answers.
        all_docs: List[Dict[str, Any]] = []
        topic_models: List[EmailTopic] = []

        for topic in detected_topics:
            topic_start = time.time()

            # ------------------------------------------------------------
            # V5.1 programme-aware retrieval query
            # ------------------------------------------------------------
            # Keep the existing topic query, but enrich it with programme
            # identity and URL when a catalogue match exists.
            # ------------------------------------------------------------
            query_text = topic["query"]

            if matched_programme:
                programme_parts = [
                    f"Programme: {matched_programme}",
                    f"Student question topic: {query_text}",
                    "Use programme-specific HTW Berlin information where available.",
                ]

                if matched_programme_url:
                    programme_parts.append(f"Programme URL: {matched_programme_url}")

                if matched_programme_application_url:
                    programme_parts.append(f"Programme application URL: {matched_programme_application_url}")

                query_text = "\n".join(programme_parts)

            intent = detect_intent(query_text)
            enhanced_query = enhance_query_by_intent(query_text, intent)

            if matched_programme:
                # Make sure the programme name remains in the final retrieval query
                # even if enhance_query_by_intent() makes the query more generic.
                enhanced_query = f"{matched_programme} {enhanced_query}"

                if matched_programme_url:
                    enhanced_query = f"{enhanced_query} {matched_programme_url}"

            # Prefer the student's input language for keyword retrieval.
            # The vector embedding remains multilingual.
            language = req.language or detected_input_language or "en"

            # Embedding + retrieval + reranking
            query_embedding = generate_embedding(enhanced_query)
            vector_results = vector_search(query_embedding, top_k=12)
            keyword_results = keyword_search(enhanced_query, language=language, top_k=12)
            fused_results = reciprocal_rank_fusion(vector_results, keyword_results)
            reranked = rerank_documents(enhanced_query, fused_results[:8], top_k=max(1, req.top_k))

            final_docs, conflicts = detect_and_resolve_conflicts(reranked)

            # V5.3 quality recovery:
            # Prefer programme-specific evidence for programme-specific topics.
            # This keeps the improved topic detection but avoids using another programme's
            # page as evidence, e.g. PROITD answered from Information Technology Master.
            final_docs = filter_docs_for_programme(
                docs=final_docs,
                context=email_context,
                topic_id=topic["topic_id"],
                min_keep=3,
            )

            # Add trusted official programme-page snippets from local cache.
            # This fixes cases where general HTW pages are retrieved but the programme
            # page contains more specific fee/admission/language information.
            official_programme_docs = get_official_programme_docs(
                context=email_context,
                topic_id=topic["topic_id"],
                limit=3,
            )

            if official_programme_docs:
                final_docs = official_programme_docs + final_docs

            all_docs.extend(final_docs)

            topic_models.append(
                EmailTopic(
                    topic_id=topic["topic_id"],
                    label=topic["label"],
                    query=query_text,
                    intent=intent,
                    source_count=len(final_docs),
                )
            )

            timing[f"retrieval_{topic['topic_id']}"] = time.time() - topic_start

        final_docs = _deduplicate_docs(all_docs, limit=10)

        # Step 4: one final staff-ready draft using all evidence.
        #
        # Important:
        # original_email should remain the real student email, so the draft
        # does not include the internal catalogue block as if the student wrote it.
        #
        # The catalogue details are passed through context.
        start = time.time()
        staff_draft = generate_staff_email_draft(
            original_email=original_email_text,
            context=email_context,
            topics=detected_topics,
            docs=final_docs,
            generation_provider=generation_provider,
            generation_model=generation_model,
        )
        timing["draft_generation"] = time.time() - start

        # V5.4 UI source cleanup:
        # Reorder sources for staff display and remap [Doc N] citations so that
        # citations in the draft still match the source cards shown in the UI.
        staff_draft, final_docs = prepare_docs_for_staff_ui(
            draft=staff_draft,
            docs=final_docs,
            context=email_context,
            max_sources=8,
        )

        # Step 5: clean footer and append programme URL for staff verification.
        # The disclaimer may stay in English for staff review, but the draft-facing
        # footer/reference heading should follow the email reply language.
        staff_draft = _cleanup_staff_draft_footer(
            staff_draft=staff_draft,
            email_context=email_context,
        )

        staff_draft = _append_staff_reference_links(
            staff_draft=staff_draft,
            email_context=email_context,
            matched_programme_url=matched_programme_url,
            matched_programme_application_url=matched_programme_application_url,
        )

        # Keep a copy of the draft before adding the final disclaimer.
        # Validation and quality checks should assess only the generated answer
        # and evidence support, not the required staff-facing disclaimer text.
        staff_draft_for_validation = staff_draft

        # Add configurable AI disclaimer as the final part of the staff email draft.
        # The text is stored in config/disclaimer.md and can be changed without
        # modifying code or prompts.
        staff_draft = append_disclaimer_to_draft(staff_draft, reply_language=reply_language)

        # Step 6: validate final draft once.
        start = time.time()
        validation = validate_answer(
            staff_draft_for_validation,
            final_docs,
            original_email_text,
        )
        timing["validation"] = time.time() - start

        # Step 7: quality and review decision.
        start = time.time()
        quality = assess_email_quality(
            context=email_context,
            topics=detected_topics,
            docs=final_docs,
            draft=staff_draft_for_validation,
            validation=validation,
        )

        # Extra safety:
        # If the answer still contains weak wording about checking the programme website,
        # mark it for staff review. This does not block draft generation. It only tells
        # staff that the draft is not ready for automatic use.
        weak_programme_phrases = [
            "please check the dedicated programme website",
            "please check the specific programme website",
            "please check the programme website",
            "contact us with the exact programme title",
            "these details should be checked on the programme",
            "not included in the general application guidelines",
        ]

        if any(phrase in staff_draft.lower() for phrase in weak_programme_phrases):
            quality["review_required"] = True
            quality["quality_label"] = "partial"
            if quality.get("review_reason"):
                quality["review_reason"] += "; Programme-specific details need staff verification"
            else:
                quality["review_reason"] = "Programme-specific details need staff verification"

        timing["quality_assessment"] = time.time() - start

        # Do not update QA session memory from Email Assistant mode.
        # Email Assistant uses email_thread_memory only. This keeps the thesis
        # modes clean: Conversational QA uses query_session_memory, while Email
        # Assistant uses thread_id-based email memory.

        # Persist lightweight thread memory for future emails in the same thread.
        save_thread_context(
            thread_key,
            student_email=req.student_email,
            subject=req.subject,
            email_context=email_context,
            detected_topics=detected_topics,
            staff_draft=staff_draft,
            quality=quality,
        )

        timing["total"] = time.time() - overall_start

        return EmailAssistantResponse(
            is_followup=followup,
            flagged_for_human=quality["review_required"],
            followup_type=followup_type,
            thread_id=thread_key,
            thread_context=previous_thread_context,
            email_context=email_context,
            detected_topics=topic_models,
            staff_draft=staff_draft,
            citations=quality["citations"],
            sources=[_source_from_doc(doc) for doc in final_docs],
            validation=Validation(**validation),
            quality=EmailQuality(**quality),
            timing=timing,
            session_id=session_id,
            mode=selected_mode,
            test_id=req.test_id,
            generation_provider=generation_provider,
            generation_model=generation_model,
        )

    except Exception as e:
        logger.error("Error processing email assistant request: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Feedback and metrics endpoints
# ============================================================

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    try:
        logger.info(
            "Feedback received for query %s: %s",
            req.query_id,
            "helpful" if req.helpful else "not helpful",
        )

        if req.add_to_regression:
            logger.info("Added query %s to regression set", req.query_id)

        return {
            "status": "success",
            "message": "Feedback recorded",
        }

    except Exception as e:
        logger.error("Error recording feedback: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    try:
        return {
            "total_queries": 0,
            "avg_response_time": 0,
            "grounding_rate": 0,
            "error_rate": 0,
            "satisfaction_rate": 0,
        }
    except Exception as e:
        logger.error("Error fetching metrics: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
