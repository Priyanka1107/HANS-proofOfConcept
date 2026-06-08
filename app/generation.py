# app/generation.py
from __future__ import annotations

from typing import Dict, Any, List
import re

from app.config import config

_DOC_CITE_RE = re.compile(r"\[(?:Doc\s*)?(\d+)\]")


def extract_citations(text: str) -> List[str]:
    """Return citations like ['[Doc 1]', '[Doc 2]'] in order of appearance."""
    seen = []
    for m in _DOC_CITE_RE.finditer(text or ""):
        num = m.group(1)
        token = f"[Doc {num}]"
        if token not in seen:
            seen.append(token)
    return seen


def _build_context(docs: List[Dict[str, Any]]) -> str:
    """Provide the LLM a numbered evidence pack. The model must cite [Doc N]."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        title = d.get("title", "")
        url = d.get("source_url", "")
        updated = d.get("last_updated", "")
        content = d.get("content", "")
        blocks.append(
            f"[Doc {i}] {title}\nURL: {url}\nLast updated: {updated}\nCONTENT:\n{content}\n"
        )
    return "\n---\n".join(blocks)


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    """
    Anthropic SDK returns usage as a Usage object (not JSON serializable).
    Convert it to a plain dict so FastAPI + json logging works.
    """
    if usage is None:
        return {}

    if isinstance(usage, dict):
        return usage

    # Pydantic model (common in modern SDKs)
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass

    # Generic object
    if hasattr(usage, "__dict__"):
        try:
            return dict(usage.__dict__)
        except Exception:
            pass

    return {"raw": str(usage)}


def generate_answer(query: str, docs: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Provider routing:
    - GENERATION_PROVIDER=anthropic -> Claude
    - GENERATION_PROVIDER=cohere    -> Cohere Chat
    - GENERATION_PROVIDER=extractive -> rule-based fallback
    """
    provider = (config.GENERATION_PROVIDER or "extractive").strip().lower()

#    if provider == "anthropic":
#        if not config.ANTHROPIC_API_KEY:
#            return _generate_extractive(query, docs, conflicts)
#        return _generate_with_anthropic(query, docs, conflicts)
#
#    if provider == "cohere":
#        if not config.COHERE_API_KEY:
#            return _generate_extractive(query, docs, conflicts)
#        return _generate_with_cohere(query, docs, conflicts)
#
#    return _generate_extractive(query, docs, conflicts)

    if provider == "mistral":
        if not getattr(config, "MISTRAL_API_KEY", ""):
            return _generate_extractive(query, docs, conflicts)
        return _generate_with_mistral(query, docs, conflicts)

    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            return _generate_extractive(query, docs, conflicts)
        return _generate_with_anthropic(query, docs, conflicts)

    if provider == "cohere":
        return _generate_with_cohere(query, docs, conflicts)

    return _generate_extractive(query, docs, conflicts)

def _generate_with_anthropic(query: str, docs: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Claude generation using Anthropic Messages API."""
    from anthropic import Anthropic

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    context = _build_context(docs)

    conflict_note = ""
    if conflicts:
        conflict_note = "NOTE: Conflicts were detected between sources. Prefer canonical and newest content.\n"

    system = (
        "You are HANS, the HTW Berlin portal assistant. "
        "Answer ONLY using the provided documents. "
        "Every factual claim must be supported by citations like [Doc 1]. "
        "If the documents do not contain the answer, say you don't have enough information and ask a clarifying question."
    )

    user = (
        f"{conflict_note}"
        f"USER QUESTION:\n{query}\n\n"
        f"EVIDENCE DOCUMENTS:\n{context}\n\n"
        "INSTRUCTIONS:\n"
        "1) Write a short answer.\n"
        "2) Include citations [Doc N] right after the sentence they support.\n"
        "3) Do not invent details.\n"
    )

    model = getattr(config, "GENERATION_MODEL", "") or "claude-3-haiku-20240307"
    print("ANTHROPIC MODEL USED:", model)

    resp = client.messages.create(
        model=model,
        max_tokens=900,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    answer = ""
    if resp and getattr(resp, "content", None):
        # content blocks, usually first contains .text
        answer = getattr(resp.content[0], "text", "") or ""

    usage_dict = _usage_to_dict(getattr(resp, "usage", None))
    return {"answer": answer, "usage": usage_dict}

def _generate_with_mistral(
    query: str,
    docs: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate an answer using the Mistral Chat API."""
    from mistralai import Mistral

    client = Mistral(api_key=config.MISTRAL_API_KEY)
    model = getattr(config, "GENERATION_MODEL", "") or "mistral-small-latest"

    context = _docs_context(docs)
    conflict_note = _conflict_note(conflicts)

    user = f"""
You are HANS, a staff-support assistant for HTW Berlin student services.

Use only the evidence below. If the evidence is missing, say that the available evidence is not sufficient.

Question:
{query}

Evidence:
{context}

Conflicts:
{conflict_note}

Write a clear, concise answer with citations like [Doc 1], [Doc 2].
""".strip()

    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user,
            }
        ],
        temperature=0.1,
        max_tokens=900,
    )

    answer = response.choices[0].message.content or ""

    return {
        "answer": answer.strip(),
        "provider": "mistral",
        "model": model,
        "usage": {},
    }

def _generate_with_cohere(query: str, docs: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohere Chat generation."""
    import cohere
    client = cohere.Client(api_key=config.COHERE_API_KEY)

    context = _build_context(docs)

    conflict_note = ""
    if conflicts:
        conflict_note = "NOTE: Conflicts were detected between sources. Prefer canonical and newest content.\n"

    user = (
        f"{conflict_note}"
        f"USER QUESTION:\n{query}\n\n"
        f"EVIDENCE DOCUMENTS:\n{context}\n\n"
        "INSTRUCTIONS:\n"
        "1) Write a short answer.\n"
        "2) Include citations [Doc N] right after the sentence they support.\n"
        "3) Do not invent details.\n"
    )

    resp = client.chat(
        model=config.COHERE_CHAT_MODEL,
        message=user,
        temperature=0.0,
    )

    answer = resp.text if resp and hasattr(resp, "text") else ""
    # Cohere usage is usually serializable, but keep consistent:
    usage = getattr(resp, "usage", {}) or {}
    return {"answer": answer, "usage": usage}


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", (s or "").lower())


def _generate_extractive(query: str, docs: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extractive fallback: score sentences by keyword overlap with query."""
    if not docs:
        return {"answer": "❌ No documents available to answer your question.", "usage": {}}

    query_tokens = set(_tokenize(query))

    scored = []
    for doc_idx, doc in enumerate(docs):
        content = doc.get("content", "") or ""
        sentences = re.split(r"[.!?]+", content)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            toks = _tokenize(sent)
            overlap = sum(1 for t in toks if t in query_tokens)
            score = overlap / max(1, len(toks))
            scored.append((score, overlap, doc_idx, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:6]

    if not top or all(s[0] < 0.05 for s in top):
        return {
            "answer": f"I don't have enough information to answer that from the documents. Could you clarify what exactly you mean by: {query}?",
            "usage": {},
        }

    lines = []
    for score, overlap, doc_idx, sent in top:
        lines.append(f"• {sent} [Doc {doc_idx + 1}]")

    return {"answer": "\n".join(lines), "usage": {}}


def validate_answer(answer: str, docs: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Rule-based validation."""
    citations = extract_citations(answer)
    has_citations = len(citations) > 0

    valid_nums = set(range(1, len(docs) + 1))
    cited_nums = [int(m.group(1)) for m in _DOC_CITE_RE.finditer(answer or "")]
    citations_valid = all(n in valid_nums for n in cited_nums) if cited_nums else False

    evidence_text = " ".join([d.get("content", "") for d in docs])
    ans_tokens = _tokenize(answer)
    ev_tokens = set(_tokenize(evidence_text))

    overlap = 0.0
    if ans_tokens:
        overlap = sum(1 for t in ans_tokens if t in ev_tokens) / max(1, len(ans_tokens))

    is_grounded = bool(has_citations and citations_valid and overlap >= 0.20)

    failure_type = None
    has_hallucinations = False
    if not has_citations:
        failure_type = "missing_citations"
        has_hallucinations = True
    elif not citations_valid:
        failure_type = "invalid_citations"
        has_hallucinations = True
    elif overlap < 0.20:
        failure_type = "low_evidence_overlap"
        has_hallucinations = True

    confidence = 0.9 if is_grounded else 0.4
    return {
        "is_grounded": is_grounded,
        "citations_valid": citations_valid,
        "has_hallucinations": has_hallucinations,
        "confidence": confidence,
        "failure_type": failure_type,
    }