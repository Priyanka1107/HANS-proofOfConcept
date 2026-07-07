# HANS Thesis Evaluation Freeze Summary — 2026-07-07

## Final freeze branch

Branch: thesis-final-freeze-20260707

This freeze records the successful final evaluation run of the HANS thesis PoC after resolving the backend/runtime issue that initially looked like a Cohere trial-limit problem.

## Technical clarification

During testing, the final PoC initially failed with Cohere 429-style errors. At first, this looked like a Cohere trial API limit issue. After step-by-step debugging, the actual situation was more nuanced:

- Cohere itself was working with the new API key.
- Qdrant was also working and returned vector-search results correctly.
- The original problem was partly caused by a stale backend process on port 8001.
- A separate SDK compatibility issue was also found in app/retrieval.py, where the installed Cohere SDK did not support the return_documents argument in rerank().
- Running the corrected backend on a clean port, 8008, resolved the runtime issue.
- The successful final run was therefore executed against the final PoC backend on port 8008 and the original HANS baseline on port 8080.

This means the earlier failures should not be interpreted as a failure of the PoC logic itself. They were caused by runtime/backend state and SDK compatibility during evaluation setup.

## Final successful evaluation run

Run timestamp: 20260707_215542

The combined evaluation completed successfully:

- Total rows: 80
- Successful rows: 80
- Failed rows: 0

Covered evaluation groups:

- Final PoC: 10 questions across 4 modes
- Original HANS: same 10 questions
- Email follow-up tests
- QA follow-up memory tests

## Response-time comparison

Original HANS average response time: 35.70 seconds

Final PoC main-mode averages:

| Mode | Average response time | Reduction compared with original HANS |
|---|---:|---:|
| Email Assistant — Claude | 5.24s | 85.3% faster |
| Email Assistant — Mistral | 2.56s | 92.8% faster |
| Baseline QA | 1.95s | 94.5% faster |
| Conversational QA | 2.24s | 93.7% faster |

The PoC therefore achieved a clear response-time improvement over the original HANS baseline. However, response time alone was not treated as the only success criterion.

## Mode-level interpretation

Email Assistant — Claude:
Best overall mode for staff-facing email support. It was slower than Mistral, but generally more cautious and safer in sensitive admissions cases. This is the preferred mode for thesis discussion when quality, reviewability, and staff use are prioritised.

Email Assistant — Mistral:
Fastest email-draft mode and often produced good drafts. However, it sometimes answered more confidently than the evidence justified. It is useful for speed comparison, but staff review remains important.

Baseline QA:
Fastest overall mode. Suitable for direct question answering, but not ideal for staff email workflows because it does not provide the same structured draft, follow-up handling, and review metadata as the Email Assistant.

Conversational QA:
Useful for memory/follow-up testing and faster than original HANS. Some follow-up cases still showed that grounded answers can miss the exact follow-up intent, so this mode should not be treated as fully reliable without review.

Original HANS:
Operationally stable in this run, but much slower and less aligned with the staff-facing email workflow. It does not provide the same quality score, review flag, detected topic list, staff draft structure, or follow-up metadata.

## Overall best mode

For the thesis PoC objective, the strongest overall mode is:

Email Assistant — Claude

Reason: It provides the best balance between staff-ready drafting, cautious wording, citations, topic handling, and review workflow. Mistral is faster, but Claude is safer for staff-facing admissions communication.

## Evaluation criteria used

The evaluation was guided by common criteria used in RAG and applied LLM evaluation literature, including:

- factual correctness
- retrieval relevance
- grounding and citation faithfulness
- answer completeness
- topic coverage
- response time / latency
- robustness under follow-up questions
- multilingual handling
- uncertainty handling
- source quality
- staff reviewability and human-in-the-loop suitability

In the analysis, these criteria were operationalised through automated logs and manual interpretation: HTTP success, response time, detected topics, grounding status, citation validity, quality score, review flags, source count, and qualitative draft assessment.

## Realistic conclusion

The final PoC run is suitable as thesis evaluation evidence. The system completed all requests successfully and showed substantial response-time improvements compared with original HANS. The Email Assistant modes also provided a more useful staff-facing workflow than the original baseline.

At the same time, the system should not be presented as fully automatic or error-free. Some drafts still required staff judgement because quality scores can be too optimistic, sources can sometimes be too generic, and a grounded answer may still miss part of the user's intent. The final conclusion should therefore remain realistic: the PoC improves speed, structure, and reviewability, but it should remain a staff-support tool with human approval before sending.
