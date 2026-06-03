# HANS Staff-Support Proof of Concept

This repository contains the experimental HANS proof of concept for staff-facing student support.

The current focus is the Email Assistant mode. The PoC helps staff prepare draft responses to student emails. It supports:

- multi-topic email handling
- programme matching
- programme-specific retrieval
- source-based draft generation
- source display for staff verification
- review flags for uncertain cases
- lightweight thread memory for follow-up handling

The system does not send emails automatically. Every generated response remains a draft and must be reviewed by staff before sending.

## Main modes

- Baseline QA mode
- Conversational QA mode
- Email Assistant mode

The current PoC uses local/rule-based logic for language detection, topic detection, programme matching, source ordering, review checks, thread handling, disclaimer handling, and logging.

External cloud APIs are currently used only for model-heavy experimental parts such as embeddings, reranking, and draft generation. For production, these parts should be replaced by HTW-approved or locally hosted services, such as the planned Mistral/HTW LLM setup.

## Disclaimer

The disclaimer text is stored separately in:

```text
config/disclaimer.md