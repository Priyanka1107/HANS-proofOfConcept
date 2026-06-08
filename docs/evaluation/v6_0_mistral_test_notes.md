\# V6.0 Mistral Test Notes



This file documents the first Mistral-integrated test version of the HANS staff-support PoC.



\## Purpose of this version



The purpose of this version is to test Mistral as the draft generation provider while keeping the rest of the RAG pipeline unchanged.



The current comparison is therefore:



| Component | v5.7 baseline | v6.0 Mistral test |

|---|---|---|

| Embeddings | Cohere | Cohere |

| Reranking | Cohere | Cohere |

| Vector store | Qdrant | Qdrant |

| Draft generation | Anthropic / Claude | Mistral |

| Topic detection | Local rules | Local rules |

| Programme matching | Local catalogue | Local catalogue |

| Review checks | Local rules | Local rules |

| Disclaimer | Config file | Config file |



Only the generation provider was changed in this first Mistral step.



\## Main observations



The Mistral integration works technically.



The backend can generate staff-facing email drafts using Mistral, and the UI can display the generated draft, detected topics, review status, source links and disclaimer.



The first tests showed that Mistral is faster than the previous generation setup.



\## Improvements observed



\- The Mistral API key is loaded from `.env`.

\- `.env.example` contains placeholders only.

\- The generated drafts are produced through the Mistral provider when `GENERATION\_PROVIDER=mistral`.

\- The disclaimer is loaded from `config/disclaimer.md`.

\- The duplicate disclaimer issue was fixed.

\- Deadline detection was made stricter, so “winter semester” alone does not create an unnecessary deadline topic.

\- The draft opening was improved so the email starts with a short polite sentence after the greeting.

\- The language-of-instruction topic was made stricter so “English-taught” as background wording does not automatically create an extra topic.



\## Issues observed after changing the model



Changing the generation model changed the draft behaviour.



Mistral follows retrieved evidence quite directly. This is useful in many cases, but it can also create issues when the retrieved documents contain general rules that need to be interpreted together with the extracted student profile.



One important example is the dual-citizenship case:



\- The student has French and Moroccan citizenship.

\- French citizenship means EU/EEA citizenship.

\- The student lives in Morocco and has a French Baccalaureat.

\- Mistral still produced an answer that sent the applicant via uni-assist because the school qualification was obtained outside Germany.

\- This is not safe enough because the HTW route depends on both citizenship and qualification background.



This issue needs further work. A deterministic guard is planned so that EU/EEA citizenship is not overridden only because the applicant lives outside the EU or has a foreign school certificate.



\## Current status



The current version is suitable for technical testing and review.



It is not yet final production quality.



The main next fixes are:



\- strengthen the EU/EEA citizenship route guard;

\- improve programme-specific retrieval for some programmes;

\- improve transfer and credit recognition handling;

\- compare Mistral output with the v5.7 baseline over all test cases;

\- decide whether Mistral Small is sufficient or whether Mistral Medium gives better draft quality.



\## Important note



The PoC remains Human-in-the-Loop.



Every generated response is a staff-facing draft and must be checked by staff before sending.

