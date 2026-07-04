\# HANS PoC – Final Thesis Evaluation Version



This branch freezes the final thesis evaluation version of the HANS PoC.



\## Purpose



HANS is a staff-review-oriented Retrieval-Augmented Generation (RAG) prototype for HTW Berlin student-support scenarios. It supports staff by drafting responses to student enquiries, retrieving supporting sources, preserving thread context, and flagging cases that need review.



This version is not intended to be a production admissions system. It is a thesis proof of concept for evaluating how a multilingual RAG assistant can support student-service communication while keeping a human-in-the-loop workflow.



\## Main features in this version



\### Four-mode evaluation setup



\- Email Assistant – Claude

\- Email Assistant – Mistral

\- Conversational QA

\- Baseline QA



\### Email Assistant workflow



\- Detects student enquiry topics

\- Matches target programme from the programme catalogue

\- Retrieves evidence per topic

\- Generates one staff-ready draft

\- Adds cited reference links for staff verification

\- Adds an AI/staff-review disclaimer

\- Logs provider, model, topics, quality score, citations, and response time



\### Thread memory



\- Stores student name, programme, degree level, country/background, citizenship category, and previous topics

\- Preserves context for follow-up emails

\- Prevents follow-up questions from accidentally switching degree level, for example Master to Bachelor



\### Multilingual support



\- Supports English and German emails

\- German emails receive German drafts and German reference headings

\- German topic patterns were added for deadlines, language, study format, fees, application route, documents, and admission topics



\### Safety and quality safeguards



\- Keeps EU/EEA citizenship separate from residence country

\- Avoids forcing uni-assist as the main route only because a student lives outside Germany or has a foreign certificate

\- Handles dual citizenship more safely

\- Uses cautious wording for International Baccalaureate and foreign school certificates

\- Prevents unsupported English-proof exemption claims

\- Separates application fees, tuition fees, and semester contributions

\- Adds official uni-assist handling-fee verification link when needed

\- Reduces quality score when external fee verification is required

\- Flags weak or uncited drafts for staff review



\### Evaluation scripts



\- Smoke script for core stability checks

\- Mixed-version script for broader evaluation across programmes, languages, modes, and edge cases



\## Final automated results



\### Smoke script



Result:



```text

8/8 passed



Mixed-version script v2 Result



Primary checks: 22/25 passed

All rows: 27/30 passed



Interpretation



The final version is suitable for thesis evaluation as a staff-review-oriented PoC. It performs well on core scenarios and demonstrates useful behaviour for multi-topic email drafting, programme matching, multilingual handling, thread memory, citations, and review safeguards.



The remaining weaknesses are useful for the empirical discussion. They mainly involve programme-specific edge cases, incomplete source coverage, imperfect review decisions, and occasional answer incompleteness. These limitations support the conclusion that the system is promising as a support tool, but not production-ready without staff verification and further source-quality improvements.



Final thesis position



This branch should be treated as the frozen implementation for final thesis evaluation.

