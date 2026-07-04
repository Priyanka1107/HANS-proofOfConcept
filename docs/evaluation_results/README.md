\# Final Evaluation Test Results



This folder contains the final automated test evidence for the thesis evaluation version of the HANS PoC.



\## Included files



\### Smoke test



\- `script\_smoke\_test\_results.csv`

\- `script\_smoke\_test\_results.json`



The smoke test covers the core workflow:



\- English Email Assistant

\- German Email Assistant

\- Email follow-up/thread memory

\- Dual citizenship application-route handling

\- Conversational QA follow-up memory



Final result:



```text

8/8 passed



Mixed-version test v2

script\_mixed\_version\_test\_results\_v2.csv

script\_mixed\_version\_test\_results\_v2.json



The mixed-version test covers broader and more difficult scenarios:



English and German Email Assistant cases

Multi-topic emails

Programme matching

Thread memory

Unknown-programme review handling

Dual citizenship route safety

Required documents

Credit recognition

Semester contribution

Accommodation

Conversational QA

Baseline QA comparison

Mistral comparison



Final result:



Primary checks: 22/25 passed

All rows: 27/30 passed

Interpretation



These files are included as reproducibility evidence for the thesis. They show that the final HANS PoC version is technically stable enough for empirical evaluation.



The remaining failed cases are not hidden. They are useful for the thesis discussion because they show remaining limitations in programme-specific retrieval, incomplete answer coverage, and review-decision logic.



The system should be interpreted as a staff-review-oriented proof of concept, not as a production admissions system.

