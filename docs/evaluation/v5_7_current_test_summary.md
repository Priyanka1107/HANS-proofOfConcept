# V5.7 Current PoC Test Summary

This file summarises the current test status of the HANS staff-facing email assistant PoC.

The raw CSV, JSONL and Excel result files are not committed to GitHub. They are generated local evaluation outputs. This summary is kept in the repository to show the main result status, key observations, and known limitations in a clean format.

## Test setup

Version label: `v5_7_multitopic_final_check`

Local result file used for this summary:

```text
evaluation/email_assistant_v5_afterFix.csv
```

This raw result file is not committed to GitHub.

Main changes tested:

* Added new edge cases for EU citizenship, residence outside the EU, foreign school certificates, IB diploma, and transfer/credit recognition.
* Improved distinction between citizenship and residence country.
* Improved application fee handling, so application fees are not treated as tuition fees or semester contribution.
* Added configurable AI disclaimer at the end of generated drafts.
* Excluded the disclaimer from review-metric phrase checks.
* Improved thread follow-up detection.
* Added stable test metadata for student email, subject, and thread ID.
* Improved focus on detected topics, so the draft should avoid unnecessary extra sections where possible.

## Multi-topic email assistant test

The multi-topic test checks whether the system can process student emails with several questions in one message and prepare one staff-facing draft.

Metrics recorded:

* expected topics
* detected topics
* missing topics
* extra topics
* topic coverage
* draft quality label
* review required
* review reason
* groundedness
* citation count
* source count
* response time
* generated staff draft

### Summary statistics

| Metric                 |       Result |
| ---------------------- | -----------: |
| Total test cases       |           23 |
| Average topic coverage |       84.06% |
| Grounded drafts        |        22/23 |
| Average response time  | 6.42 seconds |
| Average citation count |         2.65 |
| Average source count   |         7.17 |

### Quality labels

| Quality label | Count |
| ------------- | ----: |
| good          |     7 |
| partial       |    12 |
| review        |     4 |

### Review required

| Review required | Count |
| --------------- | ----: |
| Yes             |    16 |
| No              |     7 |

## Current observations

The strongest part of the current PoC is the multi-topic drafting workflow. The system can detect several topics in one email and generate one combined staff-facing draft. This matches the main direction of the PoC.

The result is useful for staff review, but it is not production-ready. Many drafts are still marked for review. This is acceptable for the current staff-facing PoC because the system is designed to support staff, not to send answers automatically.

## Stronger behaviour observed

* The system now distinguishes EU citizenship from residence country better.
* A French citizen residing in Morocco is no longer automatically treated as a non-EU applicant.
* Clarification or complaint follow-ups are now flagged for human review.
* The generated drafts include staff-facing source links and the configurable AI disclaimer.
* Test result columns such as student email, subject, thread ID, and follow-up type are now filled.
* Most generated drafts are grounded in retrieved sources.
* The system handles several multi-topic emails with usable staff-facing drafts.

## Remaining weak points

The latest tests still show some important limitations:

* Some programme names are still too vague to match confidently, for example “Bachelor’s in Business”.
* Some student wording can still trigger extra topics, such as language of instruction, even when the student only describes the programme as English-taught.
* Some programme-specific application pages are still not retrieved strongly enough.
* Application fee questions are improved, but they still need careful checking when retrieved sources mix application fees, tuition fees, and semester fees.
* Pending transcript/final result cases depend strongly on whether the programme-specific application page is available in the retrieved sources.
* General fallback sources are useful for safety, but they can produce incomplete drafts if the specific programme page is missing.
* Some transfer and credit recognition cases still need better topic detection and better source retrieval.

## Cases with missing expected topics

The following cases still had missing expected topics in the latest run:

| Case ID  | Missing topic(s)                                                   | Main observation                                                                                       |
| -------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `pd_003` | `required_documents`                                               | Required documents were not detected.                                                                  |
| `pd_004` | `required_documents`                                               | Required documents were missed, and the draft had grounding/citation issues.                           |
| `pd_006` | `start_semester`                                                   | Start semester was missed. Programme matching was also not confident.                                  |
| `pd_007` | `admission_requirements`                                           | Admission requirements were missed. Programme matching was not confident.                              |
| `pd_018` | `admission_requirements`, `grade_conversion`, `required_documents` | The system detected a generic application process topic instead of the expected specific topics.       |
| `pd_020` | `fees`                                                             | General fee information was missed because the programme choice was broad and not confidently matched. |
| `pd_021` | `credit_recognition`, `required_documents`                         | Transfer/credit recognition and required documents were not detected strongly enough.                  |

These failures are useful for the thesis evaluation because they show where the current rule-based topic detection and retrieval pipeline still need improvement.

## Thread-memory and follow-up test

Version label: `v5_7_thread_final_check`

The thread-memory test checks whether the system can handle follow-up emails using the same thread ID.

Expected behaviour:

| Follow-up situation        | Expected system behaviour                                  |
| -------------------------- | ---------------------------------------------------------- |
| New first email            | Generate a normal staff draft                              |
| Follow-up with a new topic | Use previous thread context and generate a draft           |
| Clarification or complaint | Flag for human review instead of generating a normal draft |

### Current thread-memory result

The follow-up classification now works as expected in the tested cases.

Examples:

| Case                                             | Expected                     | Result  |
| ------------------------------------------------ | ---------------------------- | ------- |
| First email in thread                            | `new_enquiry`                | Correct |
| Follow-up asking a new topic                     | `followup_new_topic`         | Correct |
| Follow-up saying the previous answer was unclear | `clarification_or_complaint` | Correct |

This supports the Human-in-the-Loop direction of the PoC. Follow-up emails can use thread context, but unclear or corrective follow-ups are flagged for staff review.

## Human-in-the-Loop

The system does not send emails automatically. Every generated response remains a draft. Staff must review and send the final email.

The disclaimer text is stored separately in:

```text
config/disclaimer.md
```

Current disclaimer:

```text
This draft was generated with AI support and must be reviewed by HTW Berlin staff before sending.
```

## Current model and API setup

The current PoC uses local/rule-based logic for:

* topic detection
* programme catalogue matching
* source ordering
* review checks
* disclaimer insertion
* CSV logging
* thread ID handling
* basic follow-up flagging

The current PoC still uses external APIs for model-heavy tasks such as:

* embeddings
* reranking
* draft generation

For production, these model-heavy parts should be moved to HTW-approved or locally controlled services where possible.

## Data and privacy note

The raw evaluation outputs are not committed because they contain generated drafts and test conversations. Only this summary file is committed.

The test cases use artificial demo emails and demo email addresses. No real student emails should be committed.

## Current conclusion

The current PoC is suitable for code review and thesis discussion.

The strongest contribution is the staff-facing multi-topic draft workflow. The thread-memory and follow-up flagging now work as supporting functionality.

The system is not production-ready yet. The main next improvements are:

* improve programme-specific source retrieval
* reduce extra-topic answering
* strengthen handling of pending transcript/final certificate cases
* improve vague programme matching
* improve transfer and credit recognition detection
* migrate external model-heavy tasks to HTW-approved or local services where possible
