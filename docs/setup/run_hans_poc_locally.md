\# Running the HANS Staff-Support PoC Locally



This guide explains how to run the HANS staff-support email assistant PoC on a local Windows machine.



The current version is a Mistral-integrated test version. It keeps the existing RAG setup mostly unchanged and changes the draft generation provider to Mistral.



\## 1. What this PoC does



HANS supports staff by preparing draft email responses for student enquiries.



The system can:



\- detect multiple topics in one student email;

\- retrieve relevant HTW source material;

\- prepare one staff-facing draft response;

\- show source links for staff verification;

\- add a configurable disclaimer;

\- detect follow-up emails using thread memory;

\- flag unclear or corrective follow-ups for human review.



The system does not send emails automatically. Every draft must be checked and sent by staff.



\## 2. Current technical setup



Current v6 test setup:



\- Generation provider: Mistral

\- Embeddings: Cohere

\- Reranking: Cohere

\- Vector store: Qdrant

\- Topic detection: local rule-based logic

\- Programme matching: local programme catalogue

\- Follow-up detection: local rule-based logic and local thread memory

\- Disclaimer: loaded from `config/disclaimer.md`



Only the generation provider was changed first. Embeddings, reranking and retrieval were kept unchanged so that the Mistral result can be compared with the previous v5.7 baseline.



\## 3. Files not included in GitHub



The following files are intentionally not committed:



\- `.env`

\- API keys

\- `.venv`

\- generated CSV/JSONL evaluation outputs

\- local thread-memory files

\- raw temporary files



The `.env.example` file shows which environment variables are needed.



\## 4. Required tools



Install the following:



\- Python 3.12 recommended

\- Git

\- Docker Desktop, if Qdrant/Postgres are run locally

\- A valid Mistral API key

\- A valid Cohere API key

\- Qdrant connection details, if using Qdrant Cloud



Python 3.12 is recommended because some dependencies may not install smoothly on Python 3.13.



\## 5. Clone the repository



```powershell

git clone https://github.com/Priyanka1107/HANS-proofOfConcept.git

cd HANS-proofOfConcept




\# If testing the Mistral branch:



git checkout v6-mistral-generation



\## 6. Create and activate virtual environment

py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



Upgrade pip:



python -m pip install --upgrade pip setuptools wheel



Install requirements:



python -m pip install -r requirements.txt



Check installed packages:



python -m pip check



\## 7. Create local .env



Copy the example file:



copy .env.example .env



Open .env:



notepad .env



Add the required keys and settings.



Example:



GENERATION\_PROVIDER=mistral

GENERATION\_MODEL=mistral-small-latest

MISTRAL\_API\_KEY=your\_mistral\_key\_here



EMBEDDING\_PROVIDER=cohere

RERANK\_PROVIDER=cohere

COHERE\_API\_KEY=your\_cohere\_key\_here



QDRANT\_URL=your\_qdrant\_url\_here

QDRANT\_API\_KEY=your\_qdrant\_key\_here

QDRANT\_COLLECTION=your\_collection\_name\_here



Do not commit .env.



\##8. Test the Mistral key



Run:



python scripts\\test\_mistral\_key.py



Expected result:



Mistral key works.



\## 9. Start the backend



Open PowerShell window 1:



.\\.venv\\Scripts\\Activate.ps1

python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload



The backend should run at:



http://127.0.0.1:8001



\## 10. Start the Streamlit UI



Open PowerShell window 2:



.\\.venv\\Scripts\\Activate.ps1

python -m streamlit run streamlit\_app.py



Open the Streamlit URL shown in the terminal.





\## 11. Run the multi-topic test script



Make sure the backend is running first.



Then run:



python scripts\\test\_multitopic\_flow.py --version v6\_0\_mistral\_generation\_check --comment "Mistral generation provider integrated; Cohere embeddings, Cohere reranking, and Qdrant retrieval unchanged"



The script writes local result files to the evaluation folder.



Generated CSV and JSONL files are local outputs and are not committed.



\## 12. Run the thread-memory test script



Clear local thread memory first:



Remove-Item evaluation\\email\_thread\_memory.json -ErrorAction SilentlyContinue



Run:



python scripts\\test\_email\_thread\_flow.py --version v6\_0\_mistral\_thread\_check --comment "Mistral generation provider with local thread-memory and follow-up routing"



\## 13. Main output files



Generated locally:



evaluation/email\_assistant\_v5\_results.csv

evaluation/email\_assistant\_v5\_results.jsonl

evaluation/email\_thread\_v5\_results.csv

evaluation/email\_thread\_v5\_results.jsonl



These are evaluation outputs and should not be committed.



\## 14. Known limitations



The current Mistral version is a working test version, not a final production version.



Known observations:



Mistral is faster than the previous Anthropic generation setup.

Mistral often produces shorter drafts.

Some answers may need stronger post-processing guards.

In some application-route cases, Mistral may follow retrieved uni-assist evidence too strongly and may not fully respect the interpreted citizenship profile.

Programme-specific retrieval still needs improvement for some cases.

Transfer and credit-recognition cases need stronger topic detection and better source retrieval.

Every draft must be reviewed by staff before sending.



\## 15. Human-in-the-Loop



The PoC is staff-facing.



HANS prepares a draft only. It does not send the email automatically.



The final decision and final email remain with HTW staff.



