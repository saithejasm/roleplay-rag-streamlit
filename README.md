# Roleplay RAG

A Streamlit app for practicing structured conversations through AI roleplay, grounded
(optionally) in your own documents via RAG. Built as a college project.

The AI always plays the authority figure; you always play the other role.

| Scenario | AI plays | You play | Optional document |
|---|---|---|---|
| 💼 Job Interview | Interviewer | Candidate | Your resume |
| 🛂 Visa Interview | Visa Officer | Applicant | Application/supporting doc |
| 🩺 Doctor & Patient | Doctor | Patient | Case notes / medical history |
| 🎓 Teacher & Student / Viva Voce | Examiner | Student | Syllabus / notes |
| 🏫 College Admission Interview | Admissions Officer | Applicant | Personal statement / resume |
| 🚓 Police Traffic Stop | Police Officer | Driver | Registration/insurance notes |
| 🏠 Landlord & Tenant | Landlord | Prospective Tenant | Rental application / references |
| 📈 Performance Review | Manager | Employee | Self-assessment / work summary |
| 🏦 Bank Loan Interview | Loan Officer | Applicant | Loan application / financial summary |

Documents can be uploaded as PDF or TXT, or typed/pasted directly in the sidebar.

When you upload a document, the AI retrieves relevant passages from it each turn and
asks questions grounded in your actual content (e.g. a specific project on your resume,
a topic in your syllabus). Without a document, it improvises a realistic generic
conversation for the scenario.

At any point, click **"End session & get feedback"** to get an AI-generated evaluation
of *your* performance — strengths, areas to improve, a rough rating, and tips — based on
a scenario-specific rubric.

## Setup

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py
```

Then pick an LLM source in the sidebar's **Settings**:

- **Local server** (free, private): run one of
  - [Ollama](https://ollama.com): `brew install ollama && ollama serve && ollama pull llama3.2`
  - LM Studio, llama.cpp `server`, or vLLM — any OpenAI-compatible server exposing `/v1/models`

  then pick the server type, confirm/edit the URL, and click "Refresh models".
- **Cloud provider** (OpenAI, Anthropic, Google Gemini, xAI Grok, Groq, MiniMax): pick a
  provider, paste an API key, and set/confirm the model name. The key is kept only in
  memory for the session and is never written to disk.

Pick a scenario from the sidebar, choose a **location/setting** and **mood/tone**
(Relaxed, Normal, Tense, or Strict — pick "Custom..." on location to type your own),
optionally upload a document (or paste details manually) for that scenario, and start
chatting in the main chat box.

## How grounding works

1. Uploaded PDFs are split page-by-page, then chunked with a paragraph-aware sliding
   window (chunk never crosses a page boundary), embedded locally with
   `sentence-transformers` (`all-MiniLM-L6-v2`), and stored in an in-memory ChromaDB
   collection scoped to the current browser session + scenario.
2. Before every AI turn, the app retrieves the top-matching chunks for that turn's
   query — the scenario's fixed seed phrase for the very first turn (e.g. "skills
   experience projects" for a resume), and your last message on every turn after that,
   so follow-up questions probe deeper into what you actually said.
3. Retrieved passages are folded into the system prompt as background context the AI
   "knows" about you — it doesn't cite documents mid-roleplay, it just asks about
   specifics naturally.
4. Switching to a different document, or uploading one mid-conversation, takes effect
   immediately on the very next AI turn.

No LangChain — the whole pipeline is plain Python, using `litellm` to unify local and
cloud model calls behind a single `completion()` call.

## Persistence

Everything — chat history, uploaded documents, feedback — lives in memory only, scoped
to your browser session and cleared entirely when the app restarts. Nothing is written
to disk. Each scenario keeps its own independent state, so you can switch between
scenarios and come back to where you left off without losing progress, for as long as
the app keeps running.

## Known limitations

- Smaller local models may not perfectly hold to "one question at a time" or fully stay
  in character — larger/cloud models are more reliable roleplayers.
- Feedback quality depends on how much you've said; very short conversations produce
  thin evaluations.
- The Doctor & Patient scenario is a communication-practice simulation, not real medical
  advice.
- The "Local server" URL field lets the app's backend make outbound HTTP requests to a
  user-supplied URL — fine for personal/local use, but this app is not hardened for
  public multi-tenant deployment.
