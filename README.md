# Wildfire Smoke Assistant

A multi-agent, retrieval-augmented generation (RAG) application for exploring wildfire smoke guidance from three U.S. EPA fact sheets. A supervisor delegates questions to specialist agents for preparation, indoor smoke exposure, and ash precautions. An optional reviewer checks the draft against the documents.

The assistant provides general information from the supplied documents. It does not provide live air-quality conditions, individualized medical advice, or evacuation orders. Follow local authorities for emergency instructions.

## 1. Set up the application

### Before you begin

- Download and extract the project folder.
- Install Conda or Python, depending on the setup option you choose below.
- Open a terminal in `project_wildfire_smoke_agent`: use **Command Prompt on Windows** or **Terminal on macOS/Linux**.
- Confirm that `environment.yml`, `requirements.txt`, and the three EPA PDFs in `docs/` are included.
- Allow time for package downloads. The dependencies include PyTorch, which can be a large download.

The existing application environment was tested with **Python 3.14**. The supplied dependency files record its package versions; a fresh installation from these files and other Python versions have not yet been tested. A Python virtual environment uses the Python version that creates it.

### Step 1: Create a Groq API key

1. Open the [Groq Console](https://console.groq.com/home) and sign in or create an account.
2. Select **API Keys → Create API Key**.
3. Enter a recognizable name and choose an expiration option.
4. Create the key, then **copy and save it immediately**. Groq will not display it again.

Groq hosts the language model used by the agents, so you do not need to run a large language model on your computer.

[View the API key setup guide with screenshots](https://drive.google.com/file/d/1rjQqgEhFqU1IiovSGNq7Bc9fPElXUn38/view?usp=sharing).

### Step 2: Create an environment

Choose **one** option.

#### Option A: Create a Conda environment

Run these commands in a terminal where Conda is available, such as Anaconda Prompt on Windows:

```bash
conda env create -f environment.yml
conda activate agentic_w11
```

The environment file installs the required packages. If `agentic_w11` is already prepared, activate it instead of creating it again.

#### Option B: Create a Python virtual environment (`venv`)

**Windows — Command Prompt**

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**macOS/Linux — Terminal**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Step 3: Add your API key

Create a plain-text file named `.env` in the project folder, alongside `app.py`, and enter:

```text
GROQ_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your copied key.

- **VS Code (Windows or Mac):** Create a new file and save it as `.env`.
- **Notepad (Windows):** Select **Save as type → All Files** and use `.env` as the filename.
- **TextEdit (Mac):** Select **Format → Make Plain Text**, then save as `.env`. Allow the filename to begin with a period if prompted.

Make sure the file is named `.env`, not `.env.txt`. Keep it private and do not upload it with your coursework or commit it to a public repository.

You can also provide `GROQ_API_KEY` as an environment variable. For this exercise, using a `.env` file in this project folder is the simplest option.

### Step 4: Run the application

With your environment activated, run:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal if the browser does not open automatically. Keep the terminal running while using the application; press **Ctrl+C** to stop it.

On the first run, the application downloads the local embedding model if it is not already cached. It then reads the PDFs and builds an in-memory vector store. Startup time depends on your connection and computer.

For a terminal-based chat instead of the browser interface:

```bash
python multi_agent.py
```

## 2. Use the assistant

1. Enter a question or choose a sample question.
2. Read the answer and its fact-sheet/page citations.
3. Expand **“How it got there”** to inspect tool calls, retrieved passages, and agent reports.
4. Open a cited PDF page to check whether it supports the recommendation.
5. Ask a follow-up question to explore conversation memory.
6. Enable the reviewer to add a document-based review of the draft.

### Questions to try

- “Smoke is expected in my area tomorrow. What should I do today?”
- “I have central air conditioning. How can I reduce smoke indoors?” Then ask: “What if I have no air conditioning?”
- “We have been allowed to return home after a fire. What ash precautions should we take?”
- “What precautions should I take for my dog during smoky conditions?”
- “What is the air quality outside my house right now?” Observe how the assistant handles information the PDFs cannot supply.

To compare reviewer settings, ask the same question with the reviewer disabled and enabled, starting a fresh conversation for each run. Answers may also vary between runs.

## 3. Understand the agents

| Agent | Role |
|---|---|
| `supervisor` | Interprets the question, delegates tasks, and maintains conversation memory. |
| `before_smoke` | Handles preparation before wildfire smoke arrives. |
| `while_smoke` | Handles reducing indoor smoke exposure. |
| `after_smoke` | Handles ash precautions and cleanup guidance. |
| `reviewer` | Checks a draft against the documents when review is requested. |

The supervisor's only tool is `task(agent_name, description)`. It passes a task description to the selected subagent and receives its answer. Specialists can search documents with `search_guidance` and load relevant checklists with `load_skill`.

Each subagent invocation starts fresh and receives only its task description. The supervisor maintains the conversation and can include relevant earlier details when assigning work.

### Response instructions

The prompts guide the agents to:

- Route substantive wildfire questions to the appropriate specialist, using more than one when needed.
- Write concise answers, normally no more than four bullet points, with important precautions first.
- Cite the fact sheet and page for document-based recommendations.
- State when the supplied documents do not cover the question.
- Ask a short follow-up question when a missing detail would affect the guidance.

The specialist instructions limit their tool use and encourage a revised search when the first search is insufficient. These are model instructions; they should not be confused with the shared tool-call limit enforced by code.

## 4. How document retrieval works

1. **Read the PDFs.** Pages are extracted with `pypdf`, preserving source names and page numbers. Lines containing only link names are removed.
2. **Split the text.** Pages are divided into approximately 600-character chunks with 100-character overlap.
3. **Create embeddings.** The local `sentence-transformers/all-MiniLM-L6-v2` model converts chunks into numerical representations. The chunks, embeddings, and source metadata are stored in memory.
4. **Search.** The same model embeds the query. `search_guidance(query, sheet)` returns up to four matching passages, including their sheet, page, and similarity score. Results below `MIN_SCORE` (0.25) are removed. A score threshold does not guarantee relevance.
5. **Generate an answer.** The specialist's language model uses the retrieved passages to write the answer. The retriever returns passages, not a draft answer.

The vector store is rebuilt when the application starts. The embedding model runs locally and needs no Groq API key. Agent responses use the Groq-hosted language model.

## 5. Skills and checklists

The `skills/` folder contains Markdown checklists for preparation, indoor smoke exposure, and ash precautions. Each checklist includes source references to the EPA fact sheets.

Agents initially see the skill names and descriptions. They can call `load_skill` to retrieve the full instructions when a checklist is relevant. Loading a skill is a tool action; the checklist itself provides guidance for completing a task.

To add a skill, create a Markdown file with a `name` and `description` header and the checklist content. Use the names in the supplied skill files when referring to them.

## 6. Citations, review, and limits

### Citation checks

Answers are instructed to include references such as `(Reduce Smoke Exposure Indoors, p. 2)`. The `check_citations` function in `tools.py` checks whether cited sheets and pages exist. Invalid references produce a warning in the interface.

**An existing page number does not prove that a claim is supported.** The citation check does not verify the meaning of the source text. Compare important claims with the cited passage yourself.

The optional reviewer searches the documents to check and revise a draft. It may improve an answer, but it can also miss errors or introduce new ones.

### Conversation memory

The supervisor uses `InMemorySaver` to retain conversation state, with a thread for each browser session. `KEEP_QUESTIONS` retains the last three questions and their associated conversation messages. **Clear conversation** starts a new thread. Memory is temporary and does not survive a program restart.

### Execution controls

- **Shared budget:** All agents share a limit of 10 tool calls per question (`MAX_TOOL_CALLS`). When the budget is exhausted, further tool calls are blocked and the model is instructed to finish using the available results.
- **Allowed inputs:** Agent and sheet names are restricted to predefined values. The search tool reads the supplied PDFs in `docs/`.
- **Tool-call recovery:** A malformed Groq tool call can trigger a retry. The configured fallback model provides another recovery option. See `guards.py` and `settings.py` for implementation details.
- **Retrieved content:** Prompts instruct agents to treat retrieved passages as evidence, not as commands.

The execution trace shows tool activity and outputs, not the model's private internal reasoning. None of these controls guarantees answer correctness.

## 7. Project files

| File or folder | Purpose |
|---|---|
| `environment.yml` | Conda environment definition and package versions. |
| `requirements.txt` | Python package versions for installation with pip. |
| `.env` | Your private Groq API key; create this file yourself. |
| `app.py` | Streamlit interface, sample questions, reviewer option, and execution trace. |
| `multi_agent.py` | Supervisor, subagents, delegation tool, `ask()`, and terminal chat. |
| `tools.py` | PDF retrieval, document search, skill loading, and citation checks. |
| `prompts.py` | Agent roles and response instructions. |
| `guards.py` | Tool-call budget, memory trimming, and tool-call recovery. |
| `settings.py` | Model names, document paths, retrieval settings, and limits. |
| `docs/` | Three EPA fact sheets. |
| `skills/` | Task checklists available to the specialists. |

## 8. Source documents

| EPA fact sheet | Pages | Publication identifier |
|---|---:|---|
| Prepare for Wildfire Smoke | 2 | EPA-452/F-26-001 |
| Reduce Smoke Exposure Indoors | 3 | EPA-452/F-26-002 |
| Protect Yourself from Ash | 3 | EPA-452/F-26-011 |

These eight pages form the document knowledge base. Websites and additional fact sheets linked inside the PDFs are not automatically included.
