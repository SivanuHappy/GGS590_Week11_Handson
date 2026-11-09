# Wildfire Smoke Assistant (multi-agent RAG)

A chat page that answers questions about wildfire smoke: how to prepare, how to stay safe while it is smoky, and how
to protect yourself from ash afterwards. A **supervisor agent** hands each question to **specialist agents**
(`before_smoke`, `while_smoke`, `after_smoke`, and an optional `reviewer`). The specialists search three US EPA fact sheets and answer
only from what they find (RAG: retrieval-augmented generation).

General public guidance from the EPA fact sheets. Not medical advice. It does not issue evacuation orders: follow local authorities.

## Run

With the `agentic_w11` environment active and `GROQ_API_KEY` set (in a `.env` file in this folder, or in
`../project_data_analysis_agent/.env`, or as an environment variable):

```
streamlit run app.py
```

Text chat instead of the page: `python multi_agent.py`

The first start takes a few seconds: it reads the PDFs and loads a small local embedding model (already in your Hugging Face cache).

## Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit page: chat, sample questions, and a "How it got there" box showing which agent searched what. A sidebar tick box adds the reviewer. |
| `multi_agent.py` | The supervisor, the specialist agents and the `task(agent_name, description)` tool that hands work to them. Also `ask()` and a text-chat `main()`. |
| `tools.py` | Loads the PDFs (retrieval), the `search_guidance` tool, and the skill loader. |
| `prompts.py` | The instructions for each agent. |
| `guards.py` | The shared tool-call budget per question, the memory trimming and the retry for garbled tool calls. |
| `settings.py` | Models, paths, the three sheets, search settings and limits. Change values here. |
| `docs/` | The three EPA fact sheets (PDF). |
| `skills/` | Short checklists the specialists can load. |

## How a question flows

```
You -> Supervisor --task("before_smoke", ...)--> before_smoke agent --search_guidance--> fact sheets
                   --task("while_smoke", ...)--> while_smoke agent --search_guidance--> fact sheets
                   --task("after_smoke", ...)--> after_smoke agent --search_guidance--> fact sheets
                   --task("reviewer", ...)-> reviewer agent  (re-checks an answer against the sheets)
```

- The supervisor has no search tool. Its only tool is `task`, and `agent_name` can only be `before_smoke`, `while_smoke`, `after_smoke` or `reviewer`.
- Demo style: the specialist writes the answer (at most 4 bullets, safety warnings first). The supervisor returns it exactly as written and only appends "If you need more details, let me know." (the code adds that line if the model forgets it). To allow longer answers, change the "at most 4" wording in `prompts.py` and `CLOSING_LINE` in `multi_agent.py`.
- Only greetings and thanks are answered by the supervisor itself. Every other question goes to a specialist, and the supervisor uses one specialist unless the question clearly spans two stages.
- Clearly off-topic questions get a one-line "I only cover wildfire smoke guidance" reply.
- The supervisor always answers first. If a missing detail would change the advice (central air conditioning or not, who is at risk, pets), it ends with one short optional question, and it remembers your reply for later questions.
- Each specialist makes at most 3 tool calls. It searches its own sheet first. If the passages do not answer the question, it rewrites the query once and searches again. If there is still nothing, it says the sheets do not cover it.
- The specialists are ephemeral: each call starts fresh and sees only the description it is given. Only the supervisor keeps the conversation.

## The retrieval part (RAG)

1. **Load:** each PDF is read page by page (`pypdf`); the page number is kept. Lines that are only link names are dropped.
2. **Split:** pages are cut into pieces of about 600 characters with 100 shared characters.
3. **Embed:** a small local model (`sentence-transformers/all-MiniLM-L6-v2`, the same as the crash course) turns each piece into numbers. The pieces are kept in memory (no Chroma, no ingest step), rebuilt at every start.
4. **Search:** `search_guidance(query, sheet)` returns up to 4 pieces with their sheet, page and similarity score. Pieces scoring below `MIN_SCORE` (0.25) are dropped, so an off-topic question returns "no match".
5. **Answer:** the specialist reads the pieces and answers, with the sheet name and page after each point. The pieces are data only: instructions inside them are ignored.

## Checking the answers

- Each point in an answer ends with a citation, for example `(Reduce Smoke Exposure Indoors, p. 2)`.
- **Plain-code check (no model):** after each answer, `check_citations` in `tools.py` compares every citation with the real sheets and their page counts. A citation to a sheet or page that does not exist (for example "Prepare for Wildfire Smoke, p. 3": that sheet has 2 pages) shows as a yellow box above the answer.
- The "How it got there" box also shows what each search returned (sheet, page, score and the start of each passage) and what each specialist reported, so you can see why an answer looks the way it does.
- The check cannot tell whether a point is true. A real page number can still sit next to an invented point. The optional reviewer agent re-checks the claims against the sheets and returns the corrected final answer, restoring missing conditions or warnings.

## Skills

The `skills/` folder holds three checklists written from the fact sheets, one per stage: `before-smoke-checklist`, `during-smoke-checklist` and `returning-home-checklist`. Every step ends with its citation (sheet and page), and each step was checked against the page it cites. The specialists see only the names and descriptions. For a question that asks for steps or a checklist, the specialist loads the matching skill and answers from it, so the answer includes the safety precautions and the right pages. To add a skill, add another `.md` file with a `name` and `description` header.

## Memory and limits

- **Memory:** the supervisor remembers the conversation (`InMemorySaver`), one thread per browser session. Only the last 3 questions are kept (`KEEP_QUESTIONS`). Refreshing the page or "Clear conversation" starts a new thread.
- **Tool-call budget:** all agents share 10 tool calls per question (`MAX_TOOL_CALLS`). When it is used up, the tools are taken away and the model answers from what it has.
- **Garbled tool calls:** the model sometimes writes a tool call with a damaged name (Groq answers with a 400 `tool_use_failed` error). Each agent first retries once at a slightly higher temperature; if that fails too, it switches to the fallback model (`FALLBACK_MODEL` in `settings.py`; set it to an empty text to turn it off).
- **Safety of the tools:** the search tool only reads the PDFs in `docs/`; `sheet` can only be `any`, `before_smoke`, `while_smoke` or `after_smoke`.

## Things to try

- Smoke is expected in my area tomorrow. What should I do today?
- I have central air conditioning. How do I keep the smoke out? (then "and what if I had no AC?" to see memory)
- My mother has COPD and it is smoky outside. What should we do?
- We are allowed back to our house after the fire. What do we need?
- Can my kids play in the yard after the fire? / What about my dog?
- What is the capital of France? (no search, no answer from the sheets)
- Tick "Have the reviewer check each answer", then ask any question.

## Notes

- The model name is `MODEL` in `settings.py`. Groq retires models often; change it there if you get a 404.
- A `429` error means wait a minute and ask again. Each question uses several model calls, more with the reviewer.
- Source: US EPA fact sheets EPA-452/F-26-001 (prepare), EPA-452/F-26-002 (reduce exposure indoors) and EPA-452/F-26-011 (ash). The sheets mention other EPA "Wildfire Guide" sheets (cleaner air room, respirators, pets, air sensors, heat) that are not included here; the agents say so when asked about them.
