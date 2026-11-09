import json
import uuid
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
ICON = str(Path(__file__).parent / "assets" / "icon.png")
st.set_page_config(page_title="Wildfire Smoke Assistant", page_icon=ICON, layout="wide")

# Captions are grey by default; make them black like the rest of the text.
st.markdown("<style>[data-testid='stCaptionContainer'], [data-testid='stCaptionContainer'] * "
            "{ color: #000000 !important; opacity: 1 !important; }</style>", unsafe_allow_html=True)

icon_col, title_col =st.columns([1, 12], vertical_alignment="center")
icon_col.image(ICON, width=70)
title_col.title("Wildfire Smoke Assistant")
st.caption("Ask how to prepare for wildfire smoke, stay safe while it is smoky, or protect yourself from ash. A supervisor "
           "agent hands each question to specialist agents (before, during, after) that search three EPA fact sheets.")
st.caption("General public guidance from US EPA fact sheets. Not medical advice. Follow local authorities for evacuation "
           "and safety orders.")

# ---------------------------------------------------------------------------
# Load the agents (multi_agent.py)
# ---------------------------------------------------------------------------
try:
    import multi_agent as ma
except ImportError as e:
    st.error(f"A required package is missing: {e}. Install it in the agentic_w11 environment, then restart the app.")
    st.stop()

try:
    # Load the fact sheets and build the agents if that has not happened yet. This is checked on every run, because
    # Streamlit reloads a module when its file changes, which would otherwise leave the sheets empty.
    if not ma.ready():
        with st.spinner("Loading the fact sheets and the search model…"):
            ma.initialize()
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.stop()

SAMPLE_QUESTIONS = [
    "Smoke is expected in my area tomorrow. What should I do today?",
    "I have central air conditioning. How do I keep the smoke out?",
    "We have no air conditioning and it is hot. Can I open the windows?",
    "My mother has COPD and it is smoky outside. What should we do?",
    "Do I need a mask outside? Will a dust mask work?",
    "We are allowed back to our house after the fire. What do we need?",
    "How should I clean up ash, and what do I do with it?",
    "What should I buy now to be ready for smoke season?",
]

# Each browser session gets its own thread, so the agent keeps one memory per conversation.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar: the sheets, sample questions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("The fact sheets")
    for key, (filename, title) in ma.settings.SHEETS.items():
        st.caption(f"**{title}** · {ma.tools.sheet_pages[key]} pages")
    st.caption(f"{ma.tools.chunk_count} searchable passages (EPA fact sheets)")

    review = st.checkbox("Have the reviewer check each answer",
                         help="Adds a reviewer agent that re-checks the answer against the fact sheets. Uses more model calls.")

    st.header("Sample questions")
    st.caption("Click one to send it.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, width="stretch"):
            st.session_state.queued = q

    st.caption(f"Model: {ma.settings.MODEL} on Groq")
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex   # a new thread: the agent forgets the old conversation
        st.rerun()
    st.caption(f"Conversation (thread) id: {st.session_state.thread_id}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def show_warnings(message):
    """Show citations that point to a sheet or page that does not exist."""
    warnings = message.get("warnings", [])
    if warnings:
        lines = "\n".join(f"- {w}" for w in warnings)
        st.warning("Some citations could not be matched to the fact sheets. "
                   f"Check them before relying on this answer:\n\n{lines}")


def matches_of(result):
    """The passages a search returned, read from the saved tool result (empty if there is none)."""
    try:
        return json.loads(result or "{}").get("matches", [])
    except ValueError:
        return []


def show_trace(message):
    """Show which agent did what for one answer."""
    trace = message.get("trace", [])
    if trace:
        with st.expander(f"How it got there: {len(trace)} tool call{'s' if len(trace) != 1 else ''}"):
            agent = None
            for n, call in enumerate(trace, 1):
                if call["tool"] == "task":
                    agent = call["arguments"].get("agent_name")
                    task_text = str(call["arguments"].get("description", ""))
                    st.markdown(f"**{n}. Supervisor → `{agent}` agent**")
                    st.caption(task_text[:300] + (" ..." if len(task_text) > 300 else ""))
                else:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
                    st.markdown(f"**{n}.** &nbsp;&nbsp; `{agent}` → `{call['tool']}({args})`")
                    if call["tool"] == "search_guidance":
                        for match in matches_of(call.get("result")):
                            st.caption(f"{match['sheet']}, p. {match['page']} (score {match['score']}): "
                                       f"{match['text'][:110]}...")
                        if not matches_of(call.get("result")):
                            st.caption("No passage matched.")
                    elif call["tool"] == "load_skill":
                        st.caption("Checklist loaded.")
            for report in message.get("reports", []):
                st.markdown(f"**What the {report['agent']} agent reported**")
                st.caption(report["text"])


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=ICON if msg["role"] == "assistant" else None):
        st.markdown(msg["content"].replace("$", r"\$"))
        show_warnings(msg)
        show_trace(msg)

# ---------------------------------------------------------------------------
# New question: typed in the box, or clicked from the sidebar
# ---------------------------------------------------------------------------
queued = st.session_state.pop("queued", None)
question = st.chat_input("Ask your question about wildfire smoke") or queued

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=ICON):
        try:
            with st.spinner("Working…"):
                # ask() gives each question its own shared tool-call budget and keeps the supervisor's memory.
                result = ma.ask(question, thread_id=st.session_state.thread_id, review=review, show=False)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            if "tool_use_failed" in str(e):
                st.caption("The model garbled a tool call several times in a row. Ask again, or change MODEL in settings.py.")
            else:
                st.caption("If this is a rate-limit (429) error, wait a minute and ask again.")
            st.stop()

        reply = {"role": "assistant", "content": result["answer"].replace("`", ""), "trace": result["trace"],
                 "reports": result["reports"], "warnings": result["warnings"]}
        st.markdown(reply["content"].replace("$", r"\$"))
        show_warnings(reply)
        show_trace(reply)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(reply)
