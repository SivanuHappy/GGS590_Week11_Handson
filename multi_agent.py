"""
The multi-agent wildfire smoke assistant: a supervisor that hands work to specialist agents.

    You -> Supervisor --task("before_smoke", ...)--> before_smoke agent (before smoke arrives)
                       --task("while_smoke", ...)--> while_smoke agent (while it is smoky)
                       --task("after_smoke", ...)--> after_smoke agent (after the fire)
                       --task("reviewer", ...)-> reviewer agent  (checks an answer against the sheets)

The supervisor has no search tool. Its only tool is task(agent_name, description), which starts one specialist and
returns that specialist's final message. Each specialist searches the EPA fact sheets with search_guidance (RAG):
the pieces it finds are added to what the model reads before it answers.
The specialists are ephemeral: each call starts fresh and sees only the description it is given.
Only the supervisor keeps the conversation (memory).

Files:  settings.py (limits, models)  tools.py (fact sheets, search)  prompts.py (instructions)  guards.py (budget, memory)

Run as a text chat:   python multi_agent.py
Use from a page:      import multi_agent as ma;  ma.ask("Smoke is coming tomorrow, what should I do?", thread_id="x")
"""
import os
from typing import Literal
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

import guards
import prompts
import settings
import tools

supervisor = None
SUBAGENTS = {}
CLOSING_LINE = "If you need more details, let me know."


def message_text(message):
    """Read text from either a plain message or a list of content blocks."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(block.get("text", "") for block in message.content if isinstance(block, dict))


@tool
def task(agent_name: Literal["before_smoke", "while_smoke", "after_smoke", "reviewer"], description: str) -> str:
    """
    Hand a task to a specialist agent and return its final message. The agent cannot see the chat history,
    so the description must contain everything it needs.
    Agents:
    - before_smoke: what to do before smoke arrives (air quality checks, air cleaners, filters, cleaner air room, supplies, pets).
    - while_smoke: what to do while it is smoky (staying indoors, filters, cooking, heat, going outside, masks, vehicles).
    - after_smoke: what to do after the fire (returning home, ash, cleanup, respirators, children, pets, food and water).
    - reviewer: re-checks the claims of a draft answer against the fact sheets.
    """
    result = SUBAGENTS[agent_name].invoke(
        {"messages": [{"role": "user", "content": description}]},
        config={"recursion_limit": 100, "max_concurrency": 1},
    )
    text = message_text(result["messages"][-1])
    context = guards.REQUEST.get()
    if context is not None:
        with context.lock:
            context.reports.append({"agent": agent_name, "text": text})
    return text


def initialize(model=None, fallback_model=None):
    """Load the fact sheets, build the specialist agents and the supervisor. Optional models are for testing."""
    global supervisor
    if model is None:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Set GROQ_API_KEY in your .env file or environment before starting.")
        model = ChatGroq(model=settings.MODEL, temperature=0, timeout=60, max_retries=2)
        if settings.FALLBACK_MODEL:
            fallback_model = ChatGroq(model=settings.FALLBACK_MODEL, temperature=0, timeout=60, max_retries=2)
    tools.load_guides()

    guard = [guards.retry_garbled_tool_call, guards.limit_tool_calls, guards.finish_when_budget_used]
    if fallback_model is not None:
        guard = [ModelFallbackMiddleware(fallback_model), *guard]      # outermost: used only after the retry also fails
    skills = "\nAvailable skills:\n" + ("\n".join(f"- {n}: {s['description']}" for n, s in tools.SKILLS.items()) or "None.")
    for key in prompts.SPECIALISTS:
        SUBAGENTS[key] = create_agent(model=model, tools=tools.SPECIALIST_TOOLS, middleware=guard,
                                      system_prompt=prompts.specialist_prompt(key) + skills, name=f"{key}_agent")
    SUBAGENTS["reviewer"] = create_agent(model=model, tools=tools.REVIEWER_TOOLS, middleware=guard,
                                         system_prompt=prompts.REVIEWER_PROMPT, name="reviewer_agent")
    supervisor = create_agent(model=model, tools=[task], middleware=[*guard, guards.trim_history],
                               system_prompt=prompts.SUPERVISOR_PROMPT,
                               checkpointer=InMemorySaver(), name="supervisor")
    return supervisor


def ready() -> bool:
    """True when the fact sheets are loaded and the agents are built (false at the start, or after a reload)."""
    return supervisor is not None and tools.store is not None


def ask(question: str, thread_id: str = "class-demo", review: bool = False, show: bool = True) -> dict:
    """Run one question through the supervisor; returns the answer, tool-call count, trace, agent reports and warnings.

    review=True asks the supervisor to have the reviewer check the answer (uses more calls).
    Use the same thread_id for follow-ups.
    """
    if supervisor is None:
        initialize()
    if review:
        question += "\n(Review requested: have the reviewer check the answer before you reply.)"
    context = guards.RequestContext()
    token = guards.REQUEST.set(context)
    try:
        result = supervisor.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}, "max_concurrency": 1, "recursion_limit": 100},
        )
        answer = message_text(result["messages"][-1])
        if context.reports and "let me know" not in answer.lower():      # a specialist was used (not a greeting)
            answer += "\n\n" + CLOSING_LINE        # demo style: short answers, then an offer of more detail
        # Plain-code check of the citations against the real PDFs (no model involved).
        warnings = tools.check_citations(answer)
        for report in context.reports:
            warnings += [f"{report['agent']} agent: {w}" for w in tools.check_citations(report["text"])]
        output = {"answer": answer, "tool_calls": context.calls, "trace": list(context.trace),
                  "reports": list(context.reports), "warnings": list(dict.fromkeys(warnings))}
    finally:
        guards.REQUEST.reset(token)
    if show:
        print(output["answer"])
        for warning in output["warnings"]:
            print(f"[check] {warning}")
    return output


def main():
    """Start a local text chat; type exit to stop."""
    initialize()
    thread_id = uuid4().hex
    print("Wildfire smoke assistant ready (before_smoke, while_smoke, after_smoke, reviewer). Type exit to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(question, thread_id=thread_id)


if __name__ == "__main__":
    main()
