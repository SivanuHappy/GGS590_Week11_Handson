"""Guards that run around the agents: a shared tool-call budget per question, memory trimming, and a retry."""
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock

from langchain.agents.middleware import before_model, wrap_model_call, wrap_tool_call
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

import settings


def _is_garbled_tool_call(error: Exception) -> bool:
    """True for Groq's 400 'tool_use_failed' error, where the model wrote a tool call with a damaged name."""
    return "tool_use_failed" in str(error)


# The agents run at temperature 0, so an identical retry would repeat the same broken output.
# The retry therefore raises the temperature a little, to make the model write the call differently.
RETRY_TEMPERATURES = [0.5]      # one cheap retry; if it fails too, the fallback model takes over


@wrap_model_call
def retry_garbled_tool_call(request, handler):
    """Retry a model call that failed because the model garbled its tool call; other errors pass through."""
    try:
        return handler(request)
    except Exception as error:
        if not _is_garbled_tool_call(error):
            raise
        last_error = error
    for temperature in RETRY_TEMPERATURES:
        time.sleep(0.5)
        try:
            settings_now = {**(request.model_settings or {}), "temperature": temperature}
            return handler(request.override(model_settings=settings_now))
        except Exception as error:
            if not _is_garbled_tool_call(error):
                raise
            last_error = error
    raise last_error


@dataclass
class RequestContext:
    """What one question collects while the agents work: tool calls used, a trace of them, and each agent's report."""
    calls: int = 0
    trace: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)


# The context of the question being answered; ask() sets it, so every agent and tool can reach it.
REQUEST = ContextVar("wildfire_request", default=None)


@wrap_tool_call
def limit_tool_calls(request, handler):
    """Count every tool execution across all agents; once the budget is used, refuse further tools."""
    context = REQUEST.get()
    if context is None:
        raise RuntimeError("Call ask() so each question receives a fresh tool budget.")
    with context.lock:
        allowed = context.calls < settings.MAX_TOOL_CALLS
        if allowed:
            context.calls += 1
            entry = {"tool": request.tool_call["name"], "arguments": request.tool_call["args"]}
            context.trace.append(entry)
    if not allowed:
        return ToolMessage(content="No further tools are available for this request. Finish using existing results; "
                                   "mention any unfinished work briefly.",
                           tool_call_id=request.tool_call["id"])
    response = handler(request)
    entry["result"] = str(getattr(response, "content", ""))[:4000]       # what the tool returned, for the trace
    return response


@wrap_model_call
def finish_when_budget_used(request, handler):
    """When the budget is used up, take the tools away so the model writes its answer from what it has."""
    context = REQUEST.get()
    if context is not None and context.calls >= settings.MAX_TOOL_CALLS:
        original = request.system_message.content if request.system_message else ""
        message = SystemMessage(content=str(original) + "\nFinish now using available tool results. Do not request "
                                "further tools or invent missing information. Briefly identify unfinished work, if any.")
        return handler(request.override(tools=[], tool_choice=None, system_message=message))
    return handler(request)


@before_model
def trim_history(state, runtime):
    """Keep complete exchanges for the latest few questions only, so a long chat stays inside the token limit."""
    messages = state["messages"]
    positions = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if len(positions) <= settings.KEEP_QUESTIONS:
        return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[positions[-settings.KEEP_QUESTIONS]:]]}
