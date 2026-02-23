# agent/graph.py
from __future__ import annotations

import json
from typing import TypedDict, List, Literal, Optional, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_ollama import ChatOllama

from agent.tools import TOOLS


# -----------------------------
# 1) Graph State (TypedDict)
# -----------------------------
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # chat history / reasoning trace


# -----------------------------
# 2) LLM (Ollama)
# -----------------------------
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.2,
).bind_tools(TOOLS)


SYSTEM_PROMPT = """You are Data Insight, an autonomous data analysis assistant.

Available tools:
- search_eda_kb: retrieve grounded EDA guidance from the knowledge base
- create_eda_plan: generate a step-by-step EDA plan from dataset columns + user goal

You may either:
1) call a tool when needed, OR
2) provide a final answer if you have enough information.

If calling a tool, call it with correct arguments.
"""


# -----------------------------
# Helper: detect JSON tool call text
# -----------------------------
def _parse_json_tool_call(text: str) -> Optional[dict]:
    """
    Some Ollama models output tool calls as plain JSON text like:
    {"name":"search_eda_kb","arguments":{"query":"missing_values","top_k":3}}
    This function detects and parses that.
    """
    if not text:
        return None
    t = text.strip()
    try:
        data = json.loads(t)
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            if isinstance(data["name"], str) and isinstance(data["arguments"], dict):
                return data
    except Exception:
        return None
    return None


# -----------------------------
# 3) Agent Node (LLM thinking)
# -----------------------------
def agent_node(state: GraphState) -> GraphState:
    messages = state["messages"]

    # Ensure system prompt is present once
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


# -----------------------------
# 4) Tool Node (exec tool calls)
# -----------------------------
tool_node = ToolNode(TOOLS)


# -----------------------------
# 5) JSON -> ToolCall converter node
# -----------------------------
def json_toolcall_prep_node(state: GraphState) -> GraphState:
    """
    Converts JSON-in-text tool request into a proper AIMessage with tool_calls
    so ToolNode can execute it.
    """
    last = state["messages"][-1]

    data = _parse_json_tool_call(getattr(last, "content", "") or "")
    if not data:
        # If parsing failed, just keep state as-is (router should not send us here in that case)
        return state

    tool_call_msg = AIMessage(
        content=last.content,  # Keep the original string content
        tool_calls=[
            {
                "name": data["name"],
                "args": data.get("arguments", {}),
                "id": "json_tool_call_1",
            }
        ],
        id=last.id if hasattr(last, "id") else None
    )

    # Return only the new message object, add_messages will replace the old one based on ID
    return {"messages": [tool_call_msg]}


# -----------------------------
# 6) Router (conditional edge)
# -----------------------------
def router(state: GraphState) -> Literal["tools", "tools_json_prep", "__end__"]:
    """
    If the last LLM message contains tool calls -> go to tools.
    If the last message is JSON tool call text -> go to tools_json_prep.
    Else -> END (final answer).
    """
    last = state["messages"][-1]

    # Case A: Native tool calls (best case)
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"

    # Case B: Tool call was printed as JSON text (common in Ollama)
    json_call = _parse_json_tool_call(getattr(last, "content", "") or "")
    if json_call:
        return "tools_json_prep"

    return END


# -----------------------------
# 7) Build Graph
# -----------------------------
def build_graph():
    g = StateGraph(GraphState)

    g.add_node("agent", agent_node)
    g.add_node("tools_json_prep", json_toolcall_prep_node)
    g.add_node("tools", tool_node)

    g.set_entry_point("agent")

    g.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "tools_json_prep": "tools_json_prep",
            END: END,
        },
    )

    # If JSON toolcall, convert -> tools
    g.add_edge("tools_json_prep", "tools")

    # After tools run, go back to agent (loop)
    g.add_edge("tools", "agent")

    return g.compile()


# -----------------------------
# 8) Utility: get last human-readable final answer
# -----------------------------
def get_last_assistant_text(messages: List[BaseMessage]) -> str:
    """
    Returns the last assistant message that has non-empty content.
    (Tool-call messages often have empty content.)
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            txt = (m.content or "").strip()
            if txt:
                return txt
    return "(No final answer text produced.)"


# -----------------------------
# 9) Simple runner (for testing)
# -----------------------------
if __name__ == "__main__":
    app = build_graph()

    # Example 1: grounding question
    state = {"messages": [HumanMessage(content="How should I handle missing values during EDA?")]}
    out = app.invoke(state)

    print("\n=== FINAL ANSWER (Example 1) ===")
    print(get_last_assistant_text(out["messages"]))

    # Example 2: plan generation
    state2 = {
        "messages": [
            HumanMessage(
                content="My dataset columns are: age, salary, city, join_date. "
                        "Goal: find patterns and outliers. Make an EDA plan."
            )
        ]
    }
    out2 = app.invoke(state2)

    print("\n=== FINAL ANSWER (Example 2) ===")
    print(get_last_assistant_text(out2["messages"]))

    # Optional: print full trace for debugging
    # print("\n--- TRACE ---")
    # for msg in out2["messages"]:
    #     print(type(msg).__name__, "tool_calls=", getattr(msg, "tool_calls", None), "content=", (msg.content or "")[:120])