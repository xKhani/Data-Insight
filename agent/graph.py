# agent/graph.py
from __future__ import annotations

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agent.tools import TOOLS


# -----------------------------
# 1) Graph State (TypedDict)
# -----------------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]  # chat history / reasoning trace


# -----------------------------
# 2) LLM (Ollama)
# -----------------------------
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.2
).bind_tools(TOOLS)


SYSTEM_PROMPT = """You are Data Insight, an autonomous data analysis assistant.
You MUST use tools when needed:
- Use search_eda_kb to retrieve grounded EDA guidance (workflow, missing values, outliers, correlation, visualization).
- Use create_eda_plan when user provides dataset columns and a goal and needs a structured EDA plan.

Follow a ReAct style:
Think -> decide tool -> use tool -> observe -> final answer.
If you have enough info, provide a final answer."""
# (Keep it short; qwen2.5-coder follows compact system prompts better)


# -----------------------------
# 3) Agent Node (LLM thinking)
# -----------------------------
def agent_node(state: GraphState) -> GraphState:
    messages = state["messages"]

    # Ensure system prompt is present once at the beginning
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(messages)
    return {"messages": messages + [response]}


# -----------------------------
# 4) Tool Node (exec tool calls)
# -----------------------------
tool_node = ToolNode(TOOLS)


# -----------------------------
# 5) Router (conditional edge)
# -----------------------------
def router(state: GraphState) -> str:
    """
    If the last LLM message contains tool calls -> go to tools
    else -> end (final answer)
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END


# -----------------------------
# 6) Build Graph
# -----------------------------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)

    g.set_entry_point("agent")

    # agent -> (tools or end)
    g.add_conditional_edges("agent", router, {"tools": "tools", END: END})

    # tools -> agent (loop)
    g.add_edge("tools", "agent")

    return g.compile()


# -----------------------------
# 7) Simple runner (for testing)
# -----------------------------
if __name__ == "__main__":
    app = build_graph()

    # Example 1: asks grounded EDA question -> should call search_eda_kb
    state = {"messages": [HumanMessage(content="How should I handle missing values during EDA?")]}
    out = app.invoke(state)
    print("\nFINAL:\n", out["messages"][-1].content)

    # Example 2: plan generation (action tool)
    state2 = {"messages": [HumanMessage(content="My dataset columns are: age, salary, city, join_date. Goal: find patterns and outliers. Make an EDA plan.")]}
    out2 = app.invoke(state2)
    print("\nFINAL:\n", out2["messages"][-1].content)