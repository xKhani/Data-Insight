from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from agent.multi_agent_graph import build_graph
from agent.csv_path import CSV_PATH, DEFAULT_SAVE_PATH
from api.schema import ChatRequest, ChatResponse, ApproveRequest


graph_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build graph once when FastAPI starts.
    This keeps the checkpointer initialized globally instead of rebuilding on every request.
    """
    global graph_app
    graph_app = build_graph()
    yield


app = FastAPI(
    title="Data-Insight Agent API",
    description="FastAPI layer for the Data-Insight LangGraph EDA agent",
    version="1.0.0",
    lifespan=lifespan,
)


def make_initial_state(message: str) -> dict:
    """
    Creates the same initial state your CLI graph expects.
    """
    return {
        "messages": [HumanMessage(content=message)],
        "csv_path": CSV_PATH,
        "user_goal": message,
        "user_input": message,
        "dataset_summary": {},
        "grounding_context": "",
        "eda_plan": {},
        "final_output": "",
        "sanitized_output": "",
        "proposed_save_path": DEFAULT_SAVE_PATH,
        "human_decision": "pending",
        "human_feedback": "",
        "save_result": {},
        "safety_status": "",
        "safety_reason": "",
    }


def build_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.get("/")
def root():
    return {"message": "Data-Insight Agent API is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Normal non-streaming endpoint.
    Runs the graph until END or until HITL interruption before save_agent.
    """
    try:
        if graph_app is None:
            raise RuntimeError("Graph app was not initialized.")

        config = build_config(request.thread_id)
        initial_state = make_initial_state(request.message)

        graph_app.invoke(initial_state, config=config)

        snapshot = graph_app.get_state(config)
        values = snapshot.values

        final_output = values.get("sanitized_output") or values.get("final_output") or ""
        safety_status = values.get("safety_status", "")
        next_nodes = list(getattr(snapshot, "next", []) or [])

        waiting_for_hitl = "save_agent" in next_nodes

        if safety_status == "UNSAFE":
            status = "blocked"
        elif waiting_for_hitl:
            status = "waiting_for_human_approval"
        else:
            status = "completed"

        resp = ChatResponse(
            answer=final_output,
            status=status,
            thread_id=request.thread_id,
            safety_status=safety_status,
            waiting_for_hitl=waiting_for_hitl,
        )
        return resp
    except Exception as e:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"\n--- ERROR at {__name__} ---\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise e


@app.post("/stream")
async def stream(request: ChatRequest):
    """
    Streaming endpoint using graph.astream().
    Sends node-by-node updates in Server-Sent Events format.
    """

    if graph_app is None:
        raise RuntimeError("Graph app was not initialized.")

    config = build_config(request.thread_id)
    initial_state = make_initial_state(request.message)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in graph_app.astream(initial_state, config=config):
                # LangGraph events look like: {"node_name": {"field": "value"}}
                for node_name, values in event.items():
                    # Map internal node names to friendly display names
                    friendly_names = {
                        "guardrail_node": "Security Check",
                        "csv_inspector_agent": "CSV Inspection",
                        "grounding_agent": "Knowledge Retrieval",
                        "planning_agent": "EDA Planning",
                        "coordinator_agent": "Drafting Proposal",
                        "save_agent": "Saving to Disk",
                        "csv_tools": "Executing CSV Tools",
                        "grounding_tools": "Searching Knowledge Base",
                    }
                    
                    step = friendly_names.get(node_name, node_name)
                    content = ""
                    
                    if "messages" in values:
                        last_msg = values["messages"][-1]
                        if isinstance(last_msg, list): last_msg = last_msg[-1]
                        content = getattr(last_msg, "content", str(last_msg))
                        # If content is a tool call, make it prettier
                        if "tool_calls" in str(content):
                            content = "Processing..."
                    elif "eda_plan" in values:
                        content = "Plan generated."
                    elif "grounding_context" in values:
                        content = "Context retrieved."
                    
                    payload = {
                        "type": "node_update",
                        "step": step,
                        "status": content[:100] + "..." if len(str(content)) > 100 else content,
                    }
                    yield f"data: {json.dumps(payload, default=str)}\n\n"

            # Final Summary
            snapshot = graph_app.get_state(config)
            values = snapshot.values
            final_output = values.get("sanitized_output") or values.get("final_output") or ""
            safety_status = values.get("safety_status", "")
            next_nodes = list(getattr(snapshot, "next", []) or [])

            done_payload = {
                "type": "done",
                "answer": final_output,
                "safety_status": safety_status,
                "waiting_for_hitl": "save_agent" in next_nodes,
                "thread_id": request.thread_id,
            }

            yield f"data: {json.dumps(done_payload, default=str)}\n\n"

        except Exception as e:
            error_payload = {
                "type": "error",
                "message": str(e),
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.post("/approve")
def approve(request: ApproveRequest):
    """
    Human approves the current draft.
    Resumes the graph with human_decision='approve'.
    """
    try:
        if graph_app is None:
            raise RuntimeError("Graph app was not initialized.")

        config = build_config(request.thread_id)
        
        # Update state to approve
        graph_app.update_state(config, {"human_decision": "approve"})
        
        # Resume graph
        graph_app.invoke(None, config=config)

        # Get final state after save
        snapshot = graph_app.get_state(config)
        values = snapshot.values
        save_result = values.get("save_result", {})

        return {
            "status": "completed",
            "save_result": save_result,
            "thread_id": request.thread_id
        }
    except Exception as e:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"\n--- ERROR at approve ---\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise e


@app.post("/cancel")
def cancel(request: ApproveRequest):
    """
    Human cancels the current draft.
    Resumes the graph with human_decision='cancel'.
    """
    try:
        if graph_app is None:
            raise RuntimeError("Graph app was not initialized.")

        config = build_config(request.thread_id)
        
        graph_app.update_state(config, {"human_decision": "cancel"})
        graph_app.invoke(None, config=config)

        return {
            "status": "cancelled",
            "thread_id": request.thread_id
        }
    except Exception as e:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"\n--- ERROR at cancel ---\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise e