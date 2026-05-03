import asyncio
from agent.multi_agent_graph import build_graph
from langchain_core.messages import HumanMessage
from agent.csv_path import CSV_PATH

def test_graph():
    app = build_graph()
    initial_state = {
        "messages": [HumanMessage(content="hi")],
        "csv_path": CSV_PATH,
        "user_goal": "hi",
        "user_input": "hi",
        "dataset_summary": {},
        "grounding_context": "",
        "eda_plan": {},
        "final_output": "",
        "sanitized_output": "",
        "proposed_save_path": "test.txt",
        "human_decision": "pending",
        "human_feedback": "",
        "save_result": {},
        "safety_status": "",
        "safety_reason": "",
    }
    config = {"configurable": {"thread_id": "test-thread"}}
    print("Invoking graph...")
    try:
        app.invoke(initial_state, config=config)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graph()
