import json
import os
import sys
import random
from datetime import datetime

# Ensure the project root is in path for agent imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.models import OllamaModel
from agent.multi_agent_graph import build_graph
from agent.csv_path import CSV_PATH, USER_GOAL, DEFAULT_SAVE_PATH
from langchain_core.messages import AIMessage

# 1. Initialize local judge (Ollama)
JUDGE_MODEL = "qwen2.5:7b-instruct"
ollama_judge = OllamaModel(model=JUDGE_MODEL)

def extract_actual_tool_calls(messages):
    """
    Extracts tool calls from a list of LangChain messages.
    """
    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc["name"],
                    "arguments": tc.get("args") or {}
                })
    return tool_calls

def run_deepeval_randomized(dataset_path="evaluation/test_dataset.json", limit=3):
    """
    Runs DeepEval metrics on 3 randomized test cases.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Randomly select cases
    random_cases = random.sample(dataset, min(limit, len(dataset)))
    
    # Initialize the agent graph
    print("Initializing Multi-Agent Graph...")
    app = build_graph()
    
    results = []
    
    print(f"Starting DeepEval evaluation on {len(random_cases)} RANDOM test cases...")

    for case in random_cases:
        query = case["query"]
        gt = case.get("ground_truth", {})
        expected_behavior = str(gt.get("expected_behavior", "A grounded EDA response."))
        expected_tools_list = gt.get("expected_tools", [])

        print(f"\n[Case {case['id']}] Category: {case['category']} | Query: {query[:60]}...")

        # Execute agent graph
        initial_state = {
            "user_input": query,
            "csv_path": CSV_PATH,
            "user_goal": USER_GOAL,
            "proposed_save_path": DEFAULT_SAVE_PATH,
            "human_decision": "approve", 
            "human_feedback": "",
            "safety_status": "",
            "safety_reason": "",
        }
        
        config = {"configurable": {"thread_id": f"deepeval_v3_{case['id']}"}}
        state_result = app.invoke(initial_state, config=config)

        actual_output = state_result.get("final_output", "")
        # Get context from the state
        context = [state_result.get("grounding_context", "")] if state_result.get("grounding_context") else []
        # Extract tool calls
        actual_tool_calls = extract_actual_tool_calls(state_result.get("messages", []))
        
        # Prepare a detailed string of tool calls for the GEval judge
        tool_call_summary = "\n".join([f"- Tool: {tc['name']}, Args: {tc['arguments']}" for tc in actual_tool_calls])
        combined_output = f"{actual_output}\n\n[TRANSCRIPT: TOOL CALLS MADE]\n{tool_call_summary if tool_call_summary else 'No tools called.'}"

        # 2. Create DeepEval Test Case
        test_case = LLMTestCase(
            input=query,
            actual_output=combined_output,
            expected_output=expected_behavior,
            retrieval_context=context
        )

        # 3. Define and measure Metrics
        print("   [Step] Measuring Faithfulness and Relevancy...")
        faithfulness = FaithfulnessMetric(threshold=0.5, model=ollama_judge)
        relevancy = AnswerRelevancyMetric(threshold=0.5, model=ollama_judge)
        
        print("   [Step] Measuring Tool Accuracy (GEval Decimal Scoring)...")
        # Custom GEval for Tool Accuracy to get decimal scores for single tool calls
        tool_accuracy_metric = GEval(
            name="Tool Accuracy",
            criteria=(
                "Evaluate if the agent selected the correct tools and used valid arguments (like csv_path). "
                f"Expected tools for this query: {', '.join(expected_tools_list)}. "
                "Score 1.0 if perfect, 0.7-0.9 if correct tool but slightly redundant args, 0.5 if tool choice was okay but usage was flawed, 0.0 if wrong/no tool."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=ollama_judge
        )

        faithfulness.measure(test_case)
        relevancy.measure(test_case)
        tool_accuracy_metric.measure(test_case)

        print(f"   - Relevancy: {relevancy.score:.2f}")
        print(f"   - Faithfulness: {faithfulness.score:.2f}")
        print(f"   - Tool Accuracy: {tool_accuracy_metric.score:.2f}")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "query": query,
            "scores": {
                "relevancy": relevancy.score,
                "faithfulness": faithfulness.score,
                "tool_accuracy": tool_accuracy_metric.score
            },
            "reasoning": {
                "relevancy": getattr(relevancy, 'reason', ""),
                "faithfulness": getattr(faithfulness, 'reason', ""),
                "tool_accuracy": getattr(tool_accuracy_metric, 'reason', "")
            }
        })

    # Aggregates
    count = len(results)
    avg_rel = sum(r["scores"]["relevancy"] for r in results) / count
    avg_faith = sum(r["scores"]["faithfulness"] for r in results) / count
    avg_tool = sum(r["scores"]["tool_accuracy"] for r in results) / count

    summary = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "sample_size": count,
        "overall_metrics": {
            "avg_relevancy": avg_rel,
            "avg_faithfulness": avg_faith,
            "avg_tool_accuracy": avg_tool
        },
        "detailed_results": results
    }

    report_path = "evaluation/deepeval_results_v3.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("📊 RANDOMIZED DEEPEVAL SUMMARY (3 CASES)")
    print(f"Avg Relevancy: {avg_rel:.2f}")
    print(f"Avg Faithfulness: {avg_faith:.2f}")
    print(f"Avg Tool Accuracy: {avg_tool:.2f}")
    print(f"Report saved to: {report_path}")
    print("="*50)

if __name__ == "__main__":
    run_deepeval_randomized()
