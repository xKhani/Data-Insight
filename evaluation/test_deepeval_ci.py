import json
import os
import sys
from datetime import datetime

# Ensure the project root is in path for agent imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from agent.multi_agent_graph import build_graph
from agent.csv_path import CSV_PATH, USER_GOAL, DEFAULT_SAVE_PATH
from langchain_core.messages import AIMessage

# 1. Custom Groq Wrapper for DeepEval
class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        self.chat_model = ChatGroq(model=model_name)

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = chat_model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

# Initialize Groq Judge
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set.")
    sys.exit(1)

groq_judge = GroqDeepEvalModel()

def extract_actual_tool_calls(messages):
    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc["name"], "arguments": tc.get("args") or {}})
    return tool_calls

def run_deepeval_ci(dataset_path="test_dataset.json"):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Limit to 5 critical cases to stay within Groq rate limits
    target_cases = dataset[:5]
    
    app = build_graph()
    results = []
    
    print(f"Starting Groq-powered DeepEval evaluation on {len(target_cases)} cases...")

    for case in target_cases:
        query = case["query"]
        gt = case.get("ground_truth", {})
        expected_behavior = str(gt.get("expected_behavior", "A grounded EDA response."))
        
        print(f"\n[Case {case['id']}] Processing...")

        initial_state = {
            "user_input": query,
            "csv_path": CSV_PATH,
            "user_goal": USER_GOAL,
            "proposed_save_path": DEFAULT_SAVE_PATH,
            "human_decision": "approve", 
            "human_feedback": "",
        }
        
        config = {"configurable": {"thread_id": f"ci_deepeval_{case['id']}"}}
        state_result = app.invoke(initial_state, config=config)

        actual_output = state_result.get("final_output", "")
        context = [state_result.get("grounding_context", "")] if state_result.get("grounding_context") else []
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_behavior,
            retrieval_context=context
        )

        # Metrics
        faithfulness = FaithfulnessMetric(threshold=0.7, model=groq_judge)
        relevancy = AnswerRelevancyMetric(threshold=0.7, model=groq_judge)
        
        faithfulness.measure(test_case)
        relevancy.measure(test_case)

        print(f"   - Relevancy: {relevancy.score:.2f} | Faithfulness: {faithfulness.score:.2f}")

        results.append({
            "id": case["id"],
            "scores": {"relevancy": relevancy.score, "faithfulness": faithfulness.score}
        })

    # Save CI report
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": results,
        "overall_pass": all(r["scores"]["relevancy"] >= 0.7 and r["scores"]["faithfulness"] >= 0.7 for r in results)
    }
    
    with open("ci_deepeval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\nGroq Evaluation Complete. Report saved to ci_deepeval_report.json")

if __name__ == "__main__":
    run_deepeval_ci()
