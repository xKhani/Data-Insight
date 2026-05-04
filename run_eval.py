from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def category_rate(summary: dict[str, Any], category: str) -> float:
    data = summary.get("by_category", {}).get(category, {})
    total = data.get("total", 0)
    passed = data.get("passed", 0)
    return 0.0 if total == 0 else passed / total


def main() -> int:
    parser = argparse.ArgumentParser(description="CI quality gate for Data-Insight agent.")
    parser.add_argument("--dataset", default="test_dataset.json")
    parser.add_argument("--thresholds", default="eval_thresholds.json")
    parser.add_argument("--results", default="ci_eval_results.json")
    parser.add_argument("--summary", default="ci_eval_summary.json")
    parser.add_argument("--gate-output", default="quality_gate_results.json")
    parser.add_argument("--eval-script", default="run_evaluation_fixed.py")
    args = parser.parse_args()

    print("Environment check:")
    print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2", "not set"))
    print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT", "not set"))
    print("LANGSMITH_API_KEY:", "set" if os.getenv("LANGSMITH_API_KEY") else "not set")
    print("OPENAI_API_KEY:", "set" if os.getenv("OPENAI_API_KEY") else "not set")

    for file in [args.dataset, args.thresholds, args.eval_script]:
        if not Path(file).exists():
            print(f"ERROR: Required file not found: {file}")
            return 1

    command = [
        sys.executable,
        args.eval_script,
        "--dataset",
        args.dataset,
        "--output",
        args.results,
        "--summary",
        args.summary,
    ]

    print("Running:", " ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        print("ERROR: Evaluation script failed.")
        return 1

    thresholds = load_json(args.thresholds)
    summary = load_json(args.summary)

    measured = {
        "minimum_pass_rate": float(summary.get("pass_rate", 0.0)),
        "minimum_average_content_coverage": float(summary.get("average_content_coverage", 0.0)),
        "minimum_security_pass_rate": category_rate(summary, "security"),
        "minimum_rag_grounding_pass_rate": category_rate(summary, "rag_grounding"),
    }

    gate_results = []
    all_passed = True

    for metric_name, threshold in thresholds.items():
        score = measured.get(metric_name, 0.0)
        passed = score >= float(threshold)
        all_passed = all_passed and passed
        gate_results.append({
            "metric": metric_name,
            "score": round(score, 4),
            "threshold": threshold,
            "passed": passed,
        })

    output = {
        "overall_passed": all_passed,
        "summary_file": args.summary,
        "results_file": args.results,
        "metrics": gate_results,
    }

    with open(args.gate_output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== QUALITY GATE RESULTS ===")
    print(json.dumps(output, indent=2))

    if all_passed:
        print("\nQUALITY GATE PASSED")
        return 0

    print("\nQUALITY GATE FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
