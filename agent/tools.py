from __future__ import annotations

import json
import os
from typing import Optional, List
from pydantic import BaseModel, Field

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool

from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME
from rag.retrieve import get_grounded_context_with_metadata


# -----------------------------
# Tool 1: CSV Inspection Tool
# -----------------------------
class CSVInspectInput(BaseModel):
    """Input schema for inspecting a CSV file."""
    file_path: str = Field(..., min_length=3, description="Absolute or relative path to the CSV file.")


@tool("inspect_csv", args_schema=CSVInspectInput)
def inspect_csv(file_path: str) -> str:
    """
    Read a CSV file and return a compact JSON summary including:
    shape, columns, dtypes, missing counts, numeric columns,
    categorical columns, datetime-like columns, and sample rows.
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"CSV file not found: {file_path}"}, ensure_ascii=False)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return json.dumps({"error": f"Failed to read CSV: {str(e)}"}, ensure_ascii=False)

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_counts = {col: int(val) for col, val in df.isnull().sum().to_dict().items()}

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    datetime_columns = []
    for col in df.columns:
        lowered = col.lower()
        if "date" in lowered or "time" in lowered:
            datetime_columns.append(col)

    sample_rows = df.head(3).fillna("").to_dict(orient="records")

    summary = {
        "file_path": file_path,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "sample_rows": sample_rows,
    }

    return json.dumps(summary, ensure_ascii=False)


# -----------------------------
# Tool 2: Grounding Tool (RAG)
# -----------------------------
class GroundingInput(BaseModel):
    """Input schema for querying the EDA knowledge base."""
    query: str = Field(..., min_length=3, description="Question to search in the EDA knowledge base.")
    top_k: int = Field(3, ge=1, le=8, description="Number of chunks to return.")
    topic: Optional[str] = Field(
        None,
        description="Optional topic filter such as missing_values, correlation, workflow, visualization, eda_general."
    )


@tool("search_eda_kb", args_schema=GroundingInput)
def search_eda_kb(query: str, top_k: int = 3, topic: Optional[str] = None) -> str:
    """
    Retrieve grounded EDA guidance from the Chroma vector database.
    """
    # Use topic filter for better relevance if a topic is provided or inferred later
    context = get_grounded_context_with_metadata(query, k=top_k, use_topic_filter=True)

    if context == "No relevant EDA guidance retrieved.":
        return "No relevant grounding found in KB."

    return f"GROUNDING RESULTS:\n{context}"


# -----------------------------
# Tool 3: EDA Proposal Tool
# -----------------------------
class EDAPlanInput(BaseModel):
    """Input schema for generating an EDA plan."""
    dataset_columns: List[str] = Field(..., min_length=1, description="List of dataset columns.")
    goal: str = Field(..., min_length=3, description="What the user wants to achieve from EDA.")
    numeric_columns: List[str] = Field(default_factory=list, description="Numeric columns.")
    categorical_columns: List[str] = Field(default_factory=list, description="Categorical columns.")
    datetime_columns: List[str] = Field(default_factory=list, description="Datetime-like columns.")
    grounding_context: Optional[str] = Field(None, description="Grounded EDA guidance from the KB.")


@tool("create_eda_plan", args_schema=EDAPlanInput)
def create_eda_plan(
    dataset_columns: List[str],
    goal: str,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None,
    grounding_context: Optional[str] = None,
) -> str:
    """
    Create a structured EDA plan based on dataset structure, user goal,
    and optional grounded EDA knowledge.
    """
    numeric_columns = numeric_columns or []
    categorical_columns = categorical_columns or []
    datetime_columns = datetime_columns or []

    recommended_plots = []

    if numeric_columns:
        recommended_plots.extend([
            "Histograms for numeric distributions",
            "Boxplots for outlier detection",
        ])

    if len(numeric_columns) >= 2:
        recommended_plots.extend([
            "Correlation heatmap",
            "Scatter plots for top numeric relationships",
        ])

    if categorical_columns:
        recommended_plots.append("Bar charts for categorical distributions")

    if datetime_columns:
        recommended_plots.append("Line charts for time-based trends")

    steps = [
        "Validate dataset shape, schema, and data types",
        "Check missing values column-wise and decide whether deletion or imputation is appropriate",
        "Generate descriptive statistics for numeric columns",
    ]

    if numeric_columns:
        steps.append("Analyze numeric feature distributions and skewness using histograms or density plots")
        steps.append("Detect outliers in key numeric columns using IQR and boxplots")

    if len(numeric_columns) >= 2:
        steps.append("Measure relationships between numeric variables using correlation analysis")

    if categorical_columns:
        steps.append("Inspect category frequencies and compare group-level patterns")

    if datetime_columns:
        steps.append("Inspect time-based trends for datetime-related fields")

    steps.append("Summarize key insights, anomalies, and data quality concerns into an EDA proposal")

    plan = {
        "goal": goal,
        "dataset_columns": dataset_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "recommended_plots": recommended_plots,
        "steps": steps,
        "grounding_used": bool(grounding_context and grounding_context.strip()),
    }

    return json.dumps(plan, ensure_ascii=False)


# -----------------------------
# Tool 4: High-Risk Action Tool
# -----------------------------
class SaveProposalInput(BaseModel):
    """Input schema for saving the final EDA proposal."""
    output_text: str = Field(..., min_length=10, description="Final EDA proposal text to save.")
    save_path: str = Field(..., min_length=3, description="Path where the proposal should be saved.")


@tool("save_eda_proposal", args_schema=SaveProposalInput)
def save_eda_proposal(output_text: str, save_path: str) -> str:
    """
    High-risk action tool.
    Writes the final EDA proposal to disk. This should only run after human approval.
    """
    try:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        result = {
            "status": "saved",
            "save_path": save_path,
            "characters_written": len(output_text),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


TOOLS = [inspect_csv, search_eda_kb, create_eda_plan, save_eda_proposal]