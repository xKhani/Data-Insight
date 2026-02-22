# agent/tools.py
from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool

# Reuse your existing settings (adjust imports if your paths differ)
from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME


# -----------------------------
# Tool 1: Grounding (Vector DB)
# -----------------------------
class GroundingInput(BaseModel):
    """Input schema for querying the EDA knowledge base."""
    query: str = Field(..., min_length=3, description="User question to search in EDA knowledge base.")
    top_k: int = Field(3, ge=1, le=8, description="Number of retrieved chunks to return.")
    topic: Optional[str] = Field(
        None,
        description="Optional metadata filter. Examples: 'missing_values', 'correlation', 'workflow', 'visualization', 'eda_general'."
    )


@tool("search_eda_kb", args_schema=GroundingInput)
def search_eda_kb(query: str, top_k: int = 3, topic: Optional[str] = None) -> str:
    """
    Use this tool when you need grounded EDA guidance (steps, best practices, definitions)
    from the project's curated knowledge base. Supports optional topic filtering.
    Returns the most relevant chunks as text.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode([query]).tolist()

    where = {"topic": topic} if topic else None

    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        where=where
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    if not docs:
        return "No relevant grounding found in KB."

    # Return a compact, LLM-friendly grounded context
    out_lines = ["GROUNDING RESULTS:"]
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        doc_type = (m or {}).get("doc_type", "unknown")
        tpc = (m or {}).get("topic", "unknown")
        src = (m or {}).get("source", "unknown")
        snippet = (d or "").strip().replace("\n", " ")
        out_lines.append(f"{i}) ({doc_type}, topic={tpc}, source={src}) {snippet}")

    return "\n".join(out_lines)


# -----------------------------
# Tool 2: Action Tool (Example)
# -----------------------------
class EDAPlanInput(BaseModel):
    """Input schema to create a structured EDA plan."""
    dataset_columns: List[str] = Field(..., min_length=1, description="List of dataset column names.")
    goal: str = Field(..., min_length=3, description="What the user wants to learn from EDA (e.g., trends, anomalies, relationships).")


@tool("create_eda_plan", args_schema=EDAPlanInput)
def create_eda_plan(dataset_columns: List[str], goal: str) -> Dict[str, Any]:
    """
    Use this tool to generate a clean step-by-step EDA plan (tasks + recommended plots)
    given dataset columns and a user goal. This is a project-specific action tool.
    """
    # Simple deterministic plan (keeps it reliable for viva)
    plan = {
        "goal": goal,
        "steps": [
            "Check dataset shape, data types, and basic schema validation",
            "Compute missing values per column and decide handling approach",
            "Compute summary statistics for numeric columns",
            "Check distributions (histograms / density) for numeric columns",
            "Detect outliers (IQR / boxplots) for key numeric columns",
            "Check correlations between numeric columns (correlation matrix / heatmap)",
            "Visualize key relationships (scatter plots) based on goal",
            "Summarize insights and potential data quality issues"
        ],
        "recommended_plots": [
            "Missing values bar chart",
            "Histograms for numeric columns",
            "Boxplots for outlier inspection",
            "Correlation heatmap",
            "Scatter plots for top correlated pairs"
        ],
        "columns_seen": dataset_columns[:]
    }
    return plan


# Export tool list for graph.py
TOOLS = [search_eda_kb, create_eda_plan]