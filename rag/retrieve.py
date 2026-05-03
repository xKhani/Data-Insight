import os
import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME, OUTPUT_DIR


def _run_query(collection, model, query: str, where=None, k: int = 5):
    """Return top-k hits with text + metadata."""
    q_emb = model.encode([query]).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        where=where
    )

    hits = []
    if not res or "documents" not in res or not res["documents"]:
        return hits

    docs = res["documents"][0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    for i in range(min(len(docs), k)):
        hits.append({
            "rank": i + 1,
            "id": ids[i] if i < len(ids) else None,
            "text": docs[i],
            "meta": metas[i] if i < len(metas) else {}
        })
    return hits


def _infer_topic_from_query(query: str):
    """Simple topic routing to improve retrieval relevance."""
    q = query.lower()

    if any(word in q for word in ["missing", "null", "impute", "imputation"]):
        return "missing_values"
    if any(word in q for word in ["outlier", "anomaly", "extreme", "iqr", "z-score", "boxplot"]):
        return "outliers"
    if any(word in q for word in ["correlation", "relationship", "heatmap", "scatter"]):
        return "correlation"
    if any(word in q for word in ["visual", "plot", "chart", "graph", "histogram", "bar chart"]):
        return "visualization"
    if any(word in q for word in ["workflow", "steps", "process", "eda plan"]):
        return "workflow"

    return None


_RETRIEVER_CACHE = {}


def get_retriever():
    """Load DB + embedding model once (singleton) and return them."""
    if "model" not in _RETRIEVER_CACHE:
        # Avoid redundant httpx connections by loading once
        _RETRIEVER_CACHE["model"] = SentenceTransformer(EMBED_MODEL_NAME)

    if "client" not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE["client"] = chromadb.PersistentClient(path=CHROMA_DIR)
        _RETRIEVER_CACHE["collection"] = _RETRIEVER_CACHE["client"].get_collection(name=COLLECTION_NAME)

    return _RETRIEVER_CACHE["collection"], _RETRIEVER_CACHE["model"]


def get_grounded_context(query: str, k: int = 5, use_topic_filter: bool = True) -> str:
    """
    Return clean retrieved context for the agent.
    This is the function your tool or graph should use.
    """
    collection, model = get_retriever()

    where = None
    if use_topic_filter:
        topic = _infer_topic_from_query(query)
        if topic:
            where = {"topic": topic}

    hits = _run_query(collection, model, query, where=where, k=k)

    if not hits and where is not None:
        # fallback without metadata filter
        hits = _run_query(collection, model, query, where=None, k=k)

    if not hits:
        return "No relevant EDA guidance retrieved."

    cleaned_chunks = []
    for hit in hits:
        snippet = (hit.get("text", "") or "").replace("\n", " ").strip()
        if snippet:
            cleaned_chunks.append(snippet)

    return "\n\n".join(cleaned_chunks)


def get_grounded_context_with_metadata(query: str, k: int = 5, use_topic_filter: bool = True) -> str:
    """
    Optional debug version: keeps lightweight metadata for inspection,
    but cleaner than raw markdown demo output.
    """
    collection, model = get_retriever()

    where = None
    if use_topic_filter:
        topic = _infer_topic_from_query(query)
        if topic:
            where = {"topic": topic}

    hits = _run_query(collection, model, query, where=where, k=k)

    if not hits and where is not None:
        hits = _run_query(collection, model, query, where=None, k=k)

    if not hits:
        return "No relevant EDA guidance retrieved."

    lines = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("meta", {}) or {}
        topic = meta.get("topic", "unknown")
        snippet = (hit.get("text", "") or "").replace("\n", " ").strip()
        lines.append(f"{i}. [topic: {topic}] {snippet}")

    return "\n\n".join(lines)


def _format_hit_md(hit: dict, max_chars: int = 450) -> str:
    meta = hit.get("meta", {}) or {}
    source = meta.get("source", "unknown")
    doc_type = meta.get("doc_type", "unknown")
    topic = meta.get("topic", "unknown")

    snippet = (hit.get("text", "") or "").replace("\n", " ").strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."

    return (
        f"**Hit {hit['rank']}** — `doc_type={doc_type}`, `topic={topic}`, `source={source}`\n\n"
        f"> {snippet}\n"
    )


def _write_retrieval_test_md(out_path: str, tests: list, all_results: list):
    lines = []
    lines.append("# retrieval_test\n")
    lines.append("This file documents retrieval queries against the local ChromaDB index.\n")
    lines.append("At least one test demonstrates metadata filtering.\n")

    for t, hits in zip(tests, all_results):
        lines.append(f"\n---\n\n## {t['title']}\n")
        lines.append(f"**Query:** {t['query']}\n")
        lines.append(f"**Filter (where):** `{t.get('where')}`\n")

        if not hits:
            lines.append("\n_No results returned._\n")
            continue

        lines.append("\n### Top Results\n")
        for h in hits:
            lines.append(_format_hit_md(h))
            lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    collection, model = get_retriever()

    tests = [
        {
            "title": "Test 1 — EDA workflow steps",
            "query": "Give me the step-by-step EDA workflow in the correct order.",
            "where": {"topic": "workflow"}
        },
        {
            "title": "Test 2 — Handling missing values",
            "query": "How should I handle missing values during EDA? Give best practices.",
            "where": {"topic": "missing_values"}
        },
        {
            "title": "Test 3 — Correlation analysis",
            "query": "Explain correlation analysis and how to interpret correlation strength.",
            "where": {"topic": "correlation"}
        }
    ]

    all_results = []
    for t in tests:
        hits = _run_query(collection, model, t["query"], where=t.get("where"), k=5)
        all_results.append(hits)

    out_md = os.path.join(OUTPUT_DIR, "retrieval_test.md")
    _write_retrieval_test_md(out_md, tests, all_results)
    print(f"SUCCESS: Wrote retrieval tests to: {out_md}")

    print("\n=== Quick Preview (Top 1 from each test) ===")
    for t, hits in zip(tests, all_results):
        print(f"\n[{t['title']}]")
        if hits:
            print(_format_hit_md(hits[0], max_chars=220))
        else:
            print("No results.")

    print("\nNow you can ask custom questions. Type 'exit' to stop.\n")
    while True:
        q = input("Query: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        print("\n=== Clean Grounded Context ===\n")
        print(get_grounded_context(q, k=5, use_topic_filter=True))
        print("\n")


if __name__ == "__main__":
    main()