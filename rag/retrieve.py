import os
import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME, OUTPUT_DIR


def _run_query(collection, model, query: str, where=None, k: int = 3):
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
    lines.append("This file documents 3 retrieval queries against the local ChromaDB index.\n")
    lines.append("At least one test demonstrates metadata filtering as required.\n")

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
            lines.append("")  # blank line

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    # Load DB + embedding model
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # ✅ 3 required tests (one includes metadata filtering)
    tests = [
        {
            "title": "Test 1 — EDA workflow steps",
            "query": "Give me the step-by-step EDA workflow in the correct order.",
            "where": None
        },
        {
            "title": "Test 2 — Handling missing values",
            "query": "How should I handle missing values during EDA? Give best practices.",
            "where": None
        },
        {
            "title": "Test 3 — Metadata filtering (ONLY correlation topic)",
            "query": "Explain correlation analysis and how to interpret correlation strength.",
            # ✅ metadata filter requirement
            "where": {"topic": "correlation"}
        }
    ]

    all_results = []
    for t in tests:
        hits = _run_query(collection, model, t["query"], where=t.get("where"), k=3)
        all_results.append(hits)

    # Write the required markdown file
    out_md = os.path.join(OUTPUT_DIR, "retrieval_test.md")
    _write_retrieval_test_md(out_md, tests, all_results)
    print(f"✅ Wrote retrieval tests to: {out_md}")

    # Optional: show results in console too
    print("\n=== Quick Preview (Top 1 from each test) ===")
    for t, hits in zip(tests, all_results):
        print(f"\n[{t['title']}]")
        if hits:
            print(_format_hit_md(hits[0], max_chars=220))
        else:
            print("No results.")

    # Interactive mode (for demo)
    print("\nNow you can ask custom questions. Type 'exit' to stop.\n")
    while True:
        q = input("Query: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # Optional: quick filter commands for demo
        # Example: type "filter:topic=outliers | how to detect outliers?"
        where = None
        if q.lower().startswith("filter:") and "|" in q:
            filt_part, real_q = q.split("|", 1)
            real_q = real_q.strip()

            # parse filter: key=value pairs separated by commas
            filt_part = filt_part[len("filter:"):].strip()
            pairs = [p.strip() for p in filt_part.split(",") if p.strip()]
            where = {}
            for p in pairs:
                if "=" in p:
                    k, v = p.split("=", 1)
                    where[k.strip()] = v.strip()

            q = real_q

        hits = _run_query(collection, model, q, where=where, k=3)
        if not hits:
            print("\nNo results.\n")
            continue

        print("\nTop results:\n")
        for h in hits:
            print(_format_hit_md(h, max_chars=260))
            print()


if __name__ == "__main__":
    main()