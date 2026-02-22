import os, json
import chromadb
from sentence_transformers import SentenceTransformer
from rag.config import OUTPUT_DIR, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(EMBED_MODEL_NAME)

    file_path = os.path.join(OUTPUT_DIR, "chunks_preview.jsonl")

    docs, ids, metas = [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(row["text"])
            ids.append(row["id"])
            metas.append(row["metadata"])

    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=ids,
        metadatas=metas
    )

    print("✅ Chroma DB built successfully")

if __name__ == "__main__":
    main()