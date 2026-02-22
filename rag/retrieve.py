import chromadb
from sentence_transformers import SentenceTransformer
from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME

def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("\nAsk something about EDA (type 'exit' to stop)\n")

    while True:
        query = input("Query: ")

        if query == "exit":
            break

        q_emb = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=q_emb,
            n_results=3
        )

        print("\nTop results:\n")
        for doc in results["documents"][0]:
            print("-", doc[:200])
            print()

if __name__ == "__main__":
    main()