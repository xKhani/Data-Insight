import os
import json
from tqdm import tqdm
from rag.config import KB_DIR, OUTPUT_DIR
from rag.utils_text import clean_text, chunk_text, infer_metadata

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []

    for root, _, files in os.walk(KB_DIR):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)

                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()

                cleaned = clean_text(raw)
                chunks = chunk_text(cleaned)

                meta = infer_metadata(file)

                for i, chunk in enumerate(chunks):
                    item = {
                        "id": f"{file}_{i}",
                        "text": chunk,
                        "metadata": meta
                    }
                    all_chunks.append(item)

    out_path = os.path.join(OUTPUT_DIR, "chunks_preview.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Created {len(all_chunks)} chunks")

if __name__ == "__main__":
    main()