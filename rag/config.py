import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Knowledge base folder (your EDA handbook)
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

# Where chunks preview will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Local vector database folder
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")

COLLECTION_NAME = "eda_knowledge"

# Free local embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings
MAX_CHARS = 1200
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 200