import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "../Vector_Store")
DATA_DIR = os.path.join(BASE_DIR, "../data")

# Create directories if they don't exist
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Model Options (Task 2)
# Keys are what the user sees, Values are the actual HF model names
EMBEDDING_MODELS = {
    "All-MiniLM-L6-v2 (Fast & Light)": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE-Small-EN (High Performance)": "BAAI/bge-small-en-v1.5",
    "BERT-Base-Uncased": "bert-base-uncased"
}

# Vector Store Options (Task 2)
VECTOR_STORES = {
    "FAISS": "faiss",
    "Chroma": "chroma"
}