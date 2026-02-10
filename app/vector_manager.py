import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from app.config import VECTOR_STORE_DIR, EMBEDDING_MODELS, VECTOR_STORES

def get_embedding_model(model_selection):
    """
    Initializes the HuggingFace embedding model based on user selection.
    """
    # config.py has keys like "All-MiniLM...", we need the value
    model_name = EMBEDDING_MODELS.get(model_selection)
    
    if not model_name:
        raise ValueError(f"Invalid model selection: {model_selection}")
        
    print(f"Loading embedding model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if you have an NVIDIA GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def create_vector_db(chunks, model_selection, db_type):
    """
    Creates and saves a Vector DB (FAISS or Chroma) from document chunks.
    """
    embeddings = get_embedding_model(model_selection)
    
    # Define a specific path for this setup so we can load it later
    # Structure: Vector_Store/FAISS_All-MiniLM.../
    safe_name = model_selection.replace(" ", "_").replace("(", "").replace(")", "")
    persist_path = os.path.join(VECTOR_STORE_DIR, f"{db_type}_{safe_name}")
    
    print(f"Creating {db_type} database at {persist_path}...")

    if db_type == "FAISS":
        # FAISS is purely in-memory but can be saved to disk
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(persist_path)
        return vector_store, path_msg(persist_path)

    elif db_type == "Chroma":
        # Chroma requires a persistent directory
        # Warning: Chroma appends by default. For this assignment, 
        # we might want to clean it to ensure a fresh start for the demo.
        if os.path.exists(persist_path):
            try:
                shutil.rmtree(persist_path) # clear old DB to avoid duplicates for now
            except:
                pass 
                
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_path
        )
        return vector_store, path_msg(persist_path)
    
    else:
        raise ValueError("Unsupported Vector DB Type")

def get_vector_db(model_selection, db_type):
    """
    Loads an existing Vector DB for querying.
    """
    embeddings = get_embedding_model(model_selection)
    safe_name = model_selection.replace(" ", "_").replace("(", "").replace(")", "")
    persist_path = os.path.join(VECTOR_STORE_DIR, f"{db_type}_{safe_name}")

    if not os.path.exists(persist_path):
        return None

    if db_type == "FAISS":
        return FAISS.load_local(
            persist_path, 
            embeddings, 
            allow_dangerous_deserialization=True # Required for local files
        )
    elif db_type == "Chroma":
        return Chroma(
            persist_directory=persist_path, 
            embedding_function=embeddings
        )

def path_msg(path):
    return f"Database saved to: {path}"