import streamlit as st
import os
from app.utils import load_documents, chunk_documents
from app.vector_manager import create_vector_db, get_vector_db
from app.config import EMBEDDING_MODELS, VECTOR_STORES

def main():
    st.set_page_config(page_title="AI Research Assistant", layout="wide")
    st.title("🤖 AI Research Assistant (Semantic Search)")

    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 1. Select Embedding Model
        model_choice = st.selectbox(
            "Choose Embedding Model",
            options=list(EMBEDDING_MODELS.keys())
        )
        
        # 2. Select Vector Database
        db_choice = st.selectbox(
            "Choose Vector Database",
            options=list(VECTOR_STORES.keys())
        )
        
        st.divider()

        # 3. Upload Data
        st.subheader("📂 Dataset Selection")
        uploaded_files = st.file_uploader(
            "Upload Research Papers (PDF/TXT)", 
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if uploaded_files:
            st.info(f"{len(uploaded_files)} files selected")
            
            if st.button("Process & Index Documents"):
                with st.spinner("Processing..."):
                    # Load
                    raw_docs = load_documents(uploaded_files)
                    st.write(f"Loaded {len(raw_docs)} documents.")
                    
                    # Split
                    chunks = chunk_documents(raw_docs)
                    st.write(f"Split into {len(chunks)} chunks.")
                    
                    # Embed & Store
                    # Map friendly names to internal keys if needed, 
                    # but our config keys match the display names mostly.
                    # We pass the raw selection string to the manager.
                    vector_store, msg = create_vector_db(chunks, model_choice, db_choice)
                    
                    # Save to session state so we don't lose it
                    st.session_state['vector_db'] = vector_store
                    st.success("Indexing Complete!")
                    st.toast(msg)

    # Main Area: Search Interface
    st.subheader("🔍 Semantic Search")
    
    # Search Input
    query = st.text_input("Enter your research question:")
    
    # Top-K Slider
    top_k = st.slider("Number of results (Top-K)", min_value=1, max_value=10, value=3)

    if query:
        if 'vector_db' in st.session_state:
            with st.spinner("Searching..."):
                db = st.session_state['vector_db']
                
                # Perform Similarity Search
                results = db.similarity_search_with_score(query, k=top_k)
                
                # Display Results
                st.write(f"### Results for: *{query}*")
                for i, (doc, score) in enumerate(results):
                    with st.expander(f"Result {i+1} (Score: {score:.4f})"):
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content)
        else:
            st.warning("⚠️ Please upload and index documents first!")

if __name__ == "__main__":
    main()