import os
import pypdf
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_pdf_text(file):
    """
    Extracts text from a PDF file object (Streamlit UploadedFile).
    """
    text = ""
    try:
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF {file.name}: {e}")
    return text

def load_documents(files):
    """
    Loads text from uploaded files (.txt or .pdf).
    Returns a list of LangChain Document objects.
    """
    documents = []
    
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()
        content = ""
        
        try:
            if file_extension == ".pdf":
                content = get_pdf_text(file)
            elif file_extension == ".txt":
                # Decode bytes to string for text files
                content = file.getvalue().decode("utf-8")
            else:
                print(f"Unsupported file type: {file.name}")
                continue
                
            # Create a LangChain Document if content was extracted
            if content.strip():
                doc = Document(
                    page_content=content,
                    metadata={"source": file.name}
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # This splits the list of Document objects into a larger list of chunked Document objects
    chunks = text_splitter.split_documents(documents)
    return chunks