import os
import pypdf
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text):
    """
    Cleans the extracted text:
    1. Replaces newlines with spaces (fixes broken sentences).
    2. Removes multiple spaces (fixes formatting issues).
    3. Removes very short lines that are likely page numbers or headers.
    """
    # Replace newlines with spaces to keep sentences together
    text = text.replace('\n', ' ')
    
    # Remove multiple spaces and strip
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_pdf_text(file):
    text = ""
    try:
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        print(f"Error reading PDF {file.name}: {e}")
    return text

def load_documents(files):
    documents = []
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()
        content = ""
        
        try:
            if file_extension == ".pdf":
                content = get_pdf_text(file)
            elif file_extension == ".txt":
                content = file.getvalue().decode("utf-8")
            
            # Apply the cleaning function here
            content = clean_text(content)

            if content:
                doc = Document(
                    page_content=content,
                    metadata={"source": file.name}
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # This priority list keeps paragraphs and sentences together
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    return chunks