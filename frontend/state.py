import os
import streamlit as st
import fitz
import io
from pipeline.utils import extract_documents_from_file
import hashlib
import json

# Control whether to skip authentication for local testing
SKIP_AUTH = True if os.getenv("ENVIRONMENT") == "local" else False


# Function to initialize the session state
def initialize_session_state(embedding_folder):
    """Initialize session state variables"""
    if "mode" not in st.session_state:
        st.session_state.mode = "Basic"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.show_chat_border = False
    else:
        st.session_state.show_chat_border = True


# Function to handle file change
def handle_file_change():
    # Reset all relevant session states when a new file is uploaded
    if 'chat_history' in st.session_state:
        del st.session_state.chat_history
    if 'annotations' in st.session_state:
        del st.session_state.annotations
    if 'chat_occurred' in st.session_state:
        del st.session_state.chat_occurred
    if 'sources' in st.session_state:
        del st.session_state.sources
    if 'total_pages' in st.session_state:
        del st.session_state.total_pages
    if 'current_page' in st.session_state:
        del st.session_state.current_page
    if 'doc' in st.session_state:
        del st.session_state.doc
    
    # Clear Streamlit cache
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Reset uploaded file state
    st.session_state.is_uploaded_file = True


# Function to process the PDF file
def process_pdf_file(file, filename):
    """
    documents: Contains chunked/split text content optimized for embedding and retrieval
    doc: Contains the complete PDF structure including pages, formatting, and visual elements
    """
    documents, file_paths = extract_documents_from_file(file, filename)
    doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")
    st.session_state.doc = doc
    st.session_state.total_pages = len(doc)
    return documents, doc, file_paths


# Function to save the file locally as a text file
def save_file_locally(file, filename, embedding_folder):
    """
    Save the file (e.g., PDF) loaded as text into the GraphRAG_embedding_input_folder.
    """
    # Define folder structure
    GraphRAG_embedding_folder = os.path.join(embedding_folder, "GraphRAG")
    GraphRAG_embedding_input_folder = os.path.join(GraphRAG_embedding_folder, "input")

    # Create folders if they do not exist
    os.makedirs(GraphRAG_embedding_input_folder, exist_ok=True)

    # Generate a shorter filename using hash
    base_name = os.path.splitext(filename)[0]
    hashed_name = hashlib.md5(base_name.encode()).hexdigest()[:8]  # Use first 8 chars of hash
    output_file_path = os.path.join(GraphRAG_embedding_input_folder, f"{hashed_name}.txt")

    # Extract text from the PDF using the provided utility function
    documents, file_paths = extract_documents_from_file(file, filename)

    # Write the extracted text into a .txt file
    # If the file does not exist, it will be created
    if os.path.exists(output_file_path):
        print(f"File already exists: {output_file_path}")
        return
    
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for doc in documents:
                # Each doc is expected to have a `page_content` attribute if it's a Document object
                if hasattr(doc, 'page_content') and doc.page_content:
                    # Write the text, followed by a newline for clarity
                    f.write(doc.page_content.strip() + "\n")
        print(f"Text successfully saved to: {output_file_path}")
    except OSError as e:
        print(f"Error saving file: {e}")
        # Create a mapping file to track original filenames if needed
        mapping_file = os.path.join(GraphRAG_embedding_folder, "filename_mapping.json")
        try:
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
            else:
                mapping = {}
            mapping[hashed_name] = base_name
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
        except Exception as e:
            print(f"Error saving filename mapping: {e}")
        raise
    return
