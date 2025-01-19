import os
import streamlit as st
import fitz
import io
from pipeline.utils import extract_documents_from_file

# Control whether to skip authentication for local testing
SKIP_AUTH = True if os.getenv("ENVIRONMENT") == "local" else False


# Function to initialize the session state
def initialize_session_state(embedding_folder):
    if "mode" not in st.session_state:
        st.session_state.mode = "TA"
    
    if "language" not in st.session_state:
        st.session_state.language = "English"
    
    # Always try to load the latest document summary for the current document
    try:
        documents_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
        with open(documents_summary_path, "r") as f:
            initial_message = f.read()
    except FileNotFoundError:
        initial_message = "Hello! How can I assist you today?"
    
    # Reset chat history with new document summary
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": initial_message}
        ]
        st.session_state.show_chat_border = False
    else:
        # If chat history exists but file changed, reset it with new summary
        if len(st.session_state.chat_history) == 0 or st.session_state.chat_history[0]["content"] != initial_message:
            st.session_state.chat_history = [
                {"role": "assistant", "content": initial_message}
            ]
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
    documents = extract_documents_from_file(file, filename)
    doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")
    st.session_state.doc = doc
    st.session_state.total_pages = len(doc)
    return documents, doc


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

    # Determine the output text file path
    base_name = os.path.splitext(filename)[0]
    output_file_path = os.path.join(GraphRAG_embedding_input_folder, f"{base_name}.txt")

    # Extract text from the PDF using the provided utility function
    documents = extract_documents_from_file(file, filename)

    # Write the extracted text into a .txt file
    # If the file does not exist, it will be created
    if os.path.exists(output_file_path):
        print(f"File already exists: {output_file_path}")
        return
    with open(output_file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            # Each doc is expected to have a `page_content` attribute if it's a Document object
            if hasattr(doc, 'page_content') and doc.page_content:
                # Write the text, followed by a newline for clarity
                f.write(doc.page_content.strip() + "\n")

    print(f"Text successfully saved to: {output_file_path}")
    return
