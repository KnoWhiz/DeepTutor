import os
import streamlit as st
import fitz
import io
from pipeline.utils import extract_documents_from_file


# Function to initialize the session state
def initialize_session_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "TA"
    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! How can I assist you today? "}
        ]
        st.session_state.show_chat_border = False
    else:
        st.session_state.show_chat_border = True


# Function to handle file change
def handle_file_change():
    # Reset states when a new file is uploaded
    st.session_state.annotations = []
    st.session_state.chat_occurred = False
    st.session_state.sources = []
    st.session_state.total_pages = 1
    st.session_state.current_page = 1
    st.session_state.uploaded_file = True
    initialize_session_state()


# Function to process the PDF file
def process_pdf_file(file, filename):
    documents = extract_documents_from_file(file, filename)
    doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")
    st.session_state.doc = doc
    st.session_state.total_pages = len(doc)
    return documents, doc


# Function to save the file locally as a text file
@st.cache_resource
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
        return
    with open(output_file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            # Each doc is expected to have a `page_content` attribute if it's a Document object
            if hasattr(doc, 'page_content') and doc.page_content:
                # Write the text, followed by a newline for clarity
                f.write(doc.page_content.strip() + "\n")

    print(f"Text successfully saved to: {output_file_path}")
    return