import os
import fitz
import io
import json
import hashlib
import streamlit as st

from dotenv import load_dotenv
from pipeline.utils import extract_documents_from_file
from pipeline.chat_history_manager import (
    create_session_id,
    save_chat_history,
    load_chat_history,
    delete_chat_history,
    cleanup_old_sessions
)
from pipeline.session_manager import ChatSession, ChatMode


load_dotenv()
# Control whether to skip authentication for local or staging environment.
SKIP_AUTH = True if os.getenv("ENVIRONMENT") == "local" or os.getenv("ENVIRONMENT") == "staging" else False
print(f"SKIP_AUTH: {SKIP_AUTH}")


# Function to initialize the session state
def initialize_session_state(embedding_folder=None):
    """Initialize all session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = create_session_id()
    
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = ChatSession(
            session_id=st.session_state.session_id
        )
        st.session_state.chat_session.initialize()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = st.session_state.chat_session.chat_history
    
    if 'mode' not in st.session_state:
        st.session_state.mode = "Basic"
        st.session_state.chat_session.set_mode(ChatMode.BASIC)
    elif st.session_state.mode == "Advanced":
        st.session_state.chat_session.set_mode(ChatMode.ADVANCED)
    else:
        st.session_state.chat_session.set_mode(ChatMode.BASIC)
    
    if 'language' not in st.session_state:
        st.session_state.language = "English"
        st.session_state.chat_session.set_language("English")
    elif st.session_state.language:
        st.session_state.chat_session.set_language(st.session_state.language)
    
    if 'show_chat_border' not in st.session_state:
        st.session_state.show_chat_border = True
    
    if 'page' not in st.session_state:
        st.session_state.page = "📑 Document reading"
    
    if embedding_folder:
        st.session_state.embedding_folder = embedding_folder


# Function to handle file change
def handle_file_change():
    """Handle changes when a new file is uploaded."""
    # Clear existing chat history
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []
    
    if 'chat_session' in st.session_state:
        st.session_state.chat_session.clear_history()
    
    # Reset session ID
    st.session_state.session_id = create_session_id()
    
    # Create new chat session
    st.session_state.chat_session = ChatSession(
        session_id=st.session_state.session_id
    )
    st.session_state.chat_session.initialize()
    
    # Update mode
    if 'mode' in st.session_state:
        st.session_state.chat_session.set_mode(
            ChatMode.ADVANCED if st.session_state.mode == "Advanced" else ChatMode.BASIC
        )
    
    # Update language
    if 'language' in st.session_state:
        st.session_state.chat_session.set_language(st.session_state.language)
    
    # Clear other session state variables
    keys_to_keep = {
        'session_id', 'chat_session', 'chat_history', 'mode', 'language',
        'is_uploaded_file', 'uploaded_file', 'page'
    }
    
    # Clear all other session state variables
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    
    # Reset document-related variables
    st.session_state.documents = None
    st.session_state.doc = None
    st.session_state.file_paths = None
    st.session_state.total_pages = 0
    st.session_state.current_page = 1
    st.session_state.annotations = []
    st.session_state.chat_occurred = False
    st.session_state.sources = {}


# Function to process the PDF file
def process_pdf_file(file, filename):
    """Process uploaded PDF file and store all necessary information in session state.
    
    Args:
        file: The uploaded PDF file content
        filename: Name of the uploaded file
        
    Returns:
        tuple: (documents, doc, file_paths) where:
            - documents: Contains chunked/split text content optimized for embedding and retrieval
            - doc: Contains the complete PDF structure including pages, formatting, and visual elements
            - file_paths: List of file paths for the processed documents
    """
    # Process the documents
    documents, file_paths = extract_documents_from_file(file, filename)
    doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")
    
    # Store everything in session state
    st.session_state.documents = documents
    st.session_state.doc = doc
    st.session_state.file_paths = file_paths
    st.session_state.total_pages = len(doc)
    st.session_state.current_page = 1  # Initialize current page
    st.session_state.annotations = []  # Initialize annotations
    st.session_state.chat_occurred = False  # Initialize chat state
    st.session_state.sources = {}  # Initialize sources
    
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

    # Generate a shorter filename using hash, and it should be unique and consistent for the same file
    base_name = os.path.splitext(filename)[0]
    hashed_name = hashlib.md5(file).hexdigest()[:8]  # Use first 8 chars of hash
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
