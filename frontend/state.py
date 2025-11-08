import os
import fitz
import io
import streamlit as st

from dotenv import load_dotenv
from pipeline.science.pipeline.doc_processor import extract_document_from_file
from pipeline.science.pipeline.chat_history_manager import create_session_id
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode

import logging
logger = logging.getLogger("tutorfrontend.state")


load_dotenv()
# Control whether to skip authentication for local or staging environment.
SKIP_AUTH = True if os.getenv("ENVIRONMENT") == "local" or os.getenv("ENVIRONMENT") == "staging" else False
logger.info(f"SKIP_AUTH: {SKIP_AUTH}")


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
    
    if 'mode' not in st.session_state:
        st.session_state.mode = "Basic"

    if st.session_state.mode == "Advanced":
        st.session_state.chat_session.set_mode(ChatMode.ADVANCED)
    elif st.session_state.mode == "Lite":
        st.session_state.chat_session.set_mode(ChatMode.LITE)
    elif st.session_state.mode == "Server Agent Basic":
        st.session_state.chat_session.set_mode(ChatMode.SERVER_AGENT_BASIC)
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
        if st.session_state.mode == "Advanced":
            st.session_state.chat_session.set_mode(ChatMode.ADVANCED)
        elif st.session_state.mode == "Lite":
            st.session_state.chat_session.set_mode(ChatMode.LITE)
        elif st.session_state.mode == "Server Agent Basic":
            st.session_state.chat_session.set_mode(ChatMode.SERVER_AGENT_BASIC)
        else:
            st.session_state.chat_session.set_mode(ChatMode.BASIC)
    
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
    st.session_state.document = None
    st.session_state.doc = None
    st.session_state.file_path = None
    st.session_state.file_path_list = None
    st.session_state.annotations = []
    st.session_state.chat_occurred = False
    st.session_state.sources = {}


# Function to process the PDF file
def state_process_pdf_file(file_path):
    """Process uploaded PDF file and store all necessary information in session state.
    
    Args:
        file: The uploaded PDF file content
        filename: Name of the uploaded file
        
    Returns:
        tuple: (document, doc, file_path) where:
            - document: Contains chunked/split text content optimized for embedding and retrieval
            - doc: Contains the complete PDF structure including pages, formatting, and visual elements
            - file_path: List of file paths for the processed document
    """
    # Process the document
    file = open(file_path, 'rb')
    with open(file_path, 'rb') as file:
        file_bytes = file.read()  # Read the file content as bytes
        document = extract_document_from_file(file_path)
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    
    # Store everything in session state
    st.session_state.document = document
    st.session_state.doc = doc
    st.session_state.file_path = file_path
    st.session_state.file_path_list = [file_path]  # Initialize file_path_list as a list with the current file
    st.session_state.total_pages = len(doc)
    st.session_state.annotations = []  # Initialize annotations
    st.session_state.chat_occurred = False  # Initialize chat state
    st.session_state.sources = {}  # Initialize sources
    
    return document, doc
