import os
import streamlit as st
import fitz
import io
from pipeline.utils import extract_documents_from_file


# Function to initialize the session state
def initialize_session_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "Normal"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# Function to handle file change
def handle_file_change():
    # Reset states when a new file is uploaded
    st.session_state.chat_history = []
    st.session_state.annotations = []
    st.session_state.chat_occurred = False
    st.session_state.sources = []
    st.session_state.total_pages = 1
    st.session_state.current_page = 1


# Function to set the response mode
def set_response_mode(mode):
    st.session_state.mode = mode


# Function to save the file locally
def save_file_locally(file):
    temp_folder = './input_files/'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    # Clear the temp folder
    for f in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    # (Optionally save the file if needed)
    # with open(os.path.join(temp_folder, "temp.pdf"), "wb") as f:
    #     f.write(file)


# Function to process the PDF file
def process_pdf_file(file, filename):
    documents = extract_documents_from_file(file, filename)
    doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")
    st.session_state.doc = doc
    st.session_state.total_pages = len(doc)
    return documents, doc
