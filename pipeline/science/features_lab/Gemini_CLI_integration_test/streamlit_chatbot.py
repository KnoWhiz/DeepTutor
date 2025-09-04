"""
Streamlit Chatbot Application with Gemini CLI Integration.

This application provides a chatbot interface that can process PDF files
and provide streaming responses using the Gemini CLI.
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import Generator

# Import our custom modules
from gemini_cli_handler import main_streaming_function
from pdf_converter import convert_uploaded_pdf

# Configure Streamlit page
st.set_page_config(
    page_title="Gemini CLI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
WORKING_DIR = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files"
SUPPORTED_FILE_TYPES = ["pdf", "txt"]


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    if "current_file_path" not in st.session_state:
        st.session_state.current_file_path = None


def display_chat_messages():
    """Display all chat messages in the main chat area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_file_upload():
    """Handle file upload functionality in the sidebar."""
    st.sidebar.header("📁 File Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=SUPPORTED_FILE_TYPES,
        help="Upload a PDF or TXT file to analyze"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    if uploaded_file.type == "application/pdf":
                        # Convert PDF to text
                        text_file_path = convert_uploaded_pdf(uploaded_file, WORKING_DIR)
                        st.session_state.current_file_path = text_file_path
                        file_type = "PDF (converted to text)"
                    else:
                        # Handle text file
                        working_dir = Path(WORKING_DIR)
                        working_dir.mkdir(parents=True, exist_ok=True)
                        
                        text_file_path = working_dir / uploaded_file.name
                        with open(text_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.session_state.current_file_path = str(text_file_path)
                        file_type = "Text file"
                    
                    # Add to uploaded files list
                    file_info = {
                        "name": uploaded_file.name,
                        "type": file_type,
                        "path": st.session_state.current_file_path,
                        "size": uploaded_file.size
                    }
                    st.session_state.uploaded_files.append(file_info)
                    
                    st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing file: {str(e)}")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.sidebar.subheader("📋 Uploaded Files")
        for i, file_info in enumerate(st.session_state.uploaded_files):
            with st.sidebar.expander(f"{file_info['name']} ({file_info['type']})"):
                st.write(f"**Size:** {file_info['size']} bytes")
                st.write(f"**Path:** {file_info['path']}")
                
                # Button to select this file for querying
                if st.button(f"Use this file", key=f"use_file_{i}"):
                    st.session_state.current_file_path = file_info['path']
                    st.sidebar.success(f"Selected: {file_info['name']}")
                
                # Button to remove this file
                if st.button(f"Remove", key=f"remove_file_{i}"):
                    try:
                        # Remove file from disk
                        Path(file_info['path']).unlink(missing_ok=True)
                        # Remove from session state
                        st.session_state.uploaded_files.pop(i)
                        st.sidebar.success(f"Removed: {file_info['name']}")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error removing file: {str(e)}")


def handle_chat_input():
    """Handle chat input and generate streaming responses."""
    if prompt := st.chat_input("Ask a question about your uploaded document..."):
        
        # Check if a file is available
        if not st.session_state.current_file_path:
            st.error("🚫 Please upload a file first before asking questions.")
            return
        
        # Check if the file still exists
        if not Path(st.session_state.current_file_path).exists():
            st.error("🚫 The selected file no longer exists. Please upload a new file.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Show spinner while initializing
                with st.spinner("🤖 Gemini is thinking..."):
                    response_generator = main_streaming_function(
                        st.session_state.current_file_path, 
                        prompt
                    )
                
                # Stream the response
                for chunk in response_generator:
                    if chunk.strip():  # Only process non-empty chunks
                        full_response += chunk + "\n"
                        # Update the message placeholder with current response
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)  # Small delay for visual streaming effect
                
                # Final update without cursor
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })
                
            except Exception as e:
                error_message = f"❌ **Error:** {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })


def display_sidebar_info():
    """Display information and controls in the sidebar."""
    st.sidebar.title("🤖 Gemini CLI Chatbot")
    
    st.sidebar.markdown("""
    ### How to use:
    1. **Upload a file** (PDF or TXT) using the file uploader
    2. **Ask questions** about the content in the chat
    3. **Get detailed responses** powered by Gemini CLI with web search
    
    ### Features:
    - 📄 PDF to text conversion
    - 🔍 Deep research with web search
    - 📚 Citation and bibliography generation
    - 💬 Streaming responses
    """)
    
    # Display current file status
    if st.session_state.current_file_path:
        current_file = Path(st.session_state.current_file_path).name
        st.sidebar.info(f"📄 **Current file:** {current_file}")
    else:
        st.sidebar.warning("⚠️ No file selected")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Clear all files button
    if st.sidebar.button("🗂️ Clear All Files"):
        # Remove all files from disk
        for file_info in st.session_state.uploaded_files:
            try:
                Path(file_info['path']).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore errors when removing files
        
        # Clear session state
        st.session_state.uploaded_files = []
        st.session_state.current_file_path = None
        st.sidebar.success("All files cleared!")
        st.rerun()


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar_info()
    handle_file_upload()
    
    # Main chat interface
    st.title("🤖 Gemini CLI Research Assistant")
    st.markdown("""
    Welcome to the Gemini CLI Research Assistant! Upload a document and ask questions to get 
    detailed research responses with web search integration and citations.
    """)
    
    # Display chat messages
    display_chat_messages()
    
    # Handle chat input
    handle_chat_input()


if __name__ == "__main__":
    main() 