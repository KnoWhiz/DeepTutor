import os
# -----------------------------------------------------------
# Prevent duplicate OpenMP runtime crash on macOS.
# When multiple Python wheels bundle their own `libomp.dylib` (e.g.,
# `numpy` + `onnxruntime`), the OpenMP runtime aborts with:
#   "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized."
# Setting this env variable before any OpenMP‑linked library is imported
# instructs `libomp` to allow duplication and continue execution.
# -----------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import streamlit as st
from pipeline.science.pipeline.logging_config import setup_logging

# Set up logging configuration
setup_logging()
logger = logging.getLogger("tutor.py")

from frontend.ui import setup_page_config

# Set page configuration
setup_page_config()

from pipeline.science.pipeline.utils import generate_file_id

from frontend.ui import (
    show_auth_top,
    show_header,
    show_file_upload,
    show_mode_option,
    show_language_option,
    show_page_option,
    show_chat_interface,
    show_pdf_viewer,
    show_footer,
    show_contact_us,
)

from frontend.state import (
    handle_file_change,
    initialize_session_state,
    state_process_pdf_file,
)

from frontend.auth import show_auth

from frontend.state import SKIP_AUTH


SERVER_AGENT_PLACEHOLDER_PDF = (
    "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/"
    "features_lab/(Benchmarks and evals, safety vs. capabilities, machine ethics) "
    "DecodingTrust A Comprehensive Assessment of Trustworthiness in GPT Models.pdf"
)


if 'isAuth' not in st.session_state:
    st.session_state['isAuth'] = SKIP_AUTH

if not SKIP_AUTH:
    show_auth()

if st.session_state['isAuth']:
    # Set up basic page configuration and header
    show_auth_top()
    show_header()

    # Show mode selection and file uploader in the sidebar
    if 'is_uploaded_file' not in st.session_state:
        st.session_state['is_uploaded_file'] = False

    # Move mode selection before file upload
    show_mode_option()
    show_file_upload(on_change=handle_file_change)
    show_language_option()
    show_page_option()
    show_footer()

    if __name__ == "__main__" and st.session_state.uploaded_file is not None and st.session_state.page == "📑 Document reading":
        # Check if we're in Lite mode with multiple files
        is_lite_mode = st.session_state.get('mode', 'Basic') == "Lite"
        is_multiple_files = is_lite_mode and isinstance(st.session_state.uploaded_file, list)
        
        # Process single file or multiple files based on mode
        if is_multiple_files:
            # Handle multiple file upload (Lite mode only)
            uploaded_files = st.session_state.uploaded_file
            
            # Check if any files were actually uploaded
            if not uploaded_files or len(uploaded_files) == 0:
                st.info("Please upload at least one PDF file.")
                st.stop()
            
            # Check if any file exceeds the size limit
            max_file_size = 50 * 1024 * 1024  # 50 MB
            for uploaded_file in uploaded_files:
                if uploaded_file.size > max_file_size:
                    st.error(f"File '{uploaded_file.name}' exceeds the 50 MB limit. Please upload smaller files.")
                    st.stop()
            
            # Set up directory for saving files
            path_prefix = os.getenv("FILE_PATH_PREFIX")
            input_dir = os.path.join(path_prefix, 'input_files')
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)
            
            # Save all files and build the file_path_list
            file_path_list = []
            for idx, uploaded_file in enumerate(uploaded_files):
                file = uploaded_file.read()
                file_path = os.path.join(input_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(file)
                file_path_list.append(file_path)
            
            # Only process the first file for preview
            document, doc = state_process_pdf_file(file_path_list[0])
            
            path_prefix = os.getenv("FILE_PATH_PREFIX")
            embedded_content_path = os.path.join(path_prefix, 'embedded_content')
            
            # Create a list of embedding folders
            embedding_folder_list = []
            for file_path in file_path_list:
                embedding_folder = os.path.join(embedded_content_path, generate_file_id(file_path))
                embedding_folder_list.append(embedding_folder)
            
            # Use the first embedding folder for initialization
            embedding_folder = embedding_folder_list[0]
            
            # Check if first document exceeds page limit
            if len(document) > 1200:
                st.error("File contains more than 1200 pages. Please upload a shorter document.")
                st.stop()
            
            # Initialize state with the main embedding folder
            initialize_session_state(embedding_folder=embedding_folder)
            
            # Store the file_path_list in session state for use in chat interface
            st.session_state.file_path_list = file_path_list
            
            # If document is found, proceed to show chat interface and PDF viewer (first file only)
            if document:
                outer_columns = st.columns([1, 1])
                
                if len(st.session_state.chat_session.chat_history) == 0:
                    with outer_columns[0]:
                        # Pass the entire file_path_list to show_pdf_viewer instead of just the first file
                        show_pdf_viewer(file_path_list)
                        
                    with outer_columns[1]:
                        show_chat_interface(
                            doc=doc,
                            document=document,
                            file_path=file_path_list,
                            embedding_folder=embedding_folder,
                        )
                else:
                    with outer_columns[1]:
                        show_chat_interface(
                            doc=doc,
                            document=document,
                            file_path=file_path_list,
                            embedding_folder=embedding_folder,
                        )
                    with outer_columns[0]:
                        show_pdf_viewer(file_path_list)
                
        else:
            # Original single file handling for Basic/Advanced/Server Agent modes
            uploaded_file = st.session_state.uploaded_file
            if uploaded_file is None:
                st.info("Please upload a file to get started.")
                st.stop()

            file_size = uploaded_file.size
            max_file_size = 50 * 1024 * 1024  # 50 MB
            if file_size > max_file_size:
                st.error("File size exceeds the 50 MB limit. Please upload a smaller file.")
            else:
                file_bytes = uploaded_file.read()
                path_prefix = os.getenv("FILE_PATH_PREFIX")
                input_dir = os.path.join(path_prefix, 'input_files')
                if not os.path.exists(input_dir):
                    os.makedirs(input_dir)

                saved_file_path = os.path.join(input_dir, uploaded_file.name)
                with open(saved_file_path, 'wb') as f:
                    f.write(file_bytes)

                current_mode = st.session_state.get('mode', 'Basic')

                if current_mode == "Server Agent Basic":
                    initialize_session_state()

                    document, doc = state_process_pdf_file(SERVER_AGENT_PLACEHOLDER_PDF)
                    st.session_state.file_path_list = [saved_file_path]
                    placeholder_path = SERVER_AGENT_PLACEHOLDER_PDF

                    if document:
                        outer_columns = st.columns([1, 1])

                        if len(st.session_state.chat_session.chat_history) == 0:
                            with outer_columns[0]:
                                show_pdf_viewer(placeholder_path)

                            with outer_columns[1]:
                                show_chat_interface(
                                    doc=doc,
                                    document=document,
                                    file_path=saved_file_path,
                                    embedding_folder=None,
                                )
                        else:
                            with outer_columns[1]:
                                show_chat_interface(
                                    doc=doc,
                                    document=document,
                                    file_path=saved_file_path,
                                    embedding_folder=None,
                                )
                            with outer_columns[0]:
                                show_pdf_viewer(placeholder_path)

                else:
                    document, doc = state_process_pdf_file(saved_file_path)
                    path_prefix = os.getenv("FILE_PATH_PREFIX")
                    embedded_content_path = os.path.join(path_prefix, 'embedded_content')
                    embedding_folder = os.path.join(embedded_content_path, generate_file_id(saved_file_path))

                    if len(document) > 1200:
                        st.error("File contains more than 1200 pages. Please upload a shorter document.")
                        st.stop()

                    initialize_session_state(embedding_folder=embedding_folder)
                    st.session_state.file_path_list = [saved_file_path]

                    if document:
                        outer_columns = st.columns([1, 1])

                    if len(st.session_state.chat_session.chat_history) == 0:
                        with outer_columns[0]:
                            show_pdf_viewer(saved_file_path)

                        with outer_columns[1]:
                            show_chat_interface(
                                doc=doc,
                                document=document,
                                file_path=saved_file_path,
                                embedding_folder=embedding_folder,
                            )
                    else:
                        with outer_columns[1]:
                            show_chat_interface(
                                doc=doc,
                                document=document,
                                file_path=saved_file_path,
                                embedding_folder=embedding_folder,
                            )
                        with outer_columns[0]:
                            show_pdf_viewer(saved_file_path)

            logger.info(f"st.session_state.current_page is {st.session_state.current_page}")

    elif __name__ == "__main__" and st.session_state.page == "📬 DeepTutor?":
        show_contact_us()
