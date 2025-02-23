import os
import streamlit as st
from pipeline.science.pipeline.logging_config import setup_logging

# Set up logging configuration
setup_logging()

from frontend.ui import setup_page_config

# Set page configuration
setup_page_config()

from pipeline.science.pipeline.tutor_agent import (
    tutor_agent,
)


from pipeline.science.pipeline.utils import (
    generate_course_id,
)

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


from frontend.auth import (
    show_auth,
)


from frontend.state import SKIP_AUTH


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
        file_size = st.session_state.uploaded_file.size
        max_file_size = 50 * 1024 * 1024  # 50 MB
        if file_size > max_file_size:
            st.error("File size exceeds the 50 MB limit. Please upload a smaller file.")
        else:
            # Save the file locally
            file = st.session_state.uploaded_file.read()
            input_dir = './input_files/'
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)
            file_path = os.path.join(input_dir, st.session_state.uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(file)

            document, doc = state_process_pdf_file(file_path)
            embedding_folder = os.path.join('embedded_content', generate_course_id(file_path))

            if len(document) > 200:
                st.error("File contains more than 200 pages. Please upload a shorter document.")
                st.stop()

            # Initialize state
            initialize_session_state(embedding_folder=embedding_folder)

            # If document are found, proceed to show chat interface and PDF viewer
            if document:
                outer_columns = st.columns([1, 1])

            with outer_columns[1]:
                show_chat_interface(
                    doc=doc,
                    document=document,
                    file_path=file_path,
                    embedding_folder=embedding_folder,
                )

                with outer_columns[0]:
                    show_pdf_viewer(file)

    elif __name__ == "__main__" and st.session_state.page == "📬 DeepTutor?":
        show_contact_us()
