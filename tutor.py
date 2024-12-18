import os
import base64
import fitz
import asyncio
import io
import json
import streamlit as st

from pipeline.get_response import (
    generate_embedding,
    generate_GraphRAG_embedding,
    get_response,
    get_response_source,
    regen_response,
)


from pipeline.utils import (
    generate_course_id,
    extract_documents_from_file
)


from frontend.ui import (
    setup_page_config,
    show_header,
    show_file_upload,
    show_mode_option,
    show_page_option,
    show_chat_interface,
    show_pdf_viewer,
    show_footer,
    show_contact_us
)


from frontend.state import (
    initialize_session_state,
    handle_file_change,
    set_response_mode,
    process_pdf_file,
    save_file_locally
)


# Set up basic page configuration and header
setup_page_config()
show_header()


# Initialize state
initialize_session_state()


# Show file uploader and response mode options in the sidebar
uploaded_file = show_file_upload(on_change=handle_file_change)
show_mode_option(uploaded_file)
show_page_option()
show_footer()


if __name__ == "__main__" and uploaded_file is not None and st.session_state.page == "📑 Document reading":
    file_size = uploaded_file.size
    max_file_size = 10 * 1024 * 1024  # 10 MB

    if file_size > max_file_size:
        st.error("File size exceeds the 10 MB limit. Please upload a smaller file.")
    else:
        file = uploaded_file.read()
        save_file_locally(file)

        # Compute hashed ID and prepare embedding folder
        file_hash = generate_course_id(file)
        course_id = file_hash
        embedding_folder = os.path.join('embedded_content', course_id)
        if not os.path.exists('embedded_content'):
            os.makedirs('embedded_content')
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)

        # Process file and create session states for documents and PDF object
        documents, doc = process_pdf_file(file, uploaded_file.name)

        # Generate embeddings based on the selected mode
        if st.session_state.mode == "Professor":
            with st.spinner("Processing file, may take 3 - 5 mins..."):
                generate_embedding(documents, embedding_folder=embedding_folder)
                generate_GraphRAG_embedding(documents, embedding_folder=embedding_folder)
        else:
            with st.spinner("Processing file..."):
                generate_embedding(documents, embedding_folder=embedding_folder)

        # If documents are found, proceed to show chat interface and PDF viewer
        if documents:
            outer_columns = st.columns([1, 1])

            with outer_columns[1]:
                show_chat_interface(
                    doc=doc,
                    documents=documents,
                    embedding_folder=embedding_folder,
                    get_response_fn=get_response,
                    get_source_fn=get_response_source,
                    regen_response=regen_response,
                )

            with outer_columns[0]:
                show_pdf_viewer(file)

elif __name__ == "__main__" and st.session_state.page == "📬 KnoWhiz?":
    show_contact_us()