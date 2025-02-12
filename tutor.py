import os
import asyncio
import streamlit as st

from frontend.ui import setup_page_config

# Set page configuration
setup_page_config()

# def css_style():
#     with open("frontend/style.css") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# css_style()

from pipeline.doc_processor import (
    generate_embedding,
    generate_GraphRAG_embedding,
)


from pipeline.get_response import (
    tutor_agent,
)


from pipeline.utils import (
    generate_course_id,
)


from pipeline.helper.index_files_saving import (
    graphrag_index_files_check,
    graphrag_index_files_compress,
    graphrag_index_files_decompress,
    vectorrag_index_files_check,
    vectorrag_index_files_compress,
    vectorrag_index_files_decompress,
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
    process_pdf_file,
    save_file_locally,
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
            file = st.session_state.uploaded_file.read()

            # Compute hashed ID and prepare embedding folder
            file_hash = generate_course_id(file)
            course_id = file_hash
            embedding_folder = os.path.join('embedded_content', course_id)
            print(f"Embedding folder: {embedding_folder}")
            if not os.path.exists('embedded_content'):
                os.makedirs('embedded_content')
            if not os.path.exists(embedding_folder):
                os.makedirs(embedding_folder)

            # Save the file locally
            save_file_locally(file, filename=st.session_state.uploaded_file.name, embedding_folder=embedding_folder)

            # Process file and create session states for documents and PDF object
            documents, doc, file_paths = process_pdf_file(file, st.session_state.uploaded_file.name)

            if len(documents) > 50:
                st.error("File contains more than 50 pages. Please upload a shorter document.")
                st.stop()
                
            # Generate embeddings based on the selected mode
            if st.session_state.mode == "Advanced":
                with st.spinner("Processing file to generate knowledge graph, may take 3 - 5 mins..."):
                    if(graphrag_index_files_decompress(embedding_folder)):
                        print("GraphRAG index files are ready.")
                    else:
                        # Files are missing and have been cleaned up
                        save_file_locally(file, filename=st.session_state.uploaded_file.name, embedding_folder=embedding_folder)
                        generate_embedding(documents, doc, file_paths, embedding_folder=embedding_folder)
                        asyncio.run(generate_GraphRAG_embedding(documents, embedding_folder=embedding_folder))
                        if(graphrag_index_files_compress(embedding_folder)):
                            print("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
                        else:
                            # Retry once if first attempt fails
                            save_file_locally(file, filename=st.session_state.uploaded_file.name, embedding_folder=embedding_folder)
                            generate_embedding(documents, doc, file_paths, embedding_folder=embedding_folder)
                            asyncio.run(generate_GraphRAG_embedding(documents, embedding_folder=embedding_folder))
                            if(graphrag_index_files_compress(embedding_folder)):
                                print("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
                            else:
                                print("Error compressing and uploading GraphRAG index files to Azure Blob Storage.")
            else:  # Basic mode
                with st.spinner("Processing file..."):
                    if(vectorrag_index_files_decompress(embedding_folder)):
                        print("VectorRAG index files are ready.")
                    else:
                        # Files are missing and have been cleaned up
                        save_file_locally(file, filename=st.session_state.uploaded_file.name, embedding_folder=embedding_folder)
                        generate_embedding(documents, doc, file_paths, embedding_folder=embedding_folder)
                        if(vectorrag_index_files_compress(embedding_folder)):
                            print("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
                        else:
                            # Retry once if first attempt fails
                            save_file_locally(file, filename=st.session_state.uploaded_file.name, embedding_folder=embedding_folder)
                            generate_embedding(documents, doc, file_paths, embedding_folder=embedding_folder)
                            if(vectorrag_index_files_compress(embedding_folder)):
                                print("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
                            else:
                                print("Error compressing and uploading VectorRAG index files to Azure Blob Storage.")

            # Initialize state
            initialize_session_state(embedding_folder=embedding_folder)
            

            # If documents are found, proceed to show chat interface and PDF viewer
            if documents:
                outer_columns = st.columns([1, 1])

            with outer_columns[1]:
                show_chat_interface(
                    doc=doc,
                    documents=documents,
                    file_paths=file_paths,
                    embedding_folder=embedding_folder,
                    tutor_agent=tutor_agent
                )

                with outer_columns[0]:
                    show_pdf_viewer(file)

    elif __name__ == "__main__" and st.session_state.page == "📬 DeepTutor?":
        show_contact_us()
