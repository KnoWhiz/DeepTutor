import os
import base64
import fitz
import shutil
import asyncio
import tempfile
import hashlib
import io
import json
import pprint
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import streamlit_nested_layout
from streamlit_float import *

from pipeline.get_response import generate_embedding
from pipeline.get_response import generate_GraphRAG_embedding
from pipeline.get_response import get_response
from pipeline.get_response import get_response_source
from pipeline.get_response import regen_with_graphrag
from pipeline.get_response import regen_with_longer_context
from pipeline.images_understanding import get_relevant_images, display_relevant_images, extract_images_with_context, save_images_temp


# Set page config
st.set_page_config(
    page_title="KnoWhiz Tutor",
    page_icon="frontend/images/logo_short.ico",  # Replace with the actual path to your .ico file
    layout="wide"
)

# Main content
with open("frontend/images/logo_short.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()
st.markdown(
    f"""
    <h2 style='text-align: center;'>
        <img src="data:image/png;base64,{encoded_image}" alt='icon' style='width:50px; height:50px; vertical-align: middle; margin-right: 10px;'>
        KnoWhiz Tutor
    </h2>
    """,
    unsafe_allow_html=True
)
st.subheader("Upload a document to get started.")

# Init float function for chat_input textbox
float_init(theme=True, include_unstable_primary=False)

# Custom function to extract document objects from uploaded file
def extract_documents_from_file(file, filename):
    input_dir = './input_files/'

    # Create the input_files directory if it doesn't exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # Clean up the input_files directory
    for existing_file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, existing_file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Save the file to the input_files directory with the original filename
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(file)

    # Load the document
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

def find_pages_with_excerpts(doc, excerpts):
    pages_with_excerpts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                pages_with_excerpts.append(page_num)
                break
    return pages_with_excerpts if pages_with_excerpts else [0]

def get_highlight_info(doc, excerpts):
    annotations = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                for inst in text_instances:
                    annotations.append(
                        {
                            "page": page_num + 1,
                            "x": inst.x0,
                            "y": inst.y0,
                            "width": inst.x1 - inst.x0,
                            "height": inst.y1 - inst.y0,
                            "color": "red",
                        }
                    )
    return annotations

def previous_page():
    if st.session_state.current_page > 1:
        st.session_state.current_page -= 1

def next_page():
    if st.session_state.current_page < st.session_state.total_pages:
        st.session_state.current_page += 1

def close_pdf():
    st.session_state.show_pdf = False

# Reset all states
def file_changed():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def chat_content():
    st.session_state.chat_history.append(
        {"role": "user", "content": st.session_state.user_input}
    )

learner_avatar = "frontend/images/learner.svg"
tutor_avatar = "frontend/images/tutor.svg"

# Layout for file uploader and mode option
file_col, option_col = st.columns([3, 1])
with file_col:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", on_change=file_changed)
with option_col:
    if uploaded_file is None:
        st.session_state.mode = st.radio("Response mode", options=["Normal", "GraphRAG", "Long context"], index=0, disabled=False)
    else:
        st.session_state.mode = st.radio("Response mode", options=["Normal", "GraphRAG", "Long context"], index=0, disabled=True)

if __name__ == "__main__" and uploaded_file is not None:
    file_size = uploaded_file.size
    max_file_size = 10 * 1024 * 1024  # 10 MB

    if file_size > max_file_size:
        st.error("File size exceeds the 10 MB limit. Please upload a smaller file.")
    else:
        file = uploaded_file.read()
        # Clear the temp file folder and save the new upload file to the folder
        temp_folder = './input_files/'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        for f in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # Compute a hashed ID based on the PDF content
        file_hash = hashlib.md5(file).hexdigest()
        course_id = file_hash
        embedding_folder = os.path.join('embedded_content', course_id)
        if not os.path.exists('embedded_content'):
            os.makedirs('embedded_content')
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)

        with st.spinner("Processing file..."):
            documents = extract_documents_from_file(file, uploaded_file.name)
            st.session_state.doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")
            st.session_state.total_pages = len(st.session_state.doc)
            generate_embedding(documents, embedding_folder=embedding_folder)
            if st.session_state.mode == "GraphRAG":
                generate_GraphRAG_embedding(documents, embedding_folder=embedding_folder)

        if documents:
            qa_chain = get_response(documents, embedding_folder=embedding_folder)
            qa_source_chain = get_response_source(documents, embedding_folder=embedding_folder)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    {"role": "assistant", "content": "Hello! How can I assist you today? "}
                ]
                st.session_state.show_chat_border = False
            else:
                st.session_state.show_chat_border = True

        outer_columns = st.columns([1,1])

        with outer_columns[1]:
            with st.container(border=st.session_state.show_chat_border, height=800):
                with st.container():
                    st.chat_input(key='user_input', on_submit=chat_content)
                    button_b_pos = "2.2rem"
                    button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
                    float_parent(css=button_css)

                # Display chat history
                for idx, msg in enumerate(st.session_state.chat_history):
                    avatar = learner_avatar if msg["role"] == "user" else tutor_avatar
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(msg["content"])
                        # If this message is from the assistant
                        if msg["role"] == "assistant":
                            # Show appropriate button based on mode
                            if st.session_state.mode == "GraphRAG":
                                st.button(
                                    "Regen with GraphRAG",
                                    key=f"regen_graphrag_{idx}",
                                    on_click=regen_with_graphrag
                                )
                            elif st.session_state.mode == "Long context":
                                st.button(
                                    "Regen with longer context",
                                    key=f"regen_longer_context_{idx}",
                                    on_click=regen_with_longer_context
                                )

                # If new user input
                if user_input := st.session_state.get('user_input', None):
                    with st.spinner("Generating response..."):
                        try:
                            parsed_result = qa_chain.invoke({"input": user_input})
                            answer = parsed_result['answer']

                            # Get sources
                            parsed_result = qa_source_chain.invoke({"input": user_input})
                            sources = parsed_result['answer']['sources']
                            try:
                                if not all(isinstance(source, str) for source in sources):
                                    raise ValueError("Sources must be a list of strings.")
                                sources = list(sources)
                            except:
                                sources = []

                            print("The content is from: ", sources)

                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer}
                            )
                            with st.chat_message("assistant", avatar=tutor_avatar):
                                st.write(answer)
                                # Show button based on mode for the newly generated response
                                if st.session_state.mode == "GraphRAG":
                                    st.button(
                                        "Regen with GraphRAG",
                                        key=f"regen_graphrag_new_{len(st.session_state.chat_history)}",
                                        on_click=regen_with_graphrag
                                    )
                                elif st.session_state.mode == "Long context":
                                    st.button(
                                        "Regen with longer context",
                                        key=f"regen_longer_context_new_{len(st.session_state.chat_history)}",
                                        on_click=regen_with_longer_context
                                    )

                            st.session_state.sources = sources
                            st.session_state.chat_occurred = True

                        except json.JSONDecodeError:
                            st.error(
                                "There was an error parsing the response. Please try again."
                            )

                    # Highlight PDF excerpts
                    if file and st.session_state.get("chat_occurred", False):
                        doc = st.session_state.doc
                        pages_with_excerpts = find_pages_with_excerpts(doc, st.session_state.sources)
                        if "current_page" not in st.session_state:
                            st.session_state.current_page = pages_with_excerpts[0] + 1
                        if 'pages_with_exerpts' not in st.session_state:
                            st.session_state.pages_with_excerpts = pages_with_excerpts
                        st.session_state.annotations = get_highlight_info(doc, st.session_state.sources)
                        if st.session_state.annotations:
                            st.session_state.current_page = min(
                                annotation["page"] for annotation in st.session_state.annotations
                            )

        with outer_columns[0]:
            if "current_page" not in st.session_state:
                st.session_state.current_page = 1
            if "annotations" not in st.session_state:
                st.session_state.annotations = []
            pdf_viewer(
                file,
                width=700,
                height=800,
                annotations=st.session_state.annotations,
                pages_to_render=[st.session_state.current_page],
                render_text=True,
            )
            col1, col2, col3, col4 = st.columns([8, 4, 3, 3],vertical_alignment='center')
            with col1:
                st.button("←", on_click=previous_page)
            with col2:
                st.write(
                    f"Page {st.session_state.current_page} of {st.session_state.total_pages}"
                )
            with col4:
                st.button("→", on_click=next_page)
