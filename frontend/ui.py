import json
import base64
import asyncio
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import float_init, float_parent, float_css_helper

from frontend.utils import previous_page, next_page, close_pdf, chat_content
from pipeline.utils import find_pages_with_excerpts, get_highlight_info


# Function to set up the page configuration
def setup_page_config():
    st.set_page_config(
        page_title="KnoWhiz Tutor",
        page_icon="frontend/images/logo_short.ico",
        layout="wide"
    )


# Function to display the header
def show_header():
    with st.sidebar:
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


# Function to display the file uploader
def show_file_upload(on_change=None):
    with st.sidebar:
        return st.file_uploader("Choose a PDF file", type="pdf", on_change=on_change)


# Function to display the response mode options
def show_mode_option(uploaded_file):
    with st.sidebar:
        disabled = uploaded_file is not None
        mode_index = 0
        st.session_state.mode = st.radio("Response mode", options=["Normal", "GraphRAG", "Long context"], index=mode_index, disabled=disabled)


# Function to display the chat interface
def show_chat_interface(doc, documents, embedding_folder, get_response_fn, get_source_fn, regen_with_graphrag, regen_with_longer_context):
    # Init float function for chat_input textbox
    float_init(theme=True, include_unstable_primary=False)
    learner_avatar = "frontend/images/learner.svg"
    tutor_avatar = "frontend/images/tutor.svg"

    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hi how can I help you today!"}
        ]
        st.session_state.show_chat_border = False
    else:
        st.session_state.show_chat_border = True

    # outer_columns = st.columns([1,1])

    # with outer_columns[1]:
    with st.container(border=st.session_state.show_chat_border, height=800):
        with st.container():
            st.chat_input(key='user_input', on_submit=chat_content)
            button_b_pos = "2.2rem"
            button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
            float_parent(css=button_css)

        # Display existing chat history
        for idx, msg in enumerate(st.session_state.chat_history):
            avatar = learner_avatar if msg["role"] == "user" else tutor_avatar
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])
                if msg["role"] == "assistant":
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

        # If new user input exists
        if user_input := st.session_state.get('user_input', None):
            with st.spinner("Generating response..."):
                try:
                    # Get response
                    answer = asyncio.run(
                        get_response_fn(
                            st.session_state.mode,
                            documents,
                            user_input,
                            chat_history=[str(x) for x in st.session_state.chat_history],
                            embedding_folder=embedding_folder
                        )
                    )

                    # Get sources
                    sources = get_source_fn(
                        documents,
                        user_input,
                        chat_history=[str(x) for x in st.session_state.chat_history],
                        embedding_folder=embedding_folder
                    )
                    # Validate sources
                    sources = sources if all(isinstance(s, str) for s in sources) else []

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    with st.chat_message("assistant", avatar=tutor_avatar):
                        st.write(answer)
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
                    st.error("There was an error parsing the response. Please try again.")

            # Highlight PDF excerpts
            if doc and st.session_state.get("chat_occurred", False):
                pages_with_excerpts = find_pages_with_excerpts(doc, st.session_state.sources)
                if "current_page" not in st.session_state:
                    st.session_state.current_page = pages_with_excerpts[0] + 1 if pages_with_excerpts else 1
                if 'pages_with_excerpts' not in st.session_state:
                    st.session_state.pages_with_excerpts = pages_with_excerpts
                st.session_state.annotations = get_highlight_info(doc, st.session_state.sources)
                if st.session_state.annotations:
                    st.session_state.current_page = min(
                        annotation["page"] for annotation in st.session_state.annotations
                    )


# Function to display the pdf viewer
def show_pdf_viewer(file):
    # left_col = st.columns([1,1])[0]
    # with left_col:
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
    col1, col2, col3, col4 = st.columns([8, 4, 3, 3], vertical_alignment='center')
    with col1:
        st.button("←", on_click=previous_page)
    with col2:
        total_pages = st.session_state.get('total_pages', 1)
        st.write(f"Page {st.session_state.current_page} of {total_pages}")
    with col4:
        st.button("→", on_click=next_page)