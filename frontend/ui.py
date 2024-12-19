import json
import base64
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import float_init, float_parent, float_css_helper

from frontend.utils import previous_page, next_page, close_pdf, chat_content
from pipeline.utils import find_pages_with_excerpts, get_highlight_info


# Function to set up the page configuration
def setup_page_config():
    st.set_page_config(
        page_title="KnoWhiz Office Hours",
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
            <h2 style='text-align: left;'>
                <img src="data:image/png;base64,{encoded_image}" alt='icon' style='width:50px; height:50px; vertical-align: left; margin-right: 10px;'>
                KnoWhiz Office Hours
            </h2>
            """,
            unsafe_allow_html=True
        )
        st.subheader(" ")
        st.subheader("Upload a document to get started.")


# Function to display the file uploader
def show_file_upload(on_change=None):
    with st.sidebar:
        return st.file_uploader(" ", type="pdf", on_change=on_change)


# Function to display the response mode options
def show_mode_option(uploaded_file):
    with st.sidebar:
        disabled = uploaded_file is not None
        mode_index = 0
        st.session_state.mode = st.radio(" ", options=["TA", "Professor"], index=mode_index, disabled=disabled)


# Function to display the chat interface
def show_page_option():
    with st.sidebar:
        # Navigation Menu
        menu = ["📑 Document reading", "📬 KnoWhiz?"]
        st.session_state.page = st.selectbox(" ", menu)


# Function to display the chat interface
def show_chat_interface(doc, documents, embedding_folder, get_response_fn, get_source_fn):
    # Init float function for chat_input textbox
    float_init(theme=True, include_unstable_primary=False)
    learner_avatar = "frontend/images/learner.svg"
    tutor_avatar = "frontend/images/tutor.svg"

    with st.container(border=st.session_state.show_chat_border, height=700):
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
                # if msg["role"] == "assistant":
                #     st.button(
                #         "Re-generate",
                #         key=f"regen_response_{idx}",
                #         on_click=regen_response
                #     )

        # If new user input exists
        if user_input := st.session_state.get('user_input', None):
            with st.spinner("Generating response..."):
                try:
                    # Get response
                    answer = get_response_fn(
                            st.session_state.mode,
                            documents,
                            user_input,
                            chat_history=st.session_state.chat_history,
                            embedding_folder=embedding_folder
                        )

                    # Get sources
                    sources = get_source_fn(
                        documents,
                        user_input,
                        answer,
                        chat_history=st.session_state.chat_history,
                        embedding_folder=embedding_folder
                    )
                    # Validate sources
                    sources = sources if all(isinstance(s, str) for s in sources) else []
                    # Print sources
                    print("Source content:", sources)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.chat_history.append(
                        {"role": "sources", "content": sources}
                    )
                    with st.chat_message("assistant", avatar=tutor_avatar):
                        st.write(answer)
                    with st.chat_message("sources", avatar=tutor_avatar):
                        st.write(sources)
                        
                        # st.button(
                        #     "Re-generate",
                        #     key=f"regen_response_{idx}",
                        #     on_click=regen_response
                        # )

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
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "annotations" not in st.session_state:
        st.session_state.annotations = []
    pdf_viewer(
        file,
        width=1000,
        height=630,
        annotations=st.session_state.annotations,
        pages_to_render=[st.session_state.current_page],
        render_text=True,
    )
    col1, col2, col3, col4 = st.columns([8, 6, 3, 3], vertical_alignment='center')
    with col1:
        st.button("←", on_click=previous_page)
    with col2:
        total_pages = st.session_state.get('total_pages', 1)
        st.write(f"Page {st.session_state.current_page} of {total_pages}")
    with col4:
        st.button("→", on_click=next_page)


# Function to display the footer
def show_footer():
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Professors** and **TAs** can make mistakes, sometimes you have to trust **YOURSELF**! 🧠")


# Function to display the contact us page
def show_contact_us():
    st.title("📬 Contact Us")
    st.markdown("""
    We'd love to hear from you! Whether you have any **question, feedback, or want to contribute**, feel free to reach out.

    - **Email:** [knowhiz.us@gmail.com](mailto:knowhiz.us@gmail.com) 📨
    - **Discord:** [Join our Discord community](https://discord.gg/7ucnweCKk8) 💬
    - **GitHub:** [Contribute on GitHub](https://github.com/KnoWhiz/KnoWhizTutor) 🛠️
    - **Follow us:** [LinkedIn](https://www.linkedin.com/company/knowhiz) | [Twitter](https://x.com/knowhizlearning) 🏄

    If you'd like to request a feature or report a bug, please **let us know!** Your suggestions are highly appreciated! 🙌
    """)
    st.title("🗂️ KnoWhiz flashcards")
    st.markdown("Want more **structured and systematic** learning? Check out our **[KnoWhiz flashcards learning platform](https://www.knowhiz.us/)!** 🚀")