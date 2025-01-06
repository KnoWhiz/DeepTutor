import json
import base64
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import float_init, float_parent, float_css_helper
from streamlit_extras.stylable_container import stylable_container

from frontend.utils import previous_page, next_page, close_pdf, chat_content
from pipeline.utils import find_pages_with_excerpts, get_highlight_info
from frontend.forms.contact import contact_form


def to_roman(num):
    """Convert an integer to a Roman numeral."""
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
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
        if st.session_state['is_uploaded_file'] is not True:
            st.session_state.uploaded_file = st.file_uploader(" ", type="pdf", on_change=on_change)
        # if file uploaded successfully, set st.session_state['uploaded_file'] to True
        if st.session_state.get('is_uploaded_file', None):
            st.session_state['is_uploaded_file'] = True


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
def show_chat_interface(doc, documents, embedding_folder, get_response_fn, get_source_fn, get_query_fn):
    # Init float function for chat_input textbox
    learner_avatar = "frontend/images/learner.svg"
    tutor_avatar = "frontend/images/tutor.svg"
    professor_avatar = "frontend/images/professor.svg"

    with st.container(border=st.session_state.show_chat_border, height=620):
        float_init(theme=True, include_unstable_primary=False)
        with st.container():
            st.chat_input(key='user_input', on_submit=chat_content)
            button_b_pos = "1.2rem"
            button_css = float_css_helper(width="1.2rem", bottom=button_b_pos, transition=0)
            float_parent(css=button_css)

        # Display existing chat history
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                avatar = learner_avatar
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                avatar = professor_avatar if st.session_state.mode == "Professor" else tutor_avatar
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])
            elif msg["role"] == "source_buttons":
                # Display source buttons in a row
                cols = st.columns(len(msg["sources"]))
                for idx, (col, source) in enumerate(zip(cols, msg["sources"]), 1):
                    page_num = msg["pages"][source]
                    with col:
                        if st.button(str(idx), key=f"source_btn_{idx}_{msg['timestamp']}", use_container_width=True):
                            st.session_state.current_page = page_num
                            st.session_state.annotations = get_highlight_info(doc, [source])

        # If new user input exists
        if user_input := st.session_state.get('user_input', None):
            with st.spinner("Generating response..."):
                try:
                    # Rephrase the user input
                    user_input = get_query_fn(
                            user_input,
                            chat_history=st.session_state.chat_history,
                            embedding_folder=embedding_folder
                        )

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

                    answer = f"""Are you asking: **{user_input}**
                    """ + "\n" + answer
                    
                    # Store source-to-page mapping
                    source_pages = {}
                    for source in sources:
                        pages_with_excerpts = find_pages_with_excerpts(doc, [source])
                        if pages_with_excerpts:
                            source_pages[source] = pages_with_excerpts[0] + 1
                    
                    # Add response to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    
                    # Add source buttons to chat history
                    st.session_state.chat_history.append({
                        "role": "source_buttons",
                        "sources": sources,
                        "pages": source_pages,
                        "timestamp": len(st.session_state.chat_history)  # Use as unique key for buttons
                    })
                    
                    with st.chat_message("assistant", avatar=tutor_avatar):
                        st.write(answer)
                    
                    # Display source buttons immediately
                    cols = st.columns(len(sources))
                    for idx, (col, source) in enumerate(zip(cols, sources), 1):
                        page_num = source_pages.get(source)
                        if page_num:
                            with col:
                                if st.button(str(idx), key=f"source_btn_{idx}_current", use_container_width=True):
                                    st.session_state.current_page = page_num
                                    st.session_state.annotations = get_highlight_info(doc, [source])
                    
                    st.session_state.sources = sources
                    st.session_state.chat_occurred = True

                except json.JSONDecodeError:
                    st.error("There was an error parsing the response. Please try again.")

            # Highlight PDF excerpts
            if doc and st.session_state.get("chat_occurred", False):
                if "current_page" not in st.session_state:
                    st.session_state.current_page = 1
                if st.session_state.get("sources"):
                    st.session_state.annotations = get_highlight_info(doc, st.session_state.sources)
    viewer_css = float_css_helper(transition=0)
    float_parent(css=viewer_css)


# Function to display the pdf viewer
def show_pdf_viewer(file):
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "annotations" not in st.session_state:
        st.session_state.annotations = []
    with st.container(border=st.session_state.show_chat_border, height=620):
        pdf_viewer(
            file,
            width=1000,
            annotations=st.session_state.annotations,
            pages_to_render=[st.session_state.current_page],
            render_text=True,
        )
        columns = st.columns([1, 1])
        with columns[0]:
            with stylable_container(
                key="left_aligned_button",
                css_styles="""
                button {
                    float: left;
                }
                """
            ):
                st.button("←", key='←', on_click=previous_page)
                button_css = float_css_helper(width="1.2rem", bottom="1.2rem", transition=0)
                float_parent(css=button_css)
        with columns[1]:
            with stylable_container(
                key="right_aligned_button",
                css_styles="""
                button {
                    float: right;
                }
                """
            ):
                st.button("→", key='→', on_click=next_page)
                button_css = float_css_helper(width="1.2rem", bottom="1.2rem", transition=0)
                float_parent(css=button_css)
    viewer_css = float_css_helper(transition=0)
    float_parent(css=viewer_css)


# Function to display the footer
def show_footer():
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Professors** and **TAs** can make mistakes, sometimes you have to trust **YOURSELF**! 🧠")


@st.dialog("Contact Us")
def show_contact_form():
    contact_form()


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
    if st.button("Feedback Form"):
        show_contact_form()
    st.title("🗂️ KnoWhiz flashcards")
    st.markdown("Want more **structured and systematic** learning? Check out our **[KnoWhiz flashcards learning platform](https://www.knowhiz.us/)!** 🚀")
