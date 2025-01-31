import json
import base64
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import float_init, float_parent, float_css_helper
from streamlit_extras.stylable_container import stylable_container
from streamlit_js_eval import streamlit_js_eval

from frontend.utils import previous_page, next_page, close_pdf, chat_content
from pipeline.utils import find_pages_with_excerpts, get_highlight_info, translate_content
from frontend.forms.contact import contact_form
from pipeline.config import load_config


def to_emoji_number(num: int) -> str:
    """Convert an integer to a bold circled number (1-20).
    
    Args:
        num: Integer to convert
        
    Returns:
        String containing the bold circled number representation for 1-20,
        or regular number for values > 20
    """
    # Use circled numbers for 1-50
    circled_numbers = [
        "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
        "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳",
        "㉑", "㉒", "㉓", "㉔", "㉕", "㉖", "㉗", "㉘", "㉙", "㉚",
        "㉛", "㉜", "㉝", "㉞", "㉟", "㊱", "㊲", "㊳", "㊴", "㊵",
        "㊶", "㊷", "㊸", "㊹", "㊺", "㊻", "㊼", "㊽", "㊾", "㊿"
    ]
    if 1 <= num <= len(circled_numbers):
        return circled_numbers[num - 1]
    return str(num)  # Use regular number if > 20


# Function to set up the page configuration
def setup_page_config():
    st.set_page_config(
        page_title="DeepTutor",
        page_icon="frontend/images/professor.svg",
        layout="wide",
        # initial_sidebar_state="collapsed",
        initial_sidebar_state="expanded"
    )


def show_auth_top():
    # st.write("")
    pass


# Function to display the header
def show_header():
    with st.sidebar:
        with open("frontend/images/professor.ico", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <h1 style='text-align: left;'>
                <img src="data:image/png;base64,{encoded_image}" alt='icon' style='width:50px; height:50px; vertical-align: left; margin-right: 10px;'>
                DeepTutor
            </h1>
            """,
            unsafe_allow_html=True
        )
        st.subheader(" ")
        st.subheader("Upload a document to get started.")


# Function to display the file uploader
def show_file_upload(on_change=None):
    with st.sidebar:
        previous_file = st.session_state.get('uploaded_file', None)
        current_file = st.file_uploader(" ", type="pdf", on_change=on_change)
        
        # Check if file has changed
        if previous_file is not None and current_file is not None:
            if previous_file.name != current_file.name:
                # File has changed, trigger the change handler
                on_change()
        
        st.session_state.uploaded_file = current_file
        
        # Set upload state
        if st.session_state.get('is_uploaded_file', None):
            st.session_state['is_uploaded_file'] = True


# Function to display the response mode options
def show_mode_option(uploaded_file):
    with st.sidebar:
        disabled = uploaded_file is not None
        mode_index = 0
        st.session_state.mode = st.radio("Basic model (faster) or Advanced model (slower but more accurate)?", options=["Basic", "Advanced"], index=mode_index, disabled=disabled)


# Function to display the language selection options in the sidebar
def show_language_option():
    """Function to display the language selection options in the sidebar."""
    with st.sidebar:
        # Load languages from config
        config = load_config()
        languages = config['languages']

        # Get current language from session state or default to English
        current_lang = st.session_state.get("language", "English")
        
        # Create the language selector
        selected_lang_display = st.selectbox(
            "🌐 Language",
            options=list(languages.keys()),
            index=list(languages.values()).index(current_lang)
        )
        
        # Update the session state with the selected language code
        st.session_state.language = languages[selected_lang_display]


# Function to display the chat interface
def show_page_option():
    with st.sidebar:
        # Navigation Menu
        menu = ["📑 Document reading", "📬 DeepTutor?"]
        st.session_state.page = st.selectbox("🖥️ Page", menu)


# Function to display the chat interface
def show_chat_interface(doc, documents, embedding_folder, tutor_agent):
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

        # Generate initial welcome message if chat history is empty
        if not st.session_state.chat_history:
            with st.spinner("Loading document summary..."):
                initial_message, _ = tutor_agent(
                    mode=st.session_state.mode,
                    _doc=doc,
                    _documents=documents,
                    user_input=None,
                    chat_history=[],
                    embedding_folder=embedding_folder
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": initial_message}
                )

        # Display chat history
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                avatar = learner_avatar
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                avatar = professor_avatar if st.session_state.mode == "Advanced" else tutor_avatar
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])
            elif msg["role"] == "source_buttons":
                # Display source buttons in a row if there are sources
                if msg["sources"] and len(msg["sources"]) > 0:
                    cols = st.columns(len(msg["sources"]))
                    for idx, (col, source) in enumerate(zip(cols, msg["sources"]), 1):
                        page_num = msg["pages"][source]
                        with col:
                            if st.button(to_emoji_number(idx), key=f"source_btn_{idx}_{msg['timestamp']}", use_container_width=True):
                                st.session_state.current_page = page_num
                                st.session_state.annotations = get_highlight_info(doc, [source])
                else:
                    st.info("No sources were found for this response.")

        # If new user input exists
        if user_input := st.session_state.get('user_input', None):
            with st.spinner("Generating response..."):
                try:
                    # Get response
                    answer, sources = tutor_agent(
                        mode=st.session_state.mode,
                        _doc=doc,
                        _documents=documents,
                        user_input=user_input,
                        chat_history=st.session_state.chat_history,
                        embedding_folder=embedding_folder
                    )
                    # Validate sources
                    sources = sources if all(isinstance(s, str) for s in sources) else []
                    # Print sources
                    print("Source content:", sources)
                    
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
                    if sources and len(sources) > 0:
                        cols = st.columns(len(sources))
                        for idx, (col, source) in enumerate(zip(cols, sources), 1):
                            page_num = source_pages.get(source)
                            if page_num:
                                with col:
                                    if st.button(to_emoji_number(idx), key=f"source_btn_{idx}_current", use_container_width=True):
                                        st.session_state.current_page = page_num
                                        st.session_state.annotations = get_highlight_info(doc, [source])
                    else:
                        st.info("No relevant sources found for this response.")
                    
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
    if "total_pages" not in st.session_state:
        # Get total pages from the PDF file
        import PyPDF2
        pdf = PyPDF2.PdfReader(file)
        st.session_state.total_pages = len(pdf.pages)
        
    # # TEST
    # page_height = streamlit_js_eval(js_expressions='window.innerHeight', key='HEIGHT', want_output=True)
    # print(f"Page height is {page_height} pixels.")
    
    with st.container(border=st.session_state.show_chat_border, height=620):
        pdf_viewer(
            file,
            width=1000,
            annotations=st.session_state.annotations,
            pages_to_render=[st.session_state.current_page],
            render_text=True,
        )
        
        # Create three columns for the navigation controls
        columns = st.columns([1, 2, 1])
        
        # Left arrow button
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
        
        # Page counter in the middle
        with columns[1]:
            st.markdown(
                f"""<div style="text-align: center; color: #666666;">
                    Page {st.session_state.current_page} of {st.session_state.total_pages}
                    </div>""",
                unsafe_allow_html=True
            )
            
        # Right arrow button
        with columns[2]:
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
    - **GitHub:** [Contribute on GitHub](https://github.com/DeepTutor/DeepTutor) 🛠️
    - **Follow us:** [LinkedIn](https://www.linkedin.com/company/knowhiz) | [Twitter](https://x.com/knowhizlearning) 🏄

    If you'd like to request a feature or report a bug, please **let us know!** Your suggestions are highly appreciated! 🙌
    """)
    if st.button("Feedback Form"):
        show_contact_form()
    st.title("🗂️ KnoWhiz flashcards")
    st.markdown("Want more **structured and systematic** learning? Check out our **[KnoWhiz flashcards learning platform](https://www.knowhiz.us/)!** 🚀")
