import json
import base64
import PyPDF2
import pprint
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import float_init, float_parent, float_css_helper
from streamlit_extras.stylable_container import stylable_container
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval

from frontend.utils import previous_page, next_page, close_pdf, chat_content, handle_follow_up_click
from pipeline.utils import find_pages_with_excerpts, get_highlight_info, translate_content
from frontend.forms.contact import contact_form
from pipeline.config import load_config
from pipeline.get_response import generate_follow_up_questions
from pipeline.chat_history_manager import save_chat_history


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
        # st.subheader("Upload a document to get started.")


# Function to display the response mode options
def show_mode_option():
    with st.sidebar:
        mode_index = 0
        st.session_state.mode = st.radio("Basic model (faster) or Advanced model (slower but more accurate)?", options=["Basic", "Advanced"], index=mode_index)


# Function to display the file uploader
def show_file_upload(on_change=None):
    with st.sidebar:
        previous_file = st.session_state.get('uploaded_file', None)
        current_file = st.file_uploader("Upload a document to get started.", type="pdf", on_change=on_change)
        
        # Check if file has changed
        if previous_file is not None and current_file is not None:
            if previous_file.name != current_file.name:
                # File has changed, trigger the change handler
                on_change()
        
        st.session_state.uploaded_file = current_file
        
        # Set upload state
        if st.session_state.get('is_uploaded_file', None):
            st.session_state['is_uploaded_file'] = True


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


def get_relevance_color(score):
    """Convert a relevance score to a shade of grey.
    
    Args:
        score: Float between 0 and 1
        
    Returns:
        Hex color code string for a shade of grey, where:
        - High relevance (1.0) = Dark grey (#404040)
        - Medium relevance (0.5) = Medium grey (#808080)
        - Low relevance (0.0) = Light grey (#C0C0C0)
    """
    # Convert score to a grey value between 192 (C0) and 100 (40)
    grey_value = int(192 - (score * 92))
    return f"#{grey_value:02x}{grey_value:02x}{grey_value:02x}"


# Function to display the chat interface
def show_chat_interface(doc, documents, file_paths, embedding_folder, tutor_agent):
    # Init float function for chat_input textbox
    learner_avatar = "frontend/images/learner.svg"
    tutor_avatar = "frontend/images/tutor.svg"
    professor_avatar = "frontend/images/professor.svg"

    with st.container():
        float_init(theme=True, include_unstable_primary=False)
        st.chat_input(key='user_input', on_submit=chat_content)
        button_b_pos = "1.2rem"
        button_css = float_css_helper(width="1.2rem", bottom=button_b_pos, transition=0)
        float_parent(css=button_css)

    chat_container = st.container(border=st.session_state.show_chat_border, height=1005)

    with chat_container:
        # Generate initial welcome message if chat history is empty
        if not st.session_state.chat_history:
            with st.spinner("Loading document summary..."):
                initial_message, sources, source_pages = tutor_agent(
                    mode=st.session_state.mode,
                    _doc=doc,
                    _documents=documents,
                    file_paths=file_paths,
                    user_input=None,
                    chat_history=[],
                    embedding_folder=embedding_folder
                )
                # Convert sources to dict if it's a list (for backward compatibility)
                if isinstance(sources, list):
                    sources = {source: 1.0 for source in sources}  # Assign max relevance to initial sources
                
                # Generate follow-up questions for initial message
                follow_up_questions = generate_follow_up_questions(initial_message, [])
                st.session_state.chat_history.append(
                    {
                        "role": "assistant", 
                        "content": initial_message,
                        "follow_up_questions": follow_up_questions
                    }
                )
                
                if sources:
                    # Store source-to-page mapping
                    source_pages = source_pages
                    # source_pages = {}
                    # for source in sources.keys():
                    #     pages_with_excerpts = find_pages_with_excerpts(doc, [source])
                    #     if pages_with_excerpts:
                    #         source_pages[source] = pages_with_excerpts[0] + 1

                    # # TEST
                    # print(f"Current source_pages: {source_pages}")
                    
                    st.session_state.chat_history.append({
                        "role": "source_buttons",
                        "sources": sources,
                        "pages": source_pages,
                        "timestamp": len(st.session_state.chat_history)
                    })
                # Save chat history after initial message
                save_chat_history(st.session_state.session_id, st.session_state.chat_history)

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
                    
                    # First display source buttons if this message has associated sources
                    next_msg = st.session_state.chat_history[idx + 1] if idx + 1 < len(st.session_state.chat_history) else None
                    if next_msg and next_msg["role"] == "source_buttons":
                        sources = next_msg["sources"]
                        # Convert sources to dict if it's a list (for backward compatibility)
                        if isinstance(sources, list):
                            sources = {source: 1.0 for source in sources}  # Assign max relevance to old sources
                            next_msg["sources"] = sources
                        
                        if sources and len(sources) > 0:
                            st.write("\n\n**📚 Sources:**")
                            # Sort sources by page numbers
                            sorted_sources = sorted(sources.items(), key=lambda x: next_msg["pages"][x[0]])
                            cols = st.columns(len(sources))
                            for src_idx, (col, (source, score)) in enumerate(zip(cols, sorted_sources), 1):
                                page_num = next_msg["pages"][source]
                                with col:
                                    # Create a stylable container for the button with custom color
                                    button_color = get_relevance_color(score)
                                    button_style = """
                                    button {{
                                        background-color: {button_color} !important;
                                        border-color: {button_color} !important;
                                        color: white !important;
                                        transition: filter 0.2s !important;
                                    }}
                                    button:hover {{
                                        background-color: {button_color} !important;
                                        border-color: {button_color} !important;
                                        filter: brightness(120%) !important;
                                    }}
                                    """
                                    with stylable_container(
                                        key=f"source_btn_container_{idx}_{src_idx}",
                                        css_styles=button_style.format(button_color=button_color)
                                    ):
                                        if st.button(to_emoji_number(src_idx), key=f"source_btn_{idx}_{src_idx}", use_container_width=True):
                                            st.session_state.current_page = page_num
                                            st.session_state.annotations = get_highlight_info(doc, [source])
                    
                    # Then display follow-up questions
                    if "follow_up_questions" in msg:
                        st.write("\n\n**📝 Follow-up Questions:**")
                        for q_idx, question in enumerate(msg["follow_up_questions"], 1):
                            if st.button(f"{q_idx}. {question}", key=f"follow_up_{idx}_{q_idx}"):
                                handle_follow_up_click(question)
            elif msg["role"] == "source_buttons":
                # Skip source buttons here since we're showing them with the assistant message
                pass

        # If there's a next question from follow-up click, process it
        if "next_question" in st.session_state:
            user_input = st.session_state.next_question
            del st.session_state.next_question
        else:
            user_input = st.session_state.get('user_input', None)

        # If we have input to process
        if user_input:
            with st.spinner("Generating response..."):
                try:
                    # Get response
                    answer, sources, source_pages = tutor_agent(
                        mode=st.session_state.mode,
                        _doc=doc,
                        _documents=documents,
                        file_paths=file_paths,
                        user_input=user_input,
                        chat_history=st.session_state.chat_history,
                        embedding_folder=embedding_folder
                    )
                    # Convert sources to dict if it's a list (for backward compatibility)
                    if isinstance(sources, list):
                        sources = {source: 1.0 for source in sources}  # Assign max relevance to old sources
                    else:
                        # Validate sources is a dictionary
                        sources = sources if isinstance(sources, dict) else {}
                    
                    # Store source-to-page mapping
                    source_pages = source_pages
                    # source_pages = {}
                    # for source in sources.keys():
                    #     pages_with_excerpts = find_pages_with_excerpts(doc, [source])
                    #     if pages_with_excerpts:
                    #         source_pages[source] = pages_with_excerpts[0] + 1

                    # # TEST
                    # print(f"Current source_pages: {source_pages}")
                    
                    # Generate follow-up questions for new response
                    follow_up_questions = generate_follow_up_questions(answer, st.session_state.chat_history)
                    
                    # Add response with follow-up questions to chat history
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant", 
                            "content": answer,
                            "follow_up_questions": follow_up_questions
                        }
                    )
                    
                    # Add source buttons to chat history
                    st.session_state.chat_history.append({
                        "role": "source_buttons",
                        "sources": sources,
                        "pages": source_pages,
                        "timestamp": len(st.session_state.chat_history)
                    })
                    # Save chat history after assistant response
                    save_chat_history(st.session_state.session_id, st.session_state.chat_history)
                    
                    # Display current response
                    with st.chat_message("assistant", avatar=tutor_avatar):
                        st.write(answer)
                        
                        # First display source buttons
                        if sources and len(sources) > 0:
                            st.write("\n\n**📚 Sources:**")
                            # Sort sources by page numbers
                            sorted_sources = sorted(sources.items(), key=lambda x: source_pages.get(x[0], 0))
                            cols = st.columns(len(sources))
                            for idx, (col, (source, score)) in enumerate(zip(cols, sorted_sources), 1):
                                page_num = source_pages.get(source)
                                if page_num:
                                    with col:
                                        # Create a stylable container for the button with custom color
                                        button_color = get_relevance_color(score)
                                        button_style = """
                                        button {{
                                            background-color: {button_color} !important;
                                            border-color: {button_color} !important;
                                            color: white !important;
                                            transition: filter 0.2s !important;
                                        }}
                                        button:hover {{
                                            background-color: {button_color} !important;
                                            border-color: {button_color} !important;
                                            filter: brightness(120%) !important;
                                        }}
                                        """
                                        with stylable_container(
                                            key=f"source_btn_container_current_{idx}",
                                            css_styles=button_style.format(button_color=button_color)
                                        ):
                                            if st.button(to_emoji_number(idx), key=f"source_btn_{idx}_current", use_container_width=True):
                                                st.session_state.current_page = page_num
                                                st.session_state.annotations = get_highlight_info(doc, [source])
                        
                        # Then display follow-up questions
                        st.write("\n\n**📝 Follow-up Questions:**")
                        for q_idx, question in enumerate(follow_up_questions, 1):
                            if st.button(f"{q_idx}. {question}", key=f"follow_up_current_{q_idx}"):
                                handle_follow_up_click(question)

                    st.session_state.sources = sources
                    st.session_state.chat_occurred = True

                except json.JSONDecodeError:
                    st.error("There was an error parsing the response. Please try again.")

            # Highlight PDF excerpts
            if doc and st.session_state.get("chat_occurred", False):
                if "current_page" not in st.session_state:
                    st.session_state.current_page = 1
                if st.session_state.get("sources"):
                    st.session_state.annotations = get_highlight_info(doc, list(st.session_state.sources.keys()))


# Function to display the pdf viewer
def show_pdf_viewer(file):
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "annotations" not in st.session_state:
        st.session_state.annotations = []
    if "total_pages" not in st.session_state:
        # Get total pages from the PDF file
        pdf = PyPDF2.PdfReader(file)
        st.session_state.total_pages = len(pdf.pages)
        
    # # TEST
    # page_height = streamlit_js_eval(js_expressions='window.innerHeight', key='HEIGHT', want_output=True)
    # print(f"Page height is {page_height} pixels.")

    with st.container():
        st.markdown("""
        <style>
        .fullHeight {
            height: 80vh;
            width: 100%;
            max-width: 100%;
            overflow: auto;
        }
        .stPdfViewer {
            width: 100% !important;
            height: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    pdf_container = st.container(border=st.session_state.show_chat_border, height=1005)

    with pdf_container:
        pdf_viewer(
            file,
            width="100%",
            annotations=st.session_state.annotations,
            pages_to_render=[st.session_state.current_page],
            render_text=True
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
    # viewer_css = float_css_helper(transition=0)
    # float_parent(css=viewer_css)


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
