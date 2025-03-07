import streamlit as st
import asyncio
from pipeline.science.pipeline.tutor_agent import tutor_agent
from pipeline.science.pipeline.get_response import generate_follow_up_questions
from pipeline.science.pipeline.session_manager import ChatSession

import logging
logger = logging.getLogger("tutorfrontend.utils")


def streamlit_tutor_agent(chat_session, file_path, user_input):    
    answer, \
    sources, \
    source_pages, \
    source_annotations, \
    refined_source_pages, \
    refined_source_index, \
    follow_up_questions = asyncio.run(tutor_agent(
        chat_session=chat_session,
        file_path=file_path,
        user_input=user_input,
        deep_thinking=True
    ))
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


# Function to display the pdf
def previous_page():
    logger.info("previous_page")
    if st.session_state.current_page > 1:
        st.session_state.current_page = st.session_state.current_page - 1
    logger.info("st.session_state.current_page is %s", st.session_state.current_page)


# Function to display the pdf
def next_page():
    logger.info("next_page")
    if st.session_state.current_page < st.session_state.total_pages:
        st.session_state.current_page = st.session_state.current_page + 1
    logger.info("st.session_state.current_page is %s", st.session_state.current_page)


# Function to close the pdf
def close_pdf():
    st.session_state.show_pdf = False


# Function to open the pdf
def file_changed():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# Function to handle follow-up question clicks
def handle_follow_up_click(chat_session: ChatSession, question: str):
    """Handle when a user clicks on a follow-up question.
    
    Args:
        question: The follow-up question text that was clicked
    """
    st.session_state.next_question = question
    
    # Create a temporary chat history for context-specific follow-up questions
    temp_chat_history = []
    
    # Find the last assistant message that generated this follow-up question
    for i in range(len(st.session_state.chat_session.chat_history) - 1, -1, -1):
        msg = st.session_state.chat_session.chat_history[i]
        if msg["role"] == "assistant" and "follow_up_questions" in msg:
            if question in msg["follow_up_questions"]:
                # Include the context: previous user question and assistant's response
                if i > 0 and st.session_state.chat_session.chat_history[i-1]["role"] == "user":
                    temp_chat_history.append(st.session_state.chat_session.chat_history[i-1])  # Previous user question
                temp_chat_history.append(msg)  # Assistant's response
                break
    
    # Store the temporary chat history in session state for the agent to use
    st.session_state.temp_chat_history = temp_chat_history
    
    # Add the new question to the full chat history
    st.session_state.chat_session.chat_history.append(
        {"role": "user", "content": question}
    )
    # Update chat session history
    st.session_state.chat_session.chat_history = st.session_state.chat_session.chat_history
