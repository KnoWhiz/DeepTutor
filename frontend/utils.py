import streamlit as st
import asyncio
from pipeline.science.pipeline.tutor_agent import tutor_agent
from pipeline.science.pipeline.get_response import generate_follow_up_questions

import logging
logger = logging.getLogger(__name__)


def streamlit_tutor_agent(chat_session, file_path, user_input):
    initial_message,\
    sources,\
    source_pages,\
    source_annotations,\
    refined_source_pages,\
    follow_up_questions = asyncio.run(tutor_agent(
        chat_session=chat_session,
        file_path=file_path,
        user_input=user_input,
        deep_thinking=True
    ))
    return initial_message, sources, source_pages, source_annotations, refined_source_pages, follow_up_questions


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


# Function to handle chat content
def chat_content():
    """Handle chat input submission."""
    if st.session_state.user_input and st.session_state.user_input.strip():
        user_input = st.session_state.user_input
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_session.chat_history = st.session_state.chat_history
        
        try:
            # Get response
            answer,\
            sources,\
            source_pages,\
            source_annotations,\
            refined_source_pages,\
            follow_up_questions = streamlit_tutor_agent(
                chat_session=st.session_state.chat_session,
                file_path=st.session_state.file_path,
                user_input=user_input
            )
            
            # Convert sources to dict if it's a list (for backward compatibility)
            if isinstance(sources, list):
                sources = {source: 1.0 for source in sources}
            
            # Generate follow-up questions for new response
            follow_up_questions = generate_follow_up_questions(answer, st.session_state.chat_history)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "follow_up_questions": follow_up_questions
            })
            
            if sources:
                st.session_state.chat_history.append({
                    "role": "source_buttons",
                    "sources": sources,
                    "pages": source_pages,
                    "timestamp": len(st.session_state.chat_history)
                })
            
            # Update chat session history
            st.session_state.chat_session.chat_history = st.session_state.chat_history
            
            # Clear input
            st.session_state.user_input = ""
            
        except Exception as e:
            st.error(f"An error occurred while processing your message: {str(e)}")
            # Remove the failed message from history
            if st.session_state.chat_history:
                st.session_state.chat_history.pop()
                st.session_state.chat_session.chat_history = st.session_state.chat_history


# Function to handle follow-up question clicks
def handle_follow_up_click(question: str):
    """Handle when a user clicks on a follow-up question.
    
    Args:
        question: The follow-up question text that was clicked
    """
    st.session_state.next_question = question
    
    # Create a temporary chat history for context-specific follow-up questions
    temp_chat_history = []
    
    # Find the last assistant message that generated this follow-up question
    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
        msg = st.session_state.chat_history[i]
        if msg["role"] == "assistant" and "follow_up_questions" in msg:
            if question in msg["follow_up_questions"]:
                # Include the context: previous user question and assistant's response
                if i > 0 and st.session_state.chat_history[i-1]["role"] == "user":
                    temp_chat_history.append(st.session_state.chat_history[i-1])  # Previous user question
                temp_chat_history.append(msg)  # Assistant's response
                break
    
    # Store the temporary chat history in session state for the agent to use
    st.session_state.temp_chat_history = temp_chat_history
    
    # Add the new question to the full chat history
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )
    # Update chat session history
    st.session_state.chat_session.chat_history = st.session_state.chat_history
