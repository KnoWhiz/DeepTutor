import streamlit as st
import asyncio
from pipeline.science.pipeline.tutor_agent import tutor_agent
from pipeline.science.pipeline.get_response import generate_follow_up_questions
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from typing import Generator

import logging
logger = logging.getLogger("tutorfrontend.utils")


def streamlit_tutor_agent(chat_session, file_path, user_input):    
    answer = asyncio.run(tutor_agent(
        chat_session=chat_session,
        file_path_list=[file_path],
        user_input=user_input,
        deep_thinking=True
    ))
    sources = {}
    source_pages = {}
    source_annotations = {}
    refined_source_pages = {}
    refined_source_index = {}
    follow_up_questions = []
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
    )


def format_response(response_content):
    """Format assistant content by removing think tags."""
    return (
        response_content.replace("<response>\n\n</response>", "")
        .replace("<response>", "")
        .replace("</response>", "")
    )


def process_response_phase(response_placeholder, stream_response: Generator, mode: ChatMode = None, stream: bool = False):
    """
    Process the response phase of the assistant's response.
    Args:
        stream_response: The generator object from the stream response.
    Returns:
        The response content as a string.
    """
    if stream:
        response_content = response_placeholder.write_stream(stream_response)
        # response_content = ""
        # with st.status("Responding...", expanded=True) as status:
        #     response_placeholder = st.empty()
            
        #     for chunk in stream_response:
        #         content = chunk or ""
        #         response_content += content
                
        #         if "<response>" in content:
        #             continue
        #         if "</response>" in content:
        #             content = content.replace("</response>", "")
        #             status.update(label="Responding complete!", state="complete", expanded=True)
        #             return response_content
        #         response_placeholder.markdown(format_response(response_content))

        # return response_content

    else:
        response_placeholder.write(stream_response)
        response_content = stream_response
    logger.info(f"Final whole response content: {response_content}")
    return response_content


def process_thinking_phase(response_placeholder, answer):
    """Process the thinking phase of the assistant's response."""
    thinking_content = ""
    with st.status("Thinking...", expanded=True) as status:
        think_placeholder = st.empty()
        
        for chunk in answer:
            content = chunk or ""
            thinking_content += content
            
            if "<think>" in content:
                continue
            if "</think>" in content:
                content = content.replace("</think>", "")
                status.update(label="Thinking complete!", state="complete", expanded=False)
                return thinking_content
            think_placeholder.markdown(format_reasoning_response(thinking_content))
    
    # return format_reasoning_response(thinking_content)
    return thinking_content


# Function to display the pdf
def previous_page():
    logger.info("previous_page")
    if st.session_state.current_page > 1:
        st.session_state.current_page = st.session_state.current_page - 1
    logger.info(f"st.session_state.current_page is {st.session_state.current_page}")


# Function to display the pdf
def next_page():
    logger.info("next_page")
    if st.session_state.current_page < st.session_state.total_pages:
        st.session_state.current_page = st.session_state.current_page + 1
    logger.info(f"st.session_state.current_page is {st.session_state.current_page}")


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
