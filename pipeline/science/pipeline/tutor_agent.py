import os
import json
import time
from typing import Dict, Generator
import re

from pipeline.science.pipeline.utils import (
    translate_content,
    generate_file_id,
    format_time_tracking,
    detect_language,
    clean_translation_prefix,
    responses_refine,
)
from pipeline.science.pipeline.doc_processor import (
    save_file_txt_locally,
    process_pdf_file,
    get_highlight_info,
)
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.helper.index_files_saving import (
    vectorrag_index_files_decompress,
    vectorrag_index_files_compress,
    graphrag_index_files_decompress,
    graphrag_index_files_compress,
    literag_index_files_decompress,
)
from pipeline.science.pipeline.embeddings_agent import embeddings_agent
from pipeline.science.pipeline.get_response import (
    get_query_helper,
    get_response,
    generate_follow_up_questions,
    Question,
)
from pipeline.science.pipeline.sources_retrieval import get_response_source
from pipeline.science.pipeline.config import load_config

# Import mode-specific implementations
from pipeline.science.pipeline.tutor_agent_lite import tutor_agent_lite, tutor_agent_lite_streaming_tracking
from pipeline.science.pipeline.tutor_agent_basic import tutor_agent_basic, tutor_agent_basic_streaming_tracking
from pipeline.science.pipeline.tutor_agent_advanced import tutor_agent_advanced, tutor_agent_advanced_streaming_tracking

import logging
logger = logging.getLogger("tutorpipeline.science.tutor_agent")


def extract_answer_content(message_content):
    sources = {}    # {source_string: source_score}
    source_pages = {}    # {source_page_string: source_page_score}
    source_annotations = {}
    refined_source_pages = {}    # {refined_source_page_string: refined_source_page_score}
    refined_source_index = {}    # {refined_source_index_string: refined_source_index_score}
    follow_up_questions = []

    # Extract the main answer (content between <response> tags)
    # The logic is: if we have <response> tags, we extract the content between them
    # Otherwise, we extract the content between <original_response> and </original_response> tags
    # If we have neither, we extract the content between <thinking> and </thinking> tags
    # If we have none of the above, we return an empty string
    answer = ""
    thinking = ""
    response_match = re.search(r'<response>(.*?)</response>', message_content, re.DOTALL)
    original_response_match = re.search(r'<original_response>(.*?)</original_response>', message_content, re.DOTALL)
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', message_content, re.DOTALL)
    if response_match:
        answer = response_match.group(1).strip()
    elif original_response_match:
        answer = original_response_match.group(1).strip()
    elif thinking_match:
        answer = thinking_match.group(1).strip()
    else:
        answer = ""

    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Extract follow-up questions (content between <followup_question> tags)
    followup_matches = re.finditer(r'<followup_question>(.*?)</followup_question>', message_content, re.DOTALL)
    for match in followup_matches:
        question = match.group(1).strip()
        if question:
            # Remove any residual XML tags
            question = re.sub(r'<followup_question>.*?</followup_question>', '', question)

            # Apply the clean_translation_prefix function
            question = clean_translation_prefix(question)

            follow_up_questions.append(question)

    # Extract sources (content between <source> tags)
    source_matches = re.finditer(r'<source>(.*?)</source>', message_content, re.DOTALL)
    for match in source_matches:
        source_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', source_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                sources[key] = float(value)
            except ValueError:
                # If conversion fails, store as string
                sources[key] = value

    # Extract source pages (content between <source_page> tags)
    source_page_matches = re.finditer(r'<source_page>(.*?)</source_page>', message_content, re.DOTALL)
    for match in source_page_matches:
        source_page_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', source_page_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                source_pages[key] = float(value)
            except ValueError:
                # If conversion fails, store as string
                source_pages[key] = value

    # Extract refined source pages (content between <refined_source_page> tags)
    refined_source_page_matches = re.finditer(r'<refined_source_page>(.*?)</refined_source_page>', message_content, re.DOTALL)
    for match in refined_source_page_matches:
        refined_source_page_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', refined_source_page_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                refined_source_pages[key] = float(value)
            except ValueError:
                # If conversion fails, store as string
                refined_source_pages[key] = value

    # Extract refined source index (content between <refined_source_index> tags)
    refined_source_index_matches = re.finditer(r'<refined_source_index>(.*?)</refined_source_index>', message_content, re.DOTALL)
    for match in refined_source_index_matches:
        refined_source_index_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', refined_source_index_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float or int
                refined_source_index[key] = float(value)
            except ValueError:
                try:
                    # Try converting to int if float conversion fails
                    refined_source_index[key] = int(value)
                except ValueError:
                    # If both conversions fail, store as string
                    refined_source_index[key] = value

    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking


def extract_lite_mode_content(message_content):
    return extract_answer_content(message_content)


def extract_basic_mode_content(message_content):
    return extract_answer_content(message_content)


def extract_advanced_mode_content(message_content):
    return extract_answer_content(message_content)


async def tutor_agent(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    """
    Taking the user input, document, and chat history, generate a response and sources.
    If user_input is None, generates the initial welcome message.

    This function acts as a router that calls the appropriate specialized function
    based on the chat session mode.
    """
    # Initialize the current message
    chat_session.current_message = ""
    if time_tracking is None:
        time_tracking = {}

    config = load_config()
    stream = config["stream"]

    # Route to appropriate specialized agent based on mode
    if chat_session.mode == ChatMode.LITE:
        return await tutor_agent_lite(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)
    elif chat_session.mode == ChatMode.BASIC:
        return await tutor_agent_basic(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)
    elif chat_session.mode == ChatMode.ADVANCED:
        return await tutor_agent_advanced(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)
    else:
        logger.error(f"Invalid chat mode: {chat_session.mode}")
        error_message = "Error: Invalid chat mode."
        # return error_message, {}, {}, {}, {}, {}, []
        return error_message
