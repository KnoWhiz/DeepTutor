import os
import json
import time
from typing import Dict, Generator
import re

from pipeline.science.pipeline.utils import (
    generate_file_id,
    format_time_tracking,
    clean_translation_prefix,
    responses_refine,
    extract_answer_content,
    extract_lite_mode_content,
    extract_basic_mode_content,
    extract_advanced_mode_content,
)
from pipeline.science.pipeline.content_translator import (
    detect_language,
    translate_content
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
