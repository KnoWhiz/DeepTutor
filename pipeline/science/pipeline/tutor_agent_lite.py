import os
import time
from typing import Dict, Generator
import re

from pipeline.science.pipeline.utils import (
    generate_file_id,
    format_time_tracking,
    clean_translation_prefix,
)
from pipeline.science.pipeline.content_translator import (
    detect_language,
    translate_content
)
from pipeline.science.pipeline.doc_processor import (
    save_file_txt_locally,
    process_pdf_file,
)
from pipeline.science.pipeline.session_manager import ChatSession
from pipeline.science.pipeline.helper.index_files_saving import (
    literag_index_files_decompress,
)
from pipeline.science.pipeline.embeddings_agent import embeddings_agent
from pipeline.science.pipeline.get_response import (
    get_response,
    generate_follow_up_questions,
    Question,
)
from pipeline.science.pipeline.config import load_config

import logging
logger = logging.getLogger("tutorpipeline.science.tutor_agent_lite")


async def tutor_agent_lite(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    """
    Lightweight tutor agent that provides basic tutoring capabilities with minimal resource usage.
    Uses LiteRAG for document processing and doesn't perform advanced source retrieval.

    Args:
        chat_session: Current chat session object
        file_path_list: List of paths to uploaded documents
        user_input: The user's query or input
        time_tracking: Dictionary to track execution time of various steps
        deep_thinking: Whether to use deep thinking for response generation

    Returns:
        Tuple containing (answer, sources, source_pages, source_annotations,
                         refined_source_pages, refined_source_index, follow_up_questions)
    """
    if time_tracking is None:
        time_tracking = {}

    return tutor_agent_lite_streaming_tracking(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)
    # answer = tutor_agent_lite_streaming_tracking(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)

    # # For Lite mode, we have minimal sources and follow-up questions
    # sources = {}
    # source_pages = {}
    # source_annotations = {}
    # refined_source_pages = {}
    # refined_source_index = {}
    # follow_up_questions = []

    # return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


async def tutor_agent_lite_streaming_tracking(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    async for chunk in tutor_agent_lite_streaming(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream):
        yield chunk
        chat_session.current_message += chunk

    # answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions = extract_lite_mode_content(chat_session.current_message)
    # logger.info(f"Extracted answer: {answer}")
    # logger.info(f"Extracted sources: {sources}")
    # logger.info(f"Extracted source pages: {source_pages}")
    # logger.info(f"Extracted source annotations: {source_annotations}")
    # logger.info(f"Extracted refined source pages: {refined_source_pages}")
    # logger.info(f"Extracted refined source index: {refined_source_index}")
    # logger.info(f"Extracted follow-up questions: {follow_up_questions}")
    # logger.info(f"Current message: {chat_session.current_message}")


async def tutor_agent_lite_streaming(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    """
    Streaming tutor agent for Lite mode.

    Args:
        chat_session: Current chat session object
        file_path_list: List of paths to uploaded documents
        user_input: The user's query or input
        time_tracking: Dictionary to track execution time of various steps
        deep_thinking: Whether to use deep thinking for response generation

    Returns:
        Generator of response chunks
    """
    if time_tracking is None:
        time_tracking = {}

    config = load_config()

    # Compute hashed ID and prepare embedding folder
    yield "<thinking>"
    yield "Processing documents ...\n\n"
    hashing_start_time = time.time()
    file_id_list = [generate_file_id(file_path) for file_path in file_path_list]
    path_prefix = os.getenv("FILE_PATH_PREFIX")
    if not path_prefix:
        path_prefix = ""
    embedded_content_path = os.path.join(path_prefix, 'embedded_content')
    embedding_folder_list = [os.path.join(embedded_content_path, file_id) for file_id in file_id_list]
    logger.info(f"Embedding folder: {embedding_folder_list}")
    if not os.path.exists(embedded_content_path):
        os.makedirs(embedded_content_path)
    for embedding_folder in embedding_folder_list:
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)
    time_tracking["file_hashing_setup_dirs"] = time.time() - hashing_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Save the file txt content locally
    save_file_start_time = time.time()
    filename_list = [os.path.basename(file_path) for file_path in file_path_list]
    for file_path, filename in zip(file_path_list, filename_list):
        save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder, chat_session=chat_session)
    time_tracking["file_loading_save_text"] = time.time() - save_file_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    yield "\n\n**📙 Loading documents done ...**\n\n"

    # Process LiteRAG embeddings
    lite_embedding_start_time = time.time()
    yield "\n\n**🔍 Loading LiteRAG embeddings ...**"
    for file_id, embedding_folder, file_path in zip(file_id_list, embedding_folder_list, file_path_list):
        if literag_index_files_decompress(embedding_folder):
            # Check if the LiteRAG index files are ready locally
            logger.info(f"LiteRAG embedding index files for {file_id} are ready.")
            yield "\n\n**🔍 LiteRAG embedding index files are ready.**"
        else:
            # Files are missing and have been cleaned up
            _document, _doc = process_pdf_file(file_path)
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder, chat_session=chat_session)
            logger.info(f"Loading LiteRAG embedding for {file_id} ...")
            yield "\n\n**🔍 Loading LiteRAG embedding ...**"
            async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                yield chunk
    time_tracking["lite_embedding_total"] = time.time() - lite_embedding_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    logger.info("LiteRAG embedding ready ...")
    yield "\n\n**🔍 LiteRAG embedding ready ...**"
    yield "</thinking>"
    yield "\n\n**🧠 Loading response ...**\n\n"

    chat_history = chat_session.chat_history
    context_chat_history = chat_history

    # Handle initial welcome message when chat history is empty
    # initial_message_start_time = time.time()
    if user_input == config["summary_wording"] or not chat_history:
        pass
        # yield "<response>"
        # yield "Hello! How can I assist you today?"
        # yield "</response>"

    # time_tracking["summary_message"] = time.time() - initial_message_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Regular chat flow - for Lite mode, we don't need to refine the user input
    question = Question(text=user_input, language=chat_session.current_language, question_type="local")

    # Get response
    response_start = time.time()
    response = await get_response(chat_session, file_path_list, question, context_chat_history, embedding_folder_list, deep_thinking=deep_thinking, stream=stream)
    answer = response[0] if isinstance(response, tuple) else response
    for chunk in answer:
        yield chunk
    time_tracking["response_generation"] = time.time() - response_start
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    yield "<appendix>"

    # For Lite mode, we have minimal sources and follow-up questions
    yield "\n\n**💬 Loading follow-up questions ...**\n\n"
    message_content = chat_session.current_message
    if isinstance(message_content, list) and len(message_content) > 0:
        message_content = message_content[0]
    
    follow_up_questions = generate_follow_up_questions(message_content, [])
    for i in range(len(follow_up_questions)):
        follow_up_questions[i] = translate_content(
            content=follow_up_questions[i],
            target_lang=chat_session.current_language,
            stream=False
        )
        # Clean up translation prefixes - apply before including in XML
        follow_up_questions[i] = clean_translation_prefix(follow_up_questions[i])

    for chunk in follow_up_questions:
        # Ensure the chunk is properly cleaned and formatted before wrapping in XML
        cleaned_chunk = chunk.strip()
        if cleaned_chunk:
            yield "<followup_question>"
            yield f"{cleaned_chunk}"
            yield "</followup_question>\n\n"
    yield "\n\n**💬 Loading follow-up questions done ...**\n\n"

    yield "\n\n**🔍 Retrieving sources ...**\n\n"
    yield "\n\n**🔍 Retrieving sources done ...**\n\n"

    yield "\n\n**🔍 Retrieving source pages ...**\n\n"
    yield "\n\n**🔍 Retrieving source pages done ...**\n\n"

    yield "\n\n**🔍 Retrieving source annotations ...**\n\n"
    yield "\n\n**🔍 Retrieving source annotations done ...**\n\n"

    yield "\n\n**🔍 Refining source pages ...**\n\n"
    yield "\n\n**🔍 Refining source pages done ...**\n\n"

    yield "\n\n**🔍 Refining source index ...**\n\n"
    yield "\n\n**🔍 Refining source index done ...**\n\n"

    yield "</appendix>"

    # Memory clean up
    _document = None
    _doc = None

    return