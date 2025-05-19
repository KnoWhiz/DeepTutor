import os
import time
from typing import Dict, Generator
import re

from pipeline.science.pipeline.utils import (
    generate_file_id,
    format_time_tracking,
    clean_translation_prefix,
    Question
)
from pipeline.science.pipeline.content_translator import (
    detect_language,
    translate_content
)
from pipeline.science.pipeline.doc_processor import (
    save_file_txt_locally,
    process_pdf_file,
    extract_document_from_file,
)
from pipeline.science.pipeline.session_manager import ChatSession
from pipeline.science.pipeline.helper.index_files_saving import (
    literag_index_files_decompress,
)
from pipeline.science.pipeline.embeddings_agent import embeddings_agent
from pipeline.science.pipeline.get_response import (
    get_response,
    generate_follow_up_questions,
)
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.sources_retrieval import get_response_source, locate_chunk_in_pdf

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
        # Ensure chunk is a string before concatenating
        if hasattr(chunk, 'content'):
            chat_session.current_message += chunk.content
        else:
            chat_session.current_message += str(chunk)

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
    # path_prefix = os.getenv("FILE_PATH_PREFIX")
    # if not path_prefix:
    #     path_prefix = ""
    # embedded_content_path = os.path.join(path_prefix, 'embedded_content')
    # embedding_folder_list = [os.path.join(embedded_content_path, file_id) for file_id in file_id_list]
    path_prefix = os.getenv("FILE_PATH_PREFIX")
    embedded_content_path = os.path.join(path_prefix, 'embedded_content/lite_mode')
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
    # yield "\n\n**🔍 Loading LiteRAG embeddings ...**"
    for file_id, embedding_folder, file_path in zip(file_id_list, embedding_folder_list, file_path_list):
        if literag_index_files_decompress(embedding_folder):
            # Check if the LiteRAG index files are ready locally
            logger.info(f"LiteRAG embedding index files for {file_id} are ready.")
            # yield "\n\n**🔍 LiteRAG embedding index files are ready.**"
        else:
            # Files are missing and have been cleaned up
            _document, _doc = process_pdf_file(file_path)
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder, chat_session=chat_session)
            logger.info(f"Loading LiteRAG embedding for {file_id} ...")
            # yield "\n\n**🔍 Loading LiteRAG embeddings ...**"
            async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                yield chunk
    time_tracking["lite_embedding_total"] = time.time() - lite_embedding_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    logger.info("LiteRAG embeddings ready ...")
    yield "\n\n**🔍 LiteRAG embeddings ready ...**"
    # yield "</thinking>"
    yield "\n\n**🧠 Loading response ...**\n\n"

    chat_history = chat_session.chat_history
    context_chat_history = chat_history

    # Load PDF content with distributed word budget
    pdf_content_loading_start = time.time()
    pdf_content = ""
    
    # Calculate word budget per file
    total_word_budget = config["basic_token_limit"] / 5
    files_count = len(file_path_list)
    words_per_file = total_word_budget // files_count if files_count > 0 else total_word_budget
    
    logger.info(f"Loading PDF content with {words_per_file} words per file from {files_count} files")
    yield "\n\n**📚 Loading PDF content with distributed word budget...**\n\n"
    
    for file_path in file_path_list:
        try:
            # Extract document from the file
            document = extract_document_from_file(file_path)
            # Extract text from the document
            file_content = ""
            for doc in document:
                if hasattr(doc, 'page_content') and doc.page_content:
                    file_content += doc.page_content.strip() + "\n"
            # Take only the words_per_file words from this file
            file_words = file_content.split()
            if len(file_words) > words_per_file:
                file_words = file_words[:words_per_file]
            file_excerpt = " ".join(file_words)
            pdf_content += f"\n\n--- Related content ---\n{file_excerpt}\n"
            logger.info(f"Added {len(file_words)} words from {file_path}")
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {str(e)}")
            yield f"\n\n**⚠️ Error loading content from {os.path.basename(file_path)}: {str(e)}**\n\n"
    
    if len(file_path_list) > 1 and user_input != config["summary_wording"]:
        # For regular queries with multiple files, we'll rely on embeddings
        # rather than including all PDF content directly
        pdf_content = ""
    
    time_tracking["pdf_content_loading"] = time.time() - pdf_content_loading_start
    logger.info(f"PDF content loading complete. Time: {format_time_tracking(time_tracking)}")
    yield "\n\n**📚 PDF content loading complete**\n\n"
    yield "</thinking>"

    # Handle initial welcome message when chat history is empty or summary is requested
    if user_input == config["summary_wording"] or not chat_history:
        # If summary is requested and we have multiple files, get_response will handle it with the special function
        # Just pass the request through to get_response
        if user_input == config["summary_wording"] and len(file_path_list) > 1:
            logger.info("Handling multiple files summary request in lite mode")
            # yield "\n\n**🧠 Analyzing multiple files and generating a comprehensive summary...**\n\n"
            # yield "</thinking>"
            # get_response will handle the multiple file summary generation
            response_start = time.time()
            question = Question(text=user_input, language=chat_session.current_language, question_type="global")
            response = await get_response(chat_session, file_path_list, question, context_chat_history, embedding_folder_list, deep_thinking=deep_thinking, stream=stream)
            answer = response[0] if isinstance(response, tuple) else response
            for chunk in answer:
                yield chunk
            time_tracking["response_generation"] = time.time() - response_start
            
            # Handle appendix with follow-up questions
            yield "<appendix>"
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

            yield "</appendix>"
            return
        else:
            # Regular summary for single file or initial message when no chat history
            # Include the PDF content in addition to user_input as refined_user_input
            refined_user_input = f"{user_input}\n\nThis is the document content: {pdf_content}"
            question = Question(text=refined_user_input, language=chat_session.current_language, question_type="local")
    else:
        # yield "</thinking>"
        refined_user_input = f"{user_input}\n\n{pdf_content}"
        # Regular chat flow - include PDF content in the user input
        question = Question(text=refined_user_input, language=chat_session.current_language, question_type="local")

    logger.info(f"Refined user input created with PDF content: {refined_user_input}")

    # time_tracking["summary_message"] = time.time() - initial_message_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get response
    response_start = time.time()
    response = await get_response(chat_session, file_path_list, question, context_chat_history, embedding_folder_list, deep_thinking=deep_thinking, stream=stream)
    answer = response[0] if isinstance(response, tuple) else response
    for chunk in answer:
        yield chunk
    time_tracking["response_generation"] = time.time() - response_start
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    yield "<appendix>"

    # Add source retrieval - similar to what's in basic and advanced modes
    yield "\n\n**🔍 Retrieving sources ...**\n\n"
    sources_start = time.time()
    # Update the embedding folder paths to include the lite_embedding subdirectory
    lite_embedding_folder_list = [os.path.join(folder, "lite_embedding") for folder in embedding_folder_list]
    
    # Ensure the markdown directories exist for storing image context and URLs
    for folder in lite_embedding_folder_list:
        markdown_dir = os.path.join(folder, "markdown")
        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir, exist_ok=True)
            logger.info(f"Created markdown directory: {markdown_dir}")
    
    sources, source_pages, refined_source_pages, refined_source_index = get_response_source(
        chat_session=chat_session,
        file_path_list=file_path_list,
        user_input=refined_user_input,
        answer=chat_session.current_message,
        chat_history=chat_history,
        embedding_folder_list=lite_embedding_folder_list
    )

    for source_key, source_value in sources.items():
        yield "<source>"
        yield "{" + str(source_key) + "}"
        yield "{" + str(source_value) + "}"
        yield "</source>"
    for source_page_key, source_page_value in source_pages.items():
        yield "<source_page>"
        yield "{" + str(source_page_key) + "}"
        yield "{" + str(source_page_value) + "}"
        yield "</source_page>"
    for refined_source_page_key, refined_source_page_value in refined_source_pages.items():
        yield "<refined_source_page>"
        yield "{" + str(refined_source_page_key) + "}"
        yield "{" + str(refined_source_page_value) + "}"
        yield "</refined_source_page>"
    for refined_source_index_key, refined_source_index_value in refined_source_index.items():
        yield "<refined_source_index>"
        yield "{" + str(refined_source_index_key) + "}"
        yield "{" + str(refined_source_index_value) + "}"
        yield "</refined_source_index>"

    time_tracking["source_retrieval"] = time.time() - sources_start
    # yield "\n\n**🔍 Retrieving sources done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    source_annotations = {}
    i = 0
    for source, index in refined_source_index.items():
        _doc = process_pdf_file(file_path_list[index-1])[1]
        # annotations, _ = get_highlight_info(_doc, [source])
        # logger.info(f"TEST: source: {source}, index: {index}, file_path: {file_path_list[refined_source_index[source]]}")
        annotations = locate_chunk_in_pdf(source, file_path_list[refined_source_index[source]])
        source_annotations[source] = annotations
        logger.info(f"For source number {i}, the annotations extraction is: {annotations}")
        i += 1
    # yield "\n\n**🔍 Retrieving source annotations done ...**\n\n"
    # logger.info(f"source_annotations: {source_annotations}")

    for source_annotations_key, source_annotations_value in source_annotations.items():
        yield "<source_annotations>"
        yield "{" + str(source_annotations_key) + "}"
        yield "{" + str(source_annotations_value) + "}"
        yield "</source_annotations>"

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

    yield "</appendix>"

    # Memory clean up
    _document = None
    _doc = None

    return