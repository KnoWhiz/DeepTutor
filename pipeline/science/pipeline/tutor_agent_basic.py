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
    Question
)
from pipeline.science.pipeline.doc_processor import (
    save_file_txt_locally,
    process_pdf_file,
    get_highlight_info,
)
from pipeline.science.pipeline.content_translator import (
    detect_language,
    translate_content
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
)
from pipeline.science.pipeline.sources_retrieval import get_response_source, locate_chunk_in_pdf
from pipeline.science.pipeline.config import load_config

import logging
logger = logging.getLogger("tutorpipeline.science.tutor_agent_basic")


async def tutor_agent_basic(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    """
    Standard tutor agent that provides comprehensive tutoring capabilities with vector-based RAG.
    Uses VectorRAG for document processing and performs source retrieval.

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

    return tutor_agent_basic_streaming_tracking(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)
    # answer = tutor_agent_basic_streaming_tracking(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)

    # # For Lite mode, we have minimal sources and follow-up questions
    # sources = {}
    # source_pages = {}
    # source_annotations = {}
    # refined_source_pages = {}
    # refined_source_index = {}
    # follow_up_questions = []

    # return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


async def tutor_agent_basic_streaming_tracking(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    async for chunk in tutor_agent_basic_streaming(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream):
        yield chunk
        # Extract text content from ChatCompletionChunk if needed
        if isinstance(chunk, str):
            chat_session.current_message += chunk
        elif hasattr(chunk, "choices") and chunk.choices:
            # Extract the actual text content from the ChatCompletionChunk
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                chat_session.current_message += delta.content
        else:
            # Fallback for other types of chunks
            try:
                chat_session.current_message += str(chunk)
            except Exception as e:
                logger.warning(f"Could not concatenate chunk to message: {e}")
                # Skip this chunk if it can't be converted to string


async def tutor_agent_basic_streaming(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    """
    Streaming tutor agent for Basic mode.
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
    summary_wording = config["summary_wording"]
    logger.info(f"Summary wording: {summary_wording}")

    file_id_list = [generate_file_id(file_path) for file_path in file_path_list]
    path_prefix = os.getenv("FILE_PATH_PREFIX")
    embedded_content_path = os.path.join(path_prefix, 'embedded_content')
    embedding_folder_list = [os.path.join(embedded_content_path, file_id) for file_id in file_id_list]

    # Compute hashed ID and prepare embedding folder
    yield "<thinking>"
    hashing_start_time = time.time()
    file_id_list = [generate_file_id(file_path) for file_path in file_path_list]
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

    # Process VectorRAG embeddings
    vectorrag_start_time = time.time()
    logger.info(f"BASIC (VectorRAG) mode for list of file ids: {file_id_list}")
    for file_id, embedding_folder, file_path in zip(file_id_list, embedding_folder_list, file_path_list):
        # Doc processing
        if vectorrag_index_files_decompress(embedding_folder):
            logger.info(f"VectorRAG index files for {file_id} are ready.")
        else:
            # Files are missing and have been cleaned up
            _document, _doc = process_pdf_file(file_path)
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder, chat_session=chat_session)
            logger.info(f"VectorRAG embedding for {file_id} ...")
            # await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
            async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                yield chunk
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            if vectorrag_index_files_compress(embedding_folder):
                logger.info(f"VectorRAG index files for {file_id} are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder, chat_session=chat_session)
                # await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
                async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                    yield chunk
                logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
                if vectorrag_index_files_compress(embedding_folder):
                    logger.info(f"VectorRAG index files for {file_id} are ready and uploaded to Azure Blob Storage.")
                else:
                    logger.info(f"Error compressing and uploading VectorRAG index files for {file_id} to Azure Blob Storage.")
    time_tracking["vectorrag_generate_embedding_total"] = time.time() - vectorrag_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    chat_history = chat_session.chat_history
    context_chat_history = chat_history

    if user_input == summary_wording or not chat_session.chat_history:
        # Handle initial welcome message when chat history is empty
        # initial_message_start_time = time.time()
        # yield "<thinking>"
        try:
            # Try to load existing document summary
            document_summary_path_list = [os.path.join(embedding_folder, "documents_summary.txt") for embedding_folder in embedding_folder_list]
            initial_message_list = []
            for document_summary_path in document_summary_path_list:
                with open(document_summary_path, "r") as f:
                    initial_message_list.append(f.read())
            # FIXME: Add a function to combine the initial messages into a single summary message
            initial_message = "\n".join(initial_message_list)
        except FileNotFoundError:
            initial_message = "Hello! How can I assist you today?"
        # yield "</thinking>"
        # yield "\n\n**Loading response ...**\n\n"

        language = detect_language(initial_message)
        if language != chat_session.current_language:
            translation_response = True
            yield "<original_response>\n\n"
            yield initial_message
            yield "</original_response>"
            yield "</thinking>"
            yield "\n\n**🧠 Loading response ...**\n\n"
            yield "<response>\n\n"        # Translate the initial message to the selected language
            answer = translate_content(
                content=initial_message,
                target_lang=chat_session.current_language,
                stream=stream
            )
            if (type(answer) is str):
                yield answer
            else:
                for chunk in answer:
                    yield chunk
            yield "</response>"
        else:
            translation_response = False
            yield "</thinking>"
            yield "\n\n**🧠 Loading response ...**\n\n"
            yield "<response>\n\n"        # Translate the initial message to the selected language
            yield initial_message
            yield "</response>"

        yield "<appendix>"
        yield "\n\n**💬 Loading follow-up questions ...**\n\n"
        follow_up_questions = generate_follow_up_questions(chat_session.current_message, [])
        for i in range(len(follow_up_questions)):
            follow_up_questions[i] = translate_content(
                content=follow_up_questions[i],
                target_lang=chat_session.current_language,
                stream=False
            )
            # Clean up translation prefixes
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

        # time_tracking["summary_message"] = time.time() - initial_message_start_time
        logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

        return

    # Regular chat flow
    # Refine user input
    yield "\n\n**🧠 Understanding the user input ...**\n\n"
    query_start = time.time()
    async for question_progress_update in get_query_helper(chat_session, user_input, context_chat_history, embedding_folder_list):
        if isinstance(question_progress_update, Question):
            # This is the final return value - a Question object
            question = question_progress_update
            logger.info(f"Received Question object from streaming function: {question}")
        elif isinstance(question_progress_update, str):
            # yield f"\n\n💬 {question_progress_update}"
            pass
        else:
            continue
    refined_user_input = question.text
    logger.info(f"Refined user input: {refined_user_input}")
    time_tracking["query_refinement"] = time.time() - query_start
    yield "\n\n**🧠 Understanding the user input done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get response
    translation_response = False
    response_start = time.time()
    response = await get_response(chat_session, file_path_list, question, context_chat_history, embedding_folder_list, deep_thinking=deep_thinking, stream=stream)
    answer = response[0] if isinstance(response, tuple) else response
    
    def needs_translation():
        """Helper function to determine if translation is needed."""
        content = extract_basic_mode_content(chat_session.current_message)[0]
        language = detect_language(content)
        return language != chat_session.current_language
        
    def is_in_thinking_mode():
        """Helper function to check if we're in thinking mode."""
        return chat_session.current_message.endswith("</thinking>") or not chat_session.current_message
    
    if deep_thinking is False:
        yield "</thinking>"
        # yield "\n\n**💡 Loading the response ...**\n\n"
    else:
        for chunk in answer:
            # Process chunks without </think> tag
            if "</think>" not in chunk:
                # Handle response opening tag
                if "<response>" in chunk:
                    translation_needed = needs_translation()
                    
                    if not translation_needed:
                        if is_in_thinking_mode():
                            yield chunk  # First response chunk
                        else:
                            # yield chunk.replace("<response>", "")  # Continuation chunk
                            yield chunk
                    else:
                        yield chunk.replace("<response>", "<original_response>")

                    if chat_session.question.question_type == "image":
                        yield f"\n\n![]({chat_session.question.image_url})\n"
                
                # Handle response closing tag
                elif "</response>" in chunk:
                    translation_needed = needs_translation()

                    if not translation_needed:
                        yield chunk
                    else:
                        # End original response and prepare for translation
                        yield chunk.replace("</response>", "</original_response>")
                        yield "</thinking>"
                        # yield "\n\n**💡 Loading the response ...**\n\n"
                        translation_response = True

                # Handle regular content
                else:
                    yield chunk

            # Handle </think> tag (end of thinking section)
            else:
                translation_response = needs_translation()

                if not translation_response:
                    chunk = chunk.replace("</think>", "")
                    # yield "\n\n**💡 Loading the response ...**\n\n"
                    yield chunk
                    yield "</think>"
                    yield "</thinking>"
                # else:
                #     yield "<response>"
                #     yield translation_response
                #     yield "</response>"
    
    time_tracking["response_generation"] = time.time() - response_start
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Refine and translate the answer to the selected language
    content=extract_basic_mode_content(chat_session.current_message)[0]
    language = detect_language(content)
    if language != chat_session.current_language:
        translation_start = time.time()
        answer = translate_content(
            content=content,
            target_lang=chat_session.current_language,
            stream=stream
        )
        # if answer != content:
        yield "<response>\n\n"
        if chat_session.question.question_type == "image":
            yield f"\n\n![]({chat_session.question.image_url})\n"
        if (type(answer) is str):
            yield answer
        else:
            for chunk in answer:
                yield chunk
        yield "</response>"
        # yield "\n\n**💡 Loading the response done ...**\n\n"
        time_tracking["translation"] = time.time() - translation_start
        logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get sources
    yield "<appendix>"
    sources = {}
    source_pages = {}
    refined_source_pages = {}
    refined_source_index = {}
    sources_start = time.time()
    yield "\n\n**🔍 Retrieving sources ...**\n\n"
    sources, source_pages, refined_source_pages, refined_source_index = get_response_source(
        chat_session=chat_session,
        file_path_list=file_path_list,
        user_input=refined_user_input,
        answer=chat_session.current_message,
        chat_history=chat_history,
        embedding_folder_list=embedding_folder_list
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

    # Process image sources
    # yield "\n\n**📊 Processing image sources ...**\n\n"
    images_processing_start = time.time()
    image_url_list = []
    for source, index, page in zip(refined_source_index.keys(), refined_source_index.values(), refined_source_pages.values()):
        logger.info(f"TEST: source: {source}, index: {index}, page: {page}")
        if source.startswith("https://knowhiztutorrag.blob"):
            image_url = source
            image_url_list.append(image_url)
    time_tracking["image_processing"] = time.time() - images_processing_start
    # yield "\n\n**📊 Processing image sources done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Append images URL in markdown format to the end of the answer
    annotations_start = time.time()
    if image_url_list:
        for image_url in image_url_list:
            if image_url:
                yield "\n"
                yield f"![]({image_url})"

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
    time_tracking["annotations"] = time.time() - annotations_start
    # yield "\n\n**🔍 Retrieving source annotations done ...**\n\n"
    # logger.info(f"source_annotations: {source_annotations}")
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    for source_annotations_key, source_annotations_value in source_annotations.items():
        yield "<source_annotations>"
        yield "{" + str(source_annotations_key) + "}"
        yield "{" + str(source_annotations_value) + "}"
        yield "</source_annotations>"

    # Generate follow-up questions
    yield "\n\n**💬 Loading follow-up questions ...**\n\n"
    followup_start = time.time()
    follow_up_questions = generate_follow_up_questions(answer, chat_history)
    for i in range(len(follow_up_questions)):
        follow_up_questions[i] = translate_content(
            content=follow_up_questions[i],
            target_lang=chat_session.current_language,
            stream=False
        )
        # Clean up translation prefixes
        follow_up_questions[i] = clean_translation_prefix(follow_up_questions[i])

    for chunk in follow_up_questions:
        # Ensure the chunk is properly cleaned and formatted before wrapping in XML
        cleaned_chunk = chunk.strip()
        if cleaned_chunk:
            yield "<followup_question>"
            yield f"{cleaned_chunk}"
            yield "</followup_question>\n\n"
    time_tracking["followup_questions"] = time.time() - followup_start
    yield "\n\n**💬 Loading follow-up questions done ...**\n\n"
    yield "</appendix>"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Memory clean up
    _document = None
    _doc = None

    return