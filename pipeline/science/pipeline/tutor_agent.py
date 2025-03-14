import os
import json
import time
from typing import Dict, Generator
import re

from pipeline.science.pipeline.utils import (
    translate_content,
    generate_file_id,
    format_time_tracking,
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

import logging
logger = logging.getLogger("tutorpipeline.science.tutor_agent")


def extract_lite_mode_content(message_content):
    """
    Extracts structured content from the lite mode message content.
    
    Args:
        message_content: The complete message string from chat_session.current_message
        
    Returns:
        Tuple containing (answer, sources, source_pages, source_annotations, 
                         refined_source_pages, refined_source_index, follow_up_questions)
    """
    # Default empty structures for lite mode
    sources = {}
    source_pages = {}
    source_annotations = {}
    refined_source_pages = {}
    refined_source_index = {}
    follow_up_questions = []
    
    # Extract the main answer (content between <response> tags)
    answer = ""
    response_match = re.search(r'<response>(.*?)</response>', message_content, re.DOTALL)
    if response_match:
        answer = response_match.group(1).strip()
    
    # Extract follow-up questions (content between <followup_question> tags)
    followup_matches = re.finditer(r'<followup_question>(.*?)</followup_question>', message_content, re.DOTALL)
    for match in followup_matches:
        question = match.group(1).strip()
        if question:
            follow_up_questions.append(question)
    
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


def extract_basic_mode_content(message_content):
    """
    Extracts structured content from the basic mode message content.
    
    Args:
        message_content: The complete message string from chat_session.current_message
        
    Returns:
        Tuple containing (answer, sources, source_pages, source_annotations, 
                         refined_source_pages, refined_source_index, follow_up_questions)
    """
    # Default empty structures for basic mode
    sources = {}
    source_pages = {}
    source_annotations = {}
    refined_source_pages = {}
    refined_source_index = {}
    follow_up_questions = []
    
    # Extract the main answer (content between <response> tags)
    answer = ""
    response_match = re.search(r'<response>(.*?)</response>', message_content, re.DOTALL)
    if response_match:
        answer = response_match.group(1).strip()
    
    # Extract follow-up questions (content between <followup_question> tags)
    followup_matches = re.finditer(r'<followup_question>(.*?)</followup_question>', message_content, re.DOTALL)
    for match in followup_matches:
        question = match.group(1).strip()
        if question:
            follow_up_questions.append(question)
    
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


def extract_advanced_mode_content(message_content):
    """
    Extracts structured content from the advanced mode message content.
    
    Args:
        message_content: The complete message string from chat_session.current_message
        
    Returns:
        Tuple containing (answer, sources, source_pages, source_annotations, 
                         refined_source_pages, refined_source_index, follow_up_questions)
    """
    # Default empty structures for advanced mode
    sources = {}
    source_pages = {}
    source_annotations = {}
    refined_source_pages = {}
    refined_source_index = {}
    follow_up_questions = []
    
    # Extract the main answer (content between <response> tags)
    answer = ""
    response_match = re.search(r'<response>(.*?)</response>', message_content, re.DOTALL)
    if response_match:
        answer = response_match.group(1).strip()
    
    # Extract follow-up questions (content between <followup_question> tags)
    followup_matches = re.finditer(r'<followup_question>(.*?)</followup_question>', message_content, re.DOTALL)
    for match in followup_matches:
        question = match.group(1).strip()
        if question:
            follow_up_questions.append(question)
    
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


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
        return error_message, {}, {}, {}, {}, {}, []


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

    # Handle initial welcome message when chat history is empty
    initial_message_start_time = time.time()
    if user_input == "Can you give me a summary of this document?" or not chat_session.chat_history:
        initial_message = "Hello! How can I assist you today?"
        
        answer = initial_message
        # Translate the initial message to the selected language
        answer = translate_content(
            content=answer,
            target_lang=chat_session.current_language
        )
        sources = {}  # Return empty dictionary for sources
        source_pages = {}
        source_annotations = {}
        refined_source_pages = {}
        refined_source_index = {}
        follow_up_questions = []

        return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions
    time_tracking["summary_message"] = time.time() - initial_message_start_time

    answer = tutor_agent_lite_streaming_tracking(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)

    # For Lite mode, we have minimal sources and follow-up questions
    sources = {}
    source_pages = {}
    source_annotations = {}
    refined_source_pages = {}
    refined_source_index = {}
    follow_up_questions = []
    
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


async def tutor_agent_lite_streaming_tracking(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    async for chunk in tutor_agent_lite_streaming(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream):
        yield chunk
        chat_session.current_message += chunk

    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions = extract_lite_mode_content(chat_session.current_message)
    logger.info(f"Extracted answer: {answer}")
    logger.info(f"Extracted sources: {sources}")
    logger.info(f"Extracted source pages: {source_pages}")
    logger.info(f"Extracted source annotations: {source_annotations}")
    logger.info(f"Extracted refined source pages: {refined_source_pages}")
    logger.info(f"Extracted refined source index: {refined_source_index}")
    logger.info(f"Extracted follow-up questions: {follow_up_questions}")
    logger.info(f"Current message: {chat_session.current_message}")


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

    # Compute hashed ID and prepare embedding folder
    yield "<thinking>"
    yield "Processing documents ...\n\n"
    hashing_start_time = time.time()
    file_id_list = [generate_file_id(file_path) for file_path in file_path_list]
    embedding_folder_list = [os.path.join("embedded_content", file_id) for file_id in file_id_list]
    logger.info(f"Embedding folder: {embedding_folder_list}")
    if not os.path.exists("embedded_content"):
        os.makedirs("embedded_content")
    for embedding_folder in embedding_folder_list:
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)
    time_tracking["file_hashing_setup_dirs"] = time.time() - hashing_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Save the file txt content locally
    save_file_start_time = time.time()
    filename_list = [os.path.basename(file_path) for file_path in file_path_list]
    for file_path, filename in zip(file_path_list, filename_list):
        save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
    time_tracking["file_loading_save_text"] = time.time() - save_file_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    yield "\n\n**Processing documents done ...**\n\n"

    # Process LiteRAG embeddings
    lite_embedding_start_time = time.time()
    yield "Generating LiteRAG embeddings ...\n\n"
    for file_id, embedding_folder, file_path in zip(file_id_list, embedding_folder_list, file_path_list):
        if literag_index_files_decompress(embedding_folder):
            # Check if the LiteRAG index files are ready locally
            logger.info(f"LiteRAG embedding index files for {file_id} are ready.")
            yield f"LiteRAG embedding index files for {file_id} are ready.\n\n"
        else:
            # Files are missing and have been cleaned up
            _document, _doc = process_pdf_file(file_path)
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            logger.info(f"Generating LiteRAG embedding for {file_id} ...")
            yield f"Generating LiteRAG embedding for {file_id} ...\n\n"
            async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                yield chunk
    time_tracking["lite_embedding_total"] = time.time() - lite_embedding_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    logger.info("LiteRAG embedding done ...")
    yield "\n\n**LiteRAG embedding done ...**\n\n"
    yield "</thinking>"

    chat_history = chat_session.chat_history
    context_chat_history = chat_history

    # Handle initial welcome message when chat history is empty
    initial_message_start_time = time.time()
    if user_input == "Can you give me a summary of this document?" or not chat_history:
        yield "<response>"
        yield "Hello! How can I assist you today?"
        yield "</response>"

    time_tracking["summary_message"] = time.time() - initial_message_start_time
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
    yield "\n\n**Generating follow-up questions ...**\n\n"
    follow_up_questions = generate_follow_up_questions(chat_session.current_message, [])
    for chunk in follow_up_questions:
        # The content should be easy to extract as XML formats
        yield "<followup_question>"
        yield f"\n{chunk}"
        yield "</followup_question>"
        yield "\n\n"
    yield "\n\n**Generating follow-up questions done ...**\n\n"

    yield "\n\n**Retrieving sources ...**\n\n"
    yield "\n\nRetrieving sources done ...**\n\n"

    yield "\n\n**Retrieving source pages ...**\n\n"
    yield "\n\nRetrieving source pages done ...**\n\n"

    yield "\n\n**Retrieving source annotations ...**\n\n"
    yield "\n\nRetrieving source annotations done ...**\n\n"

    yield "\n\n**Refining source pages ...**\n\n"
    yield "\n\nRefining source pages done ...**\n\n"

    yield "\n\n**Refining source index ...**\n\n"
    yield "\n\nRefining source index done ...**\n\n"

    yield "</appendix>"

    # Memory clean up
    _document = None
    _doc = None

    return


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

    answer = tutor_agent_basic_streaming_tracking(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)

    # For Lite mode, we have minimal sources and follow-up questions
    sources = {}
    source_pages = {}
    source_annotations = {}
    refined_source_pages = {}
    refined_source_index = {}
    follow_up_questions = []
    
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


async def tutor_agent_basic_streaming_tracking(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    async for chunk in tutor_agent_basic_streaming(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream):
        yield chunk
        chat_session.current_message += chunk

    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions = extract_basic_mode_content(chat_session.current_message)
    logger.info(f"Extracted answer: {answer}")
    logger.info(f"Extracted sources: {sources}")
    logger.info(f"Extracted source pages: {source_pages}")
    logger.info(f"Extracted source annotations: {source_annotations}")
    logger.info(f"Extracted refined source pages: {refined_source_pages}")
    logger.info(f"Extracted refined source index: {refined_source_index}")
    logger.info(f"Extracted follow-up questions: {follow_up_questions}")
    logger.info(f"Current message: {chat_session.current_message}")


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
    embedding_folder_list = [os.path.join("embedded_content", file_id) for file_id in file_id_list]

    # Compute hashed ID and prepare embedding folder
    yield "<thinking>"
    hashing_start_time = time.time()
    file_id_list = [generate_file_id(file_path) for file_path in file_path_list]
    embedding_folder_list = [os.path.join("embedded_content", file_id) for file_id in file_id_list]
    logger.info(f"Embedding folder: {embedding_folder_list}")
    if not os.path.exists("embedded_content"):
        os.makedirs("embedded_content")
    for embedding_folder in embedding_folder_list:
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)
    time_tracking["file_hashing_setup_dirs"] = time.time() - hashing_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Save the file txt content locally
    save_file_start_time = time.time()
    filename_list = [os.path.basename(file_path) for file_path in file_path_list]
    for file_path, filename in zip(file_path_list, filename_list):
        save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
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
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            logger.info(f"VectorRAG embedding for {file_id} ...")
            # await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
            async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                yield chunk
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            if vectorrag_index_files_compress(embedding_folder):
                logger.info(f"VectorRAG index files for {file_id} are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
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

    # # Handle initial welcome message when chat history is empty
    # initial_message_start_time = time.time()
    # if user_input == "Can you give me a summary of this document?" or not chat_history:
    #     try:
    #         # Try to load existing document summary
    #         document_summary_path_list = [os.path.join(embedding_folder, "documents_summary.txt") for embedding_folder in embedding_folder_list]
    #         initial_message_list = []
    #         for document_summary_path in document_summary_path_list:
    #             with open(document_summary_path, "r") as f:
    #                 initial_message_list.append(f.read())
    #         # FIXME: Add a function to combine the initial messages into a single summary message
    #         initial_message = "\n".join(initial_message_list)
    #     except FileNotFoundError:
    #         initial_message = "Hello! How can I assist you today?"

    #     yield "</thinking>"

    #     yield "<response>"

    #     answer = initial_message
    #     # Translate the initial message to the selected language
    #     answer = translate_content(
    #         content=answer,
    #         target_lang=chat_session.current_language
    #     )

    #     yield "</response>"

    #     yield "<appendix>"

    #     sources = {}  # Return empty dictionary for sources
    #     source_pages = {}
    #     source_annotations = {}
    #     refined_source_pages = {}
    #     refined_source_index = {}
    #     yield "\n\n**Generating follow-up questions ...**\n\n"
    #     follow_up_questions = generate_follow_up_questions(answer, [])

    #     for i in range(len(follow_up_questions)):
    #         follow_up_questions[i] = translate_content(
    #             content=follow_up_questions[i],
    #             target_lang=chat_session.current_language
    #         )
    #     for chunk in follow_up_questions:
    #         # The content should be easy to extract as XML formats
    #         yield "<followup_question>"
    #         yield f"\n{chunk}"
    #         yield "</followup_question>"
    #         yield "\n\n"

    #     yield "</appendix>"
    #     return
    
    if user_input == summary_wording or not chat_session.chat_history:
        # Handle initial welcome message when chat history is empty
        initial_message_start_time = time.time()
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
        yield "</thinking>"

        yield "<response>"
        answer = initial_message
        # Translate the initial message to the selected language
        answer = translate_content(
            content=answer,
            target_lang=chat_session.current_language
        )
        yield "\n\n"
        yield answer
        yield "</response>"

        yield "<appendix>"
        yield "\n\n**Generating follow-up questions ...**\n\n"
        follow_up_questions = generate_follow_up_questions(answer, [])
        for chunk in follow_up_questions:
            yield "<followup_question>"
            yield f"\n{chunk}"
            yield "</followup_question>"
            yield "\n\n"
        yield "\n\n**Generating follow-up questions done ...**\n\n"
        yield "</appendix>"

        time_tracking["summary_message"] = time.time() - initial_message_start_time
        logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

        return

    # Regular chat flow
    # Refine user input
    yield "\n\n**Understanding the user input ...**\n\n"
    query_start = time.time()
    question = get_query_helper(chat_session, user_input, context_chat_history, embedding_folder_list)
    refined_user_input = question.text
    logger.info(f"Refined user input: {refined_user_input}")
    time_tracking["query_refinement"] = time.time() - query_start
    yield "\n\n**Understanding the user input done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get response
    yield "\n\n**Generating the response ...**\n\n"
    response_start = time.time()
    response = await get_response(chat_session, file_path_list, question, context_chat_history, embedding_folder_list, deep_thinking=deep_thinking, stream=stream)
    answer = response[0] if isinstance(response, tuple) else response
    if deep_thinking is False:
        yield "</thinking>"
    else:
        for chunk in answer:
            if "</think>" not in chunk:
                if "<response>" in chunk:
                    yield "\n\n**Here is the final response**\n\n"
                yield chunk
            else:   # If the chunk contains "</think>", it means the response is done
                yield chunk
                yield "</thinking>"
    time_tracking["response_generation"] = time.time() - response_start
    yield "\n\n**Generating the response done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get sources
    yield "<appendix>"
    sources = {}
    source_pages = {}
    refined_source_pages = {}
    refined_source_index = {}
    sources_start = time.time()
    yield "\n\n**Retrieving sources ...**\n\n"
    sources, source_pages, refined_source_pages, refined_source_index = get_response_source(
        mode=chat_session.mode,
        file_path_list=file_path_list,
        user_input=refined_user_input,
        answer=chat_session.current_message,
        chat_history=chat_history,
        embedding_folder_list=embedding_folder_list
    )
    time_tracking["source_retrieval"] = time.time() - sources_start
    yield "\n\n**Retrieving sources done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Process image sources
    yield "\n\n**Processing image sources ...**\n\n"
    images_processing_start = time.time()
    image_url_list = []
    for source, index, page in zip(refined_source_index.keys(), refined_source_index.values(), refined_source_pages.values()):
        logger.info(f"TEST: source: {source}, index: {index}, page: {page}")
        if source.startswith("https://knowhiztutorrag.blob"):
            image_url = source
            image_url_list.append(image_url)
    time_tracking["image_processing"] = time.time() - images_processing_start
    yield "\n\n**Processing image sources done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # # Refine and translate the answer to the selected language
    # # FIXME:Only if currently language in not the same as the selected language
    # yield "Translating the answer to the selected language ...\n\n"
    # translation_start = time.time()
    # translated_answer = translate_content(
    #     content=chat_session.current_message,
    #     target_lang=chat_session.current_language
    # )
    # yield "<translated_answer>"
    # yield f"\n{translated_answer}\n"
    # yield "</translated_answer>"
    # time_tracking["translation"] = time.time() - translation_start
    # yield "Translating the answer to the selected language done ...\n\n"
    # logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Append images URL in markdown format to the end of the answer
    annotations_start = time.time()
    if image_url_list:
        for image_url in image_url_list:
            if image_url:
                yield "\n"
                yield f"![]({image_url})"

    yield "\n\n**Retrieving source annotations ...**\n\n"
    source_annotations = {}
    for source, index in refined_source_index.items():
        _doc = process_pdf_file(file_path_list[index-1])[1]
        annotations, _ = get_highlight_info(_doc, [source])
        source_annotations[source] = annotations
    time_tracking["annotations"] = time.time() - annotations_start
    yield "\n\n**Retrieving source annotations done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Generate follow-up questions
    yield "\n\n**Generating follow-up questions ...**\n\n"
    followup_start = time.time()
    follow_up_questions = generate_follow_up_questions(answer, chat_history)
    for i in range(len(follow_up_questions)):
        follow_up_questions[i] = translate_content(
            content=follow_up_questions[i],
            target_lang=chat_session.current_language
        )
    for chunk in follow_up_questions:
        yield "<followup_question>"
        yield f"\n{chunk}"
        yield "</followup_question>"
        yield "\n\n"
    time_tracking["followup_questions"] = time.time() - followup_start
    yield "\n\n**Generating follow-up questions done ...**\n\n"
    yield "</appendix>"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Memory clean up 
    _document = None
    _doc = None
    
    return


async def tutor_agent_advanced(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    """
    Advanced tutor agent that provides sophisticated tutoring capabilities with graph-based RAG.
    Uses GraphRAG for document processing and performs enhanced source retrieval.
    """
    if time_tracking is None:
        time_tracking = {}

    # Handle initial welcome message when chat history is empty
    initial_message_start_time = time.time()
    if user_input == "Can you give me a summary of this document?" or not chat_session.chat_history:
        initial_message = "Hello! How can I assist you today?"
        
        answer = initial_message
        # Translate the initial message to the selected language
        answer = translate_content(
            content=answer,
            target_lang=chat_session.current_language
        )
        sources = {}  # Return empty dictionary for sources
        source_pages = {}
        source_annotations = {}
        refined_source_pages = {}
        refined_source_index = {}
        follow_up_questions = []

        return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions
    time_tracking["summary_message"] = time.time() - initial_message_start_time

    answer = tutor_agent_advanced_streaming_tracking(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream)

    # For Advanced mode, we have minimal sources and follow-up questions
    sources = {}
    source_pages = {}
    source_annotations = {}
    refined_source_pages = {}
    refined_source_index = {}
    follow_up_questions = []
    
    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions


async def tutor_agent_advanced_streaming_tracking(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    async for chunk in tutor_agent_advanced_streaming(chat_session, file_path_list, user_input, time_tracking, deep_thinking, stream):
        yield chunk
        chat_session.current_message += chunk

    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions = extract_advanced_mode_content(chat_session.current_message)
    logger.info(f"Extracted answer: {answer}")
    logger.info(f"Extracted sources: {sources}")
    logger.info(f"Extracted source pages: {source_pages}")
    logger.info(f"Extracted source annotations: {source_annotations}")
    logger.info(f"Extracted refined source pages: {refined_source_pages}")
    logger.info(f"Extracted refined source index: {refined_source_index}")
    logger.info(f"Extracted follow-up questions: {follow_up_questions}")
    logger.info(f"Current message: {chat_session.current_message}")


async def tutor_agent_advanced_streaming(chat_session: ChatSession, file_path_list, user_input, time_tracking=None, deep_thinking=True, stream=False):
    """
    Streaming tutor agent for Advanced mode.

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

    # Compute hashed ID and prepare embedding folder
    yield "<thinking>"
    yield "\n\n**Processing documents ...**\n\n"
    hashing_start_time = time.time()
    file_id_list = [generate_file_id(file_path) for file_path in file_path_list]
    embedding_folder_list = [os.path.join("embedded_content", file_id) for file_id in file_id_list]
    logger.info(f"Embedding folder: {embedding_folder_list}")
    if not os.path.exists("embedded_content"):
        os.makedirs("embedded_content")
    for embedding_folder in embedding_folder_list:
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)
    time_tracking["file_hashing_setup_dirs"] = time.time() - hashing_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Save the file txt content locally
    save_file_start_time = time.time()
    filename_list = [os.path.basename(file_path) for file_path in file_path_list]
    for file_path, filename in zip(file_path_list, filename_list):
        save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
    time_tracking["file_loading_save_text"] = time.time() - save_file_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    yield "\n\n**Processing documents done ...**\n\n"

    # Process GraphRAG embeddings
    graphrag_start_time = time.time()
    yield "\n\n**Generating GraphRAG embeddings ...**\n\n"
    logger.info(f"Advanced (GraphRAG) mode for list of file ids: {file_id_list}")
    for file_id, embedding_folder, file_path in zip(file_id_list, embedding_folder_list, file_path_list):
        if graphrag_index_files_decompress(embedding_folder):
            logger.info(f"GraphRAG index files for {file_id} are ready.")
            yield f"\n\n**GraphRAG index files for {file_id} are ready.**\n\n"
        else:
            # Files are missing and have been cleaned up
            _document, _doc = process_pdf_file(file_path)
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            logger.info(f"GraphRAG embedding for {file_id} ...")
            yield f"\n\n**Generating GraphRAG embedding for {file_id} ...**\n\n"
            # await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
            async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                yield chunk
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            if graphrag_index_files_compress(embedding_folder):
                logger.info(f"GraphRAG index files for {file_id} are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                yield f"\n\n**Retrying GraphRAG embedding for {file_id} ...**\n\n"
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
                # await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
                async for chunk in embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder):
                    yield chunk
                logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
                if graphrag_index_files_compress(embedding_folder):
                    logger.info(f"GraphRAG index files for {file_id} are ready and uploaded to Azure Blob Storage.")
                    # yield f"\n\n**GraphRAG index files for {file_id} are ready and uploaded to Azure Blob Storage.**\n\n"
                else:
                    logger.info(f"Error compressing and uploading GraphRAG index files for {file_id} to Azure Blob Storage.")
                    # yield f"\n\n**Error compressing and uploading GraphRAG index files for {file_id} to Azure Blob Storage.**\n\n"
    time_tracking["graphrag_generate_embedding_total"] = time.time() - graphrag_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    yield "\n\n**GraphRAG embedding done ...**\n\n"

    chat_history = chat_session.chat_history
    context_chat_history = chat_history

    # Handle initial welcome message when chat history is empty
    initial_message_start_time = time.time()
    if user_input == "Can you give me a summary of this document?" or not chat_history:
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

        yield "</thinking>"

        yield "<response>"
        
        answer = initial_message
        # Translate the initial message to the selected language
        answer = translate_content(
            content=answer,
            target_lang=chat_session.current_language
        )
        yield answer
        
        yield "</response>"

        yield "<appendix>"

        yield "\n\n**Generating follow-up questions ...**\n\n"
        follow_up_questions = generate_follow_up_questions(answer, [])

        for i in range(len(follow_up_questions)):
            follow_up_questions[i] = translate_content(
                content=follow_up_questions[i],
                target_lang=chat_session.current_language
            )
        for chunk in follow_up_questions:
            # The content should be easy to extract as XML formats
            yield "<followup_question>"
            yield f"\n{chunk}"
            yield "</followup_question>"
            yield "\n\n"
        
        yield "\n\n**Generating follow-up questions done ...**\n\n"

        yield "</appendix>"
        return

    time_tracking["summary_message"] = time.time() - initial_message_start_time
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Regular chat flow
    # Refine user input
    yield "\n\n**Understanding the user input ...**\n\n"
    query_start = time.time()
    question = get_query_helper(chat_session, user_input, context_chat_history, embedding_folder_list)
    refined_user_input = question.text
    logger.info(f"Refined user input: {refined_user_input}")
    time_tracking["query_refinement"] = time.time() - query_start
    yield "\n\n**Understanding the user input done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    # yield "</thinking>"

    # Get response
    yield "\n\n**Generating the response ...**\n\n"
    response_start = time.time()
    response = await get_response(chat_session, file_path_list, question, context_chat_history, embedding_folder_list, deep_thinking=deep_thinking, stream=stream)
    answer = response[0] if isinstance(response, tuple) else response

    if deep_thinking is False:
        yield "</thinking>"
    else:
        for chunk in answer:
            if "</think>" not in chunk:
                if "<response>" in chunk:
                    yield "\n\n**Here is the final response**\n\n"
                yield chunk
            else:   # If the chunk contains "</think>", it means the response is done
                yield chunk
                yield "</thinking>"
    # if thinking_model is not True:
    #     yield "</thinking>"
    time_tracking["response_generation"] = time.time() - response_start
    yield "\n\n**Generating the response done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get sources
    yield "<appendix>"
    sources = {}
    source_pages = {}
    refined_source_pages = {}
    refined_source_index = {}
    sources_start = time.time()
    yield "\n\n**Retrieving sources ...**\n\n"
    sources, source_pages, refined_source_pages, refined_source_index = get_response_source(
        mode=chat_session.mode,
        file_path_list=file_path_list,
        user_input=refined_user_input,
        answer=chat_session.current_message,
        chat_history=chat_history,
        embedding_folder_list=embedding_folder_list
    )
    time_tracking["source_retrieval"] = time.time() - sources_start
    yield "\n\n**Retrieving sources done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Process image sources
    yield "\n\n**Processing image sources ...**\n\n"
    images_processing_start = time.time()
    image_url_list = []
    for source, index, page in zip(refined_source_index.keys(), refined_source_index.values(), refined_source_pages.values()):
        logger.info(f"TEST: source: {source}, index: {index}, page: {page}")
        if source.startswith("https://knowhiztutorrag.blob"):
            image_url = source
            image_url_list.append(image_url)
    time_tracking["image_processing"] = time.time() - images_processing_start
    yield "\n\n**Processing image sources done ...**\n\n"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Refine and translate the answer to the selected language
    translation_start = time.time()
    answer = translate_content(
        content=answer,
        target_lang=chat_session.current_language
    )
    time_tracking["translation"] = time.time() - translation_start
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Append images URL in markdown format to the end of the answer
    annotations_start = time.time()
    if image_url_list:
        for image_url in image_url_list:
            if image_url:
                # answer += "\n"
                yield "\n"
                # answer += f"![]({image_url})"
                yield f"![]({image_url})"

    source_annotations = {}
    for source, index in refined_source_index.items():
        _doc = process_pdf_file(file_path_list[index-1])[1]
        annotations, _ = get_highlight_info(_doc, [source])
        source_annotations[source] = annotations
    time_tracking["annotations"] = time.time() - annotations_start
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Generate follow-up questions
    yield "\n\n**Generating follow-up questions ...**\n\n"
    followup_start = time.time()
    follow_up_questions = generate_follow_up_questions(chat_session.current_message, chat_history)
    for i in range(len(follow_up_questions)):
        follow_up_questions[i] = translate_content(
            content=follow_up_questions[i],
            target_lang=chat_session.current_language
        )
    for chunk in follow_up_questions:
        yield "<followup_question>"
        yield f"\n{chunk}"
        yield "</followup_question>"
        yield "\n\n"
    time_tracking["followup_questions"] = time.time() - followup_start
    yield "\n\n**Generating follow-up questions done ...**\n\n"
    yield "</appendix>"
    logger.info(f"List of file ids: {file_id_list}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Memory clean up 
    _document = None
    _doc = None
    
    return