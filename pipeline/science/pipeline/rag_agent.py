import os
import re
import fitz
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    get_llm,
    responses_refine,
    count_tokens,
    replace_latex_formulas,
    generators_list_stream_response,
    Question
)
from pipeline.science.pipeline.embeddings import (
    get_embedding_models,
    load_embeddings,
)
from pipeline.science.pipeline.content_translator import (
    detect_language,
    translate_content
)
from pipeline.science.pipeline.inference import deep_inference_agent
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.get_rag_response import (
    get_embedding_folder_rag_response, 
    get_db_rag_response
)
from pipeline.science.pipeline.images_understanding import (
    aggregate_image_contexts_to_urls, 
    create_image_context_embeddings_db, 
    analyze_image
)

import logging
logger = logging.getLogger("tutorpipeline.science.rag_agent")


def get_page_content_from_file(file_path_list, file_index, page_number):
    """
    Get the content of a specific page from a specific file.
    
    Args:
        file_path_list (list): List of file paths
        file_index (int): Index of the file in file_path_list (0-based)
        page_number (int): Page number to extract (0-based)
        
    Returns:
        str: The text content of the specified page, or empty string if not found
        
    Raises:
        IndexError: If file_index is out of range
        ValueError: If page_number is invalid
        FileNotFoundError: If the file doesn't exist
        Exception: For other PDF processing errors
    """
    try:
        # Validate file_index
        if file_index < 0 or file_index >= len(file_path_list):
            raise IndexError(f"File index {file_index} is out of range. Available files: 0-{len(file_path_list)-1}")
        
        file_path = file_path_list[file_index]
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Open the PDF file
        doc = fitz.open(file_path)
        
        # Validate page_number
        if page_number < 0 or page_number >= len(doc):
            doc.close()
            raise ValueError(f"Page number {page_number} is out of range. Available pages: 0-{len(doc)-1}")
        
        # Extract page content
        page = doc[page_number]
        page_content = page.get_text()
        
        # Close the document
        doc.close()
        
        logger.info(f"Successfully extracted content from file {file_index} ({os.path.basename(file_path)}), page {page_number}")
        return page_content
        
    except Exception as e:
        logger.exception(f"Error extracting page content from file {file_index}, page {page_number}: {str(e)}")
        raise


async def get_rag_context(chat_session: ChatSession, file_path_list, question: Question, chat_history, embedding_folder_list, deep_thinking = True, stream=False, context=""):
    config = load_config()
    # Add the context to the user input to improve the retrieval quality
    user_input = question.text + "\n\n" + context

    # Handle Basic mode and Advanced mode
    if chat_session.mode == ChatMode.BASIC or chat_session.mode == ChatMode.ADVANCED:
        logger.info(f"Current mode is {chat_session.mode}")
        try:
            logger.info(f"Loading markdown embeddings from {[os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]}")
            markdown_embedding_folder_list = [os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]
            db = load_embeddings(markdown_embedding_folder_list, 'default')
        except Exception as e:
            logger.exception(f"Failed to load markdown embeddings for deep thinking mode: {str(e)}")
            db = load_embeddings(embedding_folder_list, 'default')
    # elif chat_session.mode == ChatMode.LITE:
    # Handle Lite mode in other cases
    else:
        logger.info(f"Current mode is {chat_session.mode}")
        actual_embedding_folder_list = [os.path.join(embedding_folder, 'lite_embedding') for embedding_folder in embedding_folder_list]
        logger.info(f"actual_embedding_folder_list in get_rag_context: {actual_embedding_folder_list}")
        db = load_embeddings(actual_embedding_folder_list, 'lite')

    # Load config for deep thinking mode
    config = load_config()
    if chat_session.mode == ChatMode.LITE:
        token_limit = config["basic_token_limit"]
    else:
        token_limit = config["inference_token_limit"]
    chat_history_string = truncate_chat_history(chat_history, token_limit=token_limit)
    user_input_string = str(user_input + "\n\n" + question.special_context)
    rag_user_input_string = str(user_input + "\n\n" + question.special_context + "\n\n" + str(question.answer_planning))
    logger.info(f"rag_user_input_string: {rag_user_input_string}")
    # Get relevant chunks for question with scores
    # First retrieve more candidates than needed to ensure we have enough after filtering
    filter_min_length = 50
    fetch_k = max(config['retriever']['k'] * 2, 20)  # Fetch 3x more to ensure enough pass the filter

    all_chunks_with_scores = db.similarity_search_with_score(rag_user_input_string, k=fetch_k)

    logger.info(f"TEST: all_chunks_with_scores: {all_chunks_with_scores}")

    # Filter chunks by length > 50
    filtered_chunks_with_scores = [
        (chunk, score) for chunk, score in all_chunks_with_scores 
        if len(chunk.page_content) > filter_min_length
    ]

    # # Sort by score (lowest score is better in many embeddings) and take top k
    # question_chunks_with_scores = sorted(
    #     filtered_chunks_with_scores, 
    #     key=lambda x: x[1]
    # )[:config['retriever']['k']]

    question_chunks_with_scores = filtered_chunks_with_scores

    # The total list of sources chunks
    sources_chunks = []
    # From the highest score to the lowest score, until the total tokens exceed 3000
    total_tokens = 0
    context_chunks = []
    context_scores = []
    context_dict = {}
    pages_context_dict = {}
    map_symbol_to_index = config["map_symbol_to_index"]
    # Reverse the key and value of the map_symbol_to_index
    map_index_to_symbol = {v: k for k, v in map_symbol_to_index.items()}
    # Get the first 3 keys from map_symbol_to_index for examples in the prompt
    first_keys = list(map_symbol_to_index.keys())[:3]
    example_keys = ", or ".join(first_keys)

    # Track seen content to avoid duplicates
    seen_contents = set()

    symbol_index = 0
    for chunk, score in question_chunks_with_scores:
        # Skip if content already seen
        if chunk.page_content in seen_contents:
            continue

        if total_tokens + count_tokens(chunk.page_content) > token_limit:
            break

        # Add content to seen set
        seen_contents.add(chunk.page_content)
        
        sources_chunks.append(chunk)
        total_tokens += count_tokens(chunk.page_content)
        context_chunks.append(chunk.page_content)
        context_scores.append(score)

        # Logic for configing the context for model input
        context_dict[map_index_to_symbol[symbol_index]] = {"content": chunk.page_content, "score": float(score)}
        symbol_index += 1

        # Logic for configing the context for model input. For pages_context_dict, the key is file index, and the value is the corresponding list of pages where the chunks are from.
        # e.g. we can have {<file_index_0>: {<page_number_1>: <page_content_1>, <page_number_2>: <page_content_2>}...}
    
    # Format context as a JSON dictionary instead of a string
    formatted_context = context_dict
    
    logger.info(f"For {chat_session.mode} model, user_input_string: {user_input_string}")
    logger.info(f"For {chat_session.mode} model, user_input_string tokens: {count_tokens(user_input_string)}")
    logger.info(f"For {chat_session.mode} model, chat_history_string: {chat_history_string}")
    logger.info(f"For {chat_session.mode} model, chat_history_string tokens: {count_tokens(chat_history_string)}")
    logger.info(f"For {chat_session.mode} model, context: {str(formatted_context)}")
    for index, (chunk, score) in enumerate(zip(context_chunks, context_scores)):
        logger.info(f"For {chat_session.mode} model, context chunk number: {index}")
        # logger.info(f"For inference model, context chunk: {chunk}")
        logger.info(f"For {chat_session.mode} model, context chunk tokens: {count_tokens(chunk)}")
        logger.info(f"For {chat_session.mode} model, context chunk score: {score}")
    logger.info(f"For {chat_session.mode} model, context tokens: {count_tokens(str(formatted_context))}")
    logger.info("before deep_inference_agent ...")

    chat_session.formatted_context = formatted_context

    # Create page_formatted_context using page-based index embeddings
    try:
        # Load page-based embeddings based on the current mode
        if chat_session.mode == ChatMode.BASIC or chat_session.mode == ChatMode.ADVANCED:
            logger.info(f"Loading page-based embeddings for {chat_session.mode} mode")
            page_embedding_folder_list = [os.path.join(embedding_folder, 'markdown', 'page_based_index') for embedding_folder in embedding_folder_list]
            page_db = load_embeddings(page_embedding_folder_list, 'default')
        else:  # ChatMode.LITE
            logger.info(f"Loading page-based embeddings for {chat_session.mode} mode")
            page_embedding_folder_list = [os.path.join(embedding_folder, 'lite_embedding', 'page_based_index') for embedding_folder in embedding_folder_list]
            page_db = load_embeddings(page_embedding_folder_list, 'lite')
        
        logger.info(f"Successfully loaded page-based embeddings from: {page_embedding_folder_list}")
        
        # For each chunk in formatted_context, find the corresponding page-level chunk
        page_chunks_dict = {}  # To store unique page chunks with their original scores
        seen_page_contents = set()  # To track duplicates
        
        for symbol, chunk_data in formatted_context.items():
            chunk_content = chunk_data["content"]
            
            # Search for the most similar page chunk for this specific chunk content
            page_chunks_with_scores = page_db.similarity_search_with_score(chunk_content, k=1)
            
            if page_chunks_with_scores:
                page_chunk, page_score = page_chunks_with_scores[0]
                page_content = page_chunk.page_content
                
                # Only add if we haven't seen this page content before
                if page_content not in seen_page_contents and len(page_content) > filter_min_length:
                    seen_page_contents.add(page_content)
                    # Use the original chunk's symbol but store page content and score
                    page_chunks_dict[symbol] = {
                        "content": page_content,
                        "score": float(page_score),
                        "original_chunk_symbol": symbol
                    }
        
        # Create page_formatted_context with unique page chunks
        page_formatted_context = {}
        page_symbol_index = 0
        
        for symbol, page_data in page_chunks_dict.items():
            if page_symbol_index < len(map_index_to_symbol):
                page_formatted_context[map_index_to_symbol[page_symbol_index]] = {
                    "content": page_data["content"],
                    "score": page_data["score"]
                }
                page_symbol_index += 1
        
        logger.info(f"Created page_formatted_context with {len(page_formatted_context)} unique pages")
        logger.info(f"Page formatted context tokens: {count_tokens(str(page_formatted_context))}")
        
        for page_symbol, page_data in page_formatted_context.items():
            logger.info(f"Page context symbol: {page_symbol}, content length: {len(page_data['content'])}, score: {page_data['score']}")
        
        chat_session.page_formatted_context = page_formatted_context
        
    except Exception as e:
        logger.exception(f"Failed to create page_formatted_context: {str(e)}")
        # Fallback to empty page context if page-based embeddings fail
        chat_session.page_formatted_context = {}

    return formatted_context