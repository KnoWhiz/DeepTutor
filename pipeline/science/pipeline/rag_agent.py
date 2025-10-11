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
    load_embeddings_with_regeneration,
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
    """
    Retrieves and processes relevant document context for RAG (Retrieval-Augmented Generation) operations.

    This function performs semantic similarity search across embedded document chunks to identify
    the most relevant content for answering user questions. It supports multiple chat modes
    (Basic, Advanced, Lite) with different embedding strategies and provides both chunk-level
    and page-level context formatting for optimal AI response generation.

    Args:
        chat_session (ChatSession): Active chat session containing mode settings and context state
        file_path_list (List[str]): Paths to uploaded document files being queried
        question (Question): Structured question object containing text, language, and metadata
        chat_history (List): Historical conversation context for improved retrieval
        embedding_folder_list (List[str]): Paths to directories containing pre-computed embeddings
        deep_thinking (bool, optional): Enable enhanced reasoning capabilities. Defaults to True
        stream (bool, optional): Enable streaming response mode. Defaults to False
        context (str, optional): Additional context to append to user input. Defaults to ""

    Returns:
        Dict[str, Dict]: Formatted context dictionary mapping symbols to content chunks with scores

    Processing Pipeline:
        1. Mode Detection & Embedding Loading:
           - Basic/Advanced: Loads markdown embeddings for rich document understanding
           - Lite: Loads lightweight embeddings for faster processing
           
        2. Query Enhancement:
           - Combines user question with additional context and special planning information
           - Truncates chat history based on mode-specific token limits
           
        3. Semantic Retrieval:
           - Performs similarity search with configurable fetch size (2x target for filtering)
           - Filters chunks by minimum length (50 characters) to ensure content quality
           - Deduplicates results to avoid redundant information
           
        4. Token Budget Management:
           - Accumulates chunks until token limit reached (varies by mode)
           - Prioritizes highest scoring (most relevant) chunks first
           - Maps chunks to symbolic references (A, B, C, etc.) for model consumption
           
        5. Dual Context Creation:
           - Chunk-level context: Fine-grained relevant text segments
           - Page-level context: Broader document context for comprehensive understanding
           - Both contexts use similarity scoring for relevance ranking
           
        6. Session State Update:
           - Stores formatted_context in chat session for response generation
           - Stores page_formatted_context for enhanced document understanding

    Context Format:
        {
            "[<1>]": {
                "content": "relevant text chunk", 
                "score": 0.85,
                "page_num": 5,      # 1-indexed page number (page 5)
                "source_index": 1   # 1-indexed file position (first file)
            },
            "[<2>]": {
                "content": "another chunk", 
                "score": 0.72,
                "page_num": 12,     # 1-indexed page number (page 12)  
                "source_index": 2   # 1-indexed file position (second file)
            },
            ...
        }

    Indexing Convention:
        - page_num: 1-indexed page numbers for user-friendly display (page 1, 2, 3, ...)
        - source_index: 1-indexed file positions for user-friendly display (file 1, 2, 3, ...)
        - Internal metadata uses 0-indexed values, but output context uses 1-indexed for clarity

    Error Handling:
        - Graceful fallback from markdown to default embeddings if loading fails
        - Empty page context fallback if page-based embedding creation fails
        - Comprehensive logging for debugging retrieval quality issues

    Performance Considerations:
        - Configurable token limits prevent context overflow
        - Deduplication reduces redundant processing
        - Mode-specific optimizations balance quality vs. speed
    """
    config = load_config()
    
    # === STEP 1: Query Enhancement ===
    # Combine user question with additional context to improve retrieval quality
    user_input = question.text #+ "\n\n" + context

    # === STEP 2: Mode-Specific Embedding Loading ===
    # Handle Basic mode and Advanced mode
    if chat_session.mode == ChatMode.BASIC or chat_session.mode == ChatMode.ADVANCED:
        logger.info(f"Current mode is {chat_session.mode}")
        try:
            logger.info(f"Loading markdown embeddings from {[os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]}")
            # Fix: Markdown embeddings are saved directly under markdown subfolder
            pagebased_embedding_folder_list = [os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]
            db = await load_embeddings(pagebased_embedding_folder_list, 'default', file_path_list)
        except Exception as e:
            logger.exception(f"Failed to load markdown embeddings for deep thinking mode: {str(e)}")
            # Fallback to main embedding folder if markdown embeddings don't exist
            db = await load_embeddings(embedding_folder_list, 'default', file_path_list)
    # === STEP 2b: Lite Mode Handling ===
    # Handle Lite mode in other cases - uses lightweight embeddings
    else:
        logger.info(f"Current mode is {chat_session.mode}")
        actual_embedding_folder_list = [os.path.join(embedding_folder, 'lite_embedding') for embedding_folder in embedding_folder_list]
        logger.info(f"actual_embedding_folder_list in get_rag_context: {actual_embedding_folder_list}")
        db = await load_embeddings_with_regeneration(actual_embedding_folder_list, 'lite', allow_regeneration=True, file_path_list=file_path_list)

    # === STEP 3: Token Budget Configuration ===
    # Configure token limits based on chat mode for optimal context size
    config = load_config()
    if chat_session.mode == ChatMode.LITE:
        token_limit = config["basic_token_limit"]
    else:
        token_limit = config["inference_token_limit"]
    chat_history_string = truncate_chat_history(chat_history, token_limit=token_limit)
    # user_input_string = str(user_input + "\n\n" + question.special_context)
    user_input_string = str(user_input)
    # rag_user_input_string = str(user_input + "\n\n" + question.special_context + "\n\n" + str(question.answer_planning))
    rag_user_input_string = str(user_input)
    logger.info(f"rag_user_input_string: {rag_user_input_string}")
    # === STEP 4: Semantic Similarity Search ===
    # Retrieve document chunks most similar to the enhanced user query
    # Fetch 2x more candidates than needed to ensure sufficient quality chunks after filtering
    filter_min_length = 50
    fetch_k = min(config['retriever']['k'] * 2, 30)  # Fetch 3x more to ensure enough pass the filter

    all_chunks_with_scores = db.similarity_search_with_score(rag_user_input_string, k=fetch_k)

    # logger.info(f"TEST: all_chunks_with_scores: {all_chunks_with_scores}")

    # === STEP 5: Quality Filtering ===
    # Remove short chunks that don't provide meaningful context
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

    # === STEP 6: Context Building & Token Management ===
    # Initialize data structures for context accumulation
    sources_chunks = []  # Source chunks for further processing
    total_tokens = 0     # Running token count to stay within limits
    context_chunks = []  # Selected chunk contents
    context_scores = []  # Corresponding similarity scores
    context_dict = {}    # Symbolic mapping for model consumption
    pages_context_dict = {}  # Page-level context mapping
    
    # Configure symbolic references (A, B, C, etc.) for chunk identification
    map_symbol_to_index = config["map_symbol_to_index"]
    map_index_to_symbol = {v: k for k, v in map_symbol_to_index.items()}
    first_keys = list(map_symbol_to_index.keys())[:3]
    example_keys = ", or ".join(first_keys)

    # Deduplication tracking to avoid redundant context
    seen_contents = set()

    # === STEP 7: Context Accumulation Loop ===
    # Process chunks in order of relevance until token budget exhausted
    symbol_index = 0
    for chunk, score in question_chunks_with_scores:
        # Skip duplicate content to avoid redundancy
        if chunk.page_content in seen_contents:
            continue

        # Stop if adding this chunk would exceed token budget
        if total_tokens + count_tokens(chunk.page_content) > token_limit:
            break

        # Mark content as seen and add to context
        seen_contents.add(chunk.page_content)
        sources_chunks.append(chunk)
        total_tokens += count_tokens(chunk.page_content)
        context_chunks.append(chunk.page_content)
        context_scores.append(score)

        # Map chunk to symbolic reference for model consumption
        # Extract metadata with safe defaults for missing keys
        # NOTE: Both page_num and source_index are converted to 1-indexed for user-friendly display
        # - page_num: 1-indexed page number (chunk.metadata['page'] is 0-indexed internally)
        # - source_index: 1-indexed file position (chunk.metadata['file_index'] is 0-indexed internally)
        page_num = chunk.metadata.get('page', 0) + 1  # Convert 0-indexed to 1-indexed, default to page 1
        source_index = chunk.metadata.get('file_index', 0) + 1  # Convert 0-indexed to 1-indexed, default to file 1
        
        context_dict[map_index_to_symbol[symbol_index]] = {
            "content": chunk.page_content, 
            "score": float(score),
            "page_num": page_num,  # 1-indexed page number
            "source_index": source_index  # 1-indexed file position
        }
        symbol_index += 1
    
    # === STEP 8: Context Formatting ===
    # Sort context items by source_index first, then by page_number
    sorted_context_items = sorted(
        context_dict.items(),
        key=lambda x: (x[1]["source_index"], x[1]["page_num"])
    )
    
    # Reassign symbols to maintain alphabetical ordering (A, B, C, etc.)
    formatted_context = {}
    for new_index, (old_symbol, context_data) in enumerate(sorted_context_items):
        if new_index < len(map_index_to_symbol):
            new_symbol = map_index_to_symbol[new_index]
            formatted_context[new_symbol] = context_data
    
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
    
    # Log formatted context with metadata
    for symbol, context_data in formatted_context.items():
        logger.info(f"Context {symbol}: tokens={count_tokens(context_data['content'])}, score={context_data['score']:.3f}, page={context_data['page_num']}, source={context_data['source_index']}")
    
    logger.info("before deep_inference_agent ...")

    # Store chunk-level context in session for response generation
    chat_session.formatted_context = formatted_context

    # === STEP 9: Page-Level Context Creation ===
    # Create broader page-level context for enhanced document understanding
    try:
        # Load page-based embeddings based on the current mode
        if chat_session.mode == ChatMode.BASIC or chat_session.mode == ChatMode.ADVANCED:
            logger.info(f"Loading page-based embeddings for {chat_session.mode} mode")
            # Fix: Page-based embeddings are saved directly under markdown subfolder
            page_embedding_folder_list = [os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]
            page_db = await load_embeddings(page_embedding_folder_list, 'default', file_path_list)
        else:  # ChatMode.LITE
            logger.info(f"Loading page-based embeddings for {chat_session.mode} mode")
            page_embedding_folder_list = [os.path.join(embedding_folder, 'lite_embedding') for embedding_folder in embedding_folder_list]
            page_db = await load_embeddings_with_regeneration(page_embedding_folder_list, 'lite', allow_regeneration=True, file_path_list=file_path_list)
        
        logger.info(f"Successfully loaded page-based embeddings from: {page_embedding_folder_list}")
        
        # === STEP 9b: Page-Level Retrieval ===
        # For each selected chunk, find corresponding page-level content
        page_chunks_dict = {}     # Store unique page chunks with scores
        seen_page_contents = set()  # Deduplicate page content
        
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
                    # Extract metadata from page chunk with safe defaults
                    # NOTE: Converting both values to 1-indexed for consistency with chunk-level context
                    # - page_num: 1-indexed page number (metadata['page'] is 0-indexed internally)
                    # - source_index: 1-indexed file position (metadata['file_index'] is 0-indexed internally)
                    page_num = page_chunk.metadata.get('page', 0) + 1  # Convert to 1-indexed
                    source_index = page_chunk.metadata.get('file_index', 0) + 1  # Convert to 1-indexed
                    
                    # Use the original chunk's symbol but store page content and metadata
                    page_chunks_dict[symbol] = {
                        "content": page_content,
                        "score": float(page_score),
                        "page_num": page_num,  # 1-indexed page number
                        "source_index": source_index,  # 1-indexed file position
                        "original_chunk_symbol": symbol
                    }
        
        # === STEP 9c: Page Context Formatting ===
        # Sort page chunks by source_index first, then by page_number
        sorted_page_items = sorted(
            page_chunks_dict.items(),
            key=lambda x: (x[1]["source_index"], x[1]["page_num"])
        )
        
        # Create symbolic mapping for page-level chunks with proper ordering
        page_formatted_context = {}
        for page_symbol_index, (original_symbol, page_data) in enumerate(sorted_page_items):
            if page_symbol_index < len(map_index_to_symbol):
                page_formatted_context[map_index_to_symbol[page_symbol_index]] = {
                    "content": page_data["content"],
                    "score": page_data["score"],
                    "page_num": page_data["page_num"],
                    "source_index": page_data["source_index"]
                }
        
        logger.info(f"Created page_formatted_context with {len(page_formatted_context)} unique pages")
        logger.info(f"Page formatted context tokens: {count_tokens(str(page_formatted_context))}")
        
        for page_symbol, page_data in page_formatted_context.items():
            logger.info(f"Page context symbol: {page_symbol}, content length: {len(page_data['content'])}, score: {page_data['score']}, page: {page_data['page_num']}, source: {page_data['source_index']}")
        
        chat_session.page_formatted_context = page_formatted_context
        
    except Exception as e:
        logger.exception(f"Failed to create page_formatted_context: {str(e)}")
        # Fallback to empty page context if page-based embeddings fail
        chat_session.page_formatted_context = {}

    return formatted_context