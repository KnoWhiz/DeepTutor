import os
import re
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
    filter_min_length = 100
    fetch_k = max(config['retriever']['k'] * 2, 20)  # Fetch 3x more to ensure enough pass the filter

    all_chunks_with_scores = db.similarity_search_with_score(rag_user_input_string, k=fetch_k)

    logger.info(f"TEST: all_chunks_with_scores: {all_chunks_with_scores}")

    # Filter chunks by length > 100
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
        context_dict[map_index_to_symbol[symbol_index]] = {"content": chunk.page_content, "score": float(score)}
        symbol_index += 1
    
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

    return formatted_context