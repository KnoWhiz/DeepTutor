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
from pipeline.science.pipeline.get_graphrag_response import get_GraphRAG_global_response
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
logger = logging.getLogger("tutorpipeline.science.get_response")


async def get_rag_context(chat_session: ChatSession, file_path_list, question: Question, chat_history, embedding_folder_list, deep_thinking = True, stream=False):
    config = load_config()
    user_input = question.text
    user_input_with_context = user_input + "\n\n" + question.special_context
    
    # Handle Lite mode first
    if chat_session.mode == ChatMode.LITE:
        return None

    # Handle Advanced mode
    if chat_session.mode == ChatMode.ADVANCED:
        return None

    if chat_session.mode == ChatMode.BASIC:
        return None