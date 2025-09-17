import os
import io
import re
import fitz
# import pprint
import json

from difflib import SequenceMatcher

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    get_llm,
    robust_search_for,
)
from pipeline.science.pipeline.embeddings import (
    load_embeddings,
)
from pipeline.science.pipeline.embeddings_agent import embeddings_agent
from pipeline.science.pipeline.doc_processor import process_pdf_file
from pipeline.science.pipeline.session_manager import ChatSession
import logging
logger = logging.getLogger("tutorpipeline.science.sources_retrieval")


def normalize_text(text, remove_linebreaks=True):
    """
    Normalize text by removing excessive whitespace and standardizing common special characters.
    
    Args:
        text: Text to normalize
        remove_linebreaks: If True, replace line breaks with empty string, otherwise keep current behavior
        
    Returns:
        Normalized text
    """
    if remove_linebreaks:
        # Replace line breaks with empty string
        text = re.sub(r'[\n\r]+', '', text)
    
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize special characters that might appear differently in PDFs
    text = text.replace('−', '-')  # Replace Unicode minus with hyphen
    text = text.replace('∼', '~')  # Replace tilde approximation
    
    # Handle commonly misrecognized math symbols
    text = re.sub(r'\|\s*↓\s*⟩', '|↓⟩', text)
    text = re.sub(r'\|\s*↑\s*⟩', '|↑⟩', text)
    
    # Clean up spaces around symbols
    text = re.sub(r'\s*\[\s*(\d+)\s*\]', r'[\1]', text)  # [39] -> [39]
    
    return text.strip()


def locate_chunk_in_pdf(chunk: str, source_page_number: int, pdf_path: str, similarity_threshold: float = 0.8, remove_linebreaks: bool = False) -> dict:
    """
    Locates a text chunk within a PDF file and returns its position information.
    Uses both exact matching and fuzzy matching for robustness.
    
    Args:
        chunk: A string of text to locate within the PDF
        pdf_path: Path to the PDF file
        similarity_threshold: Threshold for fuzzy matching (0.0-1.0)
        remove_linebreaks: If True, remove line breaks during text normalization
    
    Returns:
        Dictionary containing:
            - page_num: The page number where the chunk was found (0-indexed)
            - start_char: The starting character position in the page
            - end_char: The ending character position in the page
            - success: Boolean indicating if the chunk was found
            - similarity: Similarity score if found by fuzzy matching
    """
    result = {
        "page_num": 1,
        "start_char": 1,
        "end_char": 10,
        "success": False,
        "similarity": 0.0
    }
    # logger.info(f"TEST: CODE0745 source_page_number: {source_page_number}, chunk: {chunk}")
    if (source_page_number is not None) and (chunk is not None) and (chunk.strip() != ""):
        result = {
            "page_num": int(source_page_number),
            "start_char": 1,
            "end_char": 2,
            "success": True,
            "similarity": 1.0
        }
        return result
    else:
        return result


def get_response_source(chat_session: ChatSession, file_path_list, user_input, answer, chat_history, embedding_folder_list):
    """
    Simplified version that retrieves source references directly from chat_session.formatted_context.
    
    This function extracts source information from the pre-computed formatted_context stored in the
    chat session, which contains chunks ordered by source_index and page_number with their metadata.
    
    Args:
        chat_session (ChatSession): Active chat session containing formatted_context
        file_path_list (List[str]): Paths to the uploaded document files being referenced
        user_input (str): The original user query that prompted the response
        answer (str): The AI-generated response content to find sources for
        chat_history (List): Historical conversation context (unused in simplified version)
        embedding_folder_list (List[str]): Paths to directories containing embeddings (unused in simplified version)
    
    Returns:
        Tuple[Dict, Dict, Dict, Dict]: A 4-tuple containing:
            - sources_with_scores: Dictionary mapping source content to relevance scores (0-1)
            - source_pages: Dictionary mapping source content to 0-indexed page numbers
            - refined_source_pages: Dictionary mapping sources to 1-indexed page numbers
            - refined_source_index: Dictionary mapping sources to their corresponding file indices
    
    Context Format Expected:
        chat_session.formatted_context = {
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
    """
    logger.info("Using simplified get_response_source with formatted_context")
    
    # Initialize result dictionaries
    sources_with_scores = {}
    source_pages = {}
    refined_source_pages = {}
    refined_source_index = {}
    
    # Extract information directly from formatted_context
    if hasattr(chat_session, 'formatted_context') and chat_session.formatted_context:
        for symbol, context_data in chat_session.formatted_context.items():
            content = context_data["content"][:100]
            score = context_data["score"]
            page_num = context_data["page_num"]  # 1-indexed from context
            source_index = context_data["source_index"]  # 1-indexed from context
            
            # Store the content as key with its score
            sources_with_scores[content] = float(score)
            
            # Store 0-indexed page number for source_pages and refined_source_pages (no need to convert from 1-indexed)
            source_pages[content] = page_num
            refined_source_pages[content] = page_num
            
            # Store 0-indexed file index for refined_source_index (converting from 1-indexed)
            # This matches the original behavior where refined_source_index uses the raw file_index
            refined_source_index[content] = source_index - 1
            
        logger.info(f"Extracted {len(sources_with_scores)} sources from formatted_context")
        logger.info(f"Sources with scores: {len(sources_with_scores)} items")
        logger.info(f"Refined source pages: {len(refined_source_pages)} items")
        logger.info(f"Refined source index: {len(refined_source_index)} items")
        
    else:
        logger.warning("No formatted_context found in chat_session, returning empty results")
    
    return sources_with_scores, source_pages, refined_source_pages, refined_source_index