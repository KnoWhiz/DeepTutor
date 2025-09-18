import os
import io
import re
import fitz
# import pprint
import json

import math
from collections import Counter, defaultdict

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
from pipeline.science.pipeline.doc_processor import process_pdf_file, extract_document_from_file
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
import logging
logger = logging.getLogger("tutorpipeline.science.sources_retrieval")


def get_page_raw_text(pdf_path: str, page_number: int) -> str:
    """
    Extract and clean raw text from a specific page of a PDF file.
    
    This function extracts text from a PDF page and applies text cleanup including:
    - Removing hyphenation at line breaks (e.g., "word-\n" becomes "word")
    - Normalizing whitespace (multiple spaces/tabs become single spaces)
    - Stripping leading/trailing whitespace
    
    Args:
        pdf_path (str): Path to the PDF file
        page_number (int): Page number to extract text from (1-indexed, like page 1, 2, 3, etc.)
        
    Returns:
        str: Cleaned text content of the specified page, or empty string if page not found or empty
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If page_number is invalid (less than 1)
        Exception: For other PDF processing errors
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if page_number < 1:
        raise ValueError(f"Page number must be >= 1, got: {page_number}")
    
    try:
        # Extract document using the existing function from doc_processor
        document = extract_document_from_file(pdf_path)
        
        # Convert to 0-indexed for document access
        page_index = page_number - 1
        
        # Check if the page exists
        if page_index >= len(document):
            logger.warning(f"Page {page_number} not found in PDF {pdf_path}. Total pages: {len(document)}")
            return ""
        
        # Extract the page content
        page_doc = document[page_index]
        if hasattr(page_doc, 'page_content') and page_doc.page_content:
            page_text = page_doc.page_content
            
            # Clean up the text using the same logic as in embeddings.py
            clean_text = page_text.strip()
            if clean_text:
                # Remove hyphenation at line breaks
                clean_text = clean_text.replace("-\n", "")
                # Normalize spaces
                clean_text = " ".join(clean_text.split())
                
                logger.info(f"Successfully extracted and cleaned {len(clean_text)} characters from page {page_number} of {os.path.basename(pdf_path)}")
                return clean_text
            else:
                logger.warning(f"No content found on page {page_number} of {pdf_path} after cleanup")
                return ""
        else:
            logger.warning(f"No content found on page {page_number} of {pdf_path}")
            return ""
            
    except Exception as e:
        logger.exception(f"Error extracting text from page {page_number} of {pdf_path}: {str(e)}")
        raise Exception(f"Error extracting text from page {page_number}: {str(e)}")


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



def find_most_relevant_chunk(answer: str,
                             full_content: str,
                             user_input: str = "",
                             divider_number: int = 4) -> str:
    """
    Split full_content into `divider_number` chunks and return the single chunk
    most relevant to (primarily) `user_input` and (secondarily) `answer`.
    No LLMs used; ranking is a weighted combo of TF-IDF cosine, fuzzy ratio,
    and Jaccard token overlap. User input is weighted more than answer.

    Args:
        answer: The assistant's answer string (secondary signal)
        full_content: The entire source content to split and search
        user_input: The original user question (primary signal)
        divider_number: Number of chunks to split full_content into (>=1)

    Returns:
        str: The most relevant chunk (raw text)
    """

    # ----------------------
    # Guard clauses & setup
    # ----------------------
    if not isinstance(full_content, str) or not full_content.strip():
        return full_content or ""

    divider_number = max(1, int(divider_number or 1))
    text = full_content

    # ----------------------
    # Chunking (near-equal, with soft boundary nudging to sentence ends)
    # ----------------------
    def _find_boundary_near(idx: int, span: int = 120) -> int:
        # Try to snap to a nearby "natural" boundary to avoid cutting sentences
        if idx <= 0 or idx >= len(text):
            return max(0, min(idx, len(text)))
        lo = max(0, idx - span)
        hi = min(len(text), idx + span)
        window = text[lo:hi]
        # Prefer sentence-ish boundaries: ., ?, !, ;, \n
        candidates = []
        for i, ch in enumerate(window):
            if ch in ".?!;\n":
                candidates.append(lo + i)
        if not candidates:
            return idx
        # Choose the boundary closest to idx
        return min(candidates, key=lambda x: abs(x - idx))

    approx = max(1, len(text) // divider_number)
    cuts = [0]
    for k in range(1, divider_number):
        cuts.append(_find_boundary_near(k * approx))
    cuts.append(len(text))

    chunks = []
    for i in range(len(cuts) - 1):
        chunk_i = text[cuts[i]:cuts[i + 1]].strip()
        if chunk_i:
            chunks.append(chunk_i)
    if not chunks:
        return text

    # If we ended up with fewer than requested chunks (e.g., very short text),
    # just return the only one.
    if len(chunks) == 1:
        return chunks[0]

    # ----------------------
    # Text normalization & features
    # ----------------------
    def _normalize(s: str) -> str:
        s = s or ""
        s = s.lower()
        # Match the normalize_text() behavior lightly (keep simple)
        s = s.replace('−', '-').replace('∼', '~')
        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _tokens(s: str):
        # Unicode-aware, keeps CJK and alnum; drops most punctuation
        # Using \w with UNICODE grabs letters/numbers/_ plus CJK word chars
        return re.findall(r'\w+', s, flags=re.UNICODE)

    def _char_ngrams(s: str, n: int = 3):
        # Remove spaces to let n-grams span words slightly (helps fuzzy match)
        z = re.sub(r'\s+', '', s)
        if len(z) < n:
            return []
        return [z[i:i+n] for i in range(len(z) - n + 1)]

    def _feature_counts(s: str) -> Counter:
        s_norm = _normalize(s)
        toks = _tokens(s_norm)
        grams = _char_ngrams(s_norm, 3)
        # Prefix feature namespaces so they don't collide
        return Counter([f"w:{t}" for t in toks] + [f"c:{g}" for g in grams])

    # ----------------------
    # Build TF-IDF over chunks
    # ----------------------
    chunk_features = [ _feature_counts(c) for c in chunks ]
    # Document frequency from chunks only
    df = Counter()
    for feats in chunk_features:
        for f in feats.keys():
            df[f] += 1

    N = len(chunk_features)

    def _idf(f: str) -> float:
        # Smoothed IDF
        return math.log((1 + N) / (1 + df.get(f, 0))) + 1.0

    def _tfidf_vec(feats: Counter) -> dict:
        vec = {}
        for f, tf in feats.items():
            # log-scaled TF
            w = (1 + math.log(tf)) * _idf(f)
            vec[f] = w
        return vec

    def _cosine(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        # dot
        dot = 0.0
        # iterate over smaller vector for speed
        if len(a) > len(b):
            a, b = b, a
        for f, wa in a.items():
            wb = b.get(f)
            if wb:
                dot += wa * wb
        # norms
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ----------------------
    # Build query vector (favor user_input over answer)
    # ----------------------
    user_w = 0.75
    ans_w  = 0.25

    q_user = _feature_counts(user_input)
    q_ans  = _feature_counts(answer)

    # Weighted merge before TF-IDF transform
    q_merged = Counter()
    for f, v in q_user.items():
        q_merged[f] += user_w * v
    for f, v in q_ans.items():
        q_merged[f] += ans_w * v

    q_vec = _tfidf_vec(q_merged)

    # Precompute chunk vectors
    chunk_vecs = [ _tfidf_vec(cf) for cf in chunk_features ]

    # ----------------------
    # Additional lightweight signals
    # ----------------------
    user_norm = _normalize(user_input)
    ans_norm  = _normalize(answer)

    user_tokens = set(_tokens(user_norm))
    ans_tokens  = set(_tokens(ans_norm))
    query_tokens = user_tokens | ans_tokens

    def _jaccard_tokens(chunk_text: str) -> float:
        ctoks = set(_tokens(_normalize(chunk_text)))
        if not ctoks and not query_tokens:
            return 0.0
        inter = len(ctoks & query_tokens)
        union = len(ctoks | query_tokens) or 1
        return inter / union

    def _fuzzy_ratio(a: str, b: str) -> float:
        # SequenceMatcher ratio in [0,1]
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    # ----------------------
    # Score & select best chunk
    # ----------------------
    # Weights: TF-IDF cosine dominates; fuzzy & Jaccard as stabilizers
    W_cosine = 0.60
    W_fuzzy  = 0.25   # internally favors user_input over answer
    W_jacc   = 0.15

    best_idx = 0
    best_score = float("-inf")

    for i, (c_text, c_vec) in enumerate(zip(chunks, chunk_vecs)):
        cos = _cosine(q_vec, c_vec)

        # Heavily bias fuzzy toward user_input
        f_user = _fuzzy_ratio(user_norm, _normalize(c_text))
        f_ans  = _fuzzy_ratio(ans_norm,  _normalize(c_text))
        f_mix  = 0.70 * f_user + 0.30 * f_ans

        jac = _jaccard_tokens(c_text)

        score = W_cosine * cos + W_fuzzy * f_mix + W_jacc * jac

        if score > best_score:
            best_score = score
            best_idx = i

    return chunks[best_idx]


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
            # content = context_data["content"][:100]
            # content = context_data["content"]
            if chat_session.mode == ChatMode.LITE:
                full_content=context_data["content"]
                content = find_most_relevant_chunk(answer, full_content, user_input=user_input, divider_number=4)
            else:
                full_content = get_page_raw_text(file_path_list[0], context_data["page_num"])
                content = find_most_relevant_chunk(answer, full_content, user_input=user_input, divider_number=4)
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