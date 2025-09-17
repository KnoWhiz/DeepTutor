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


def find_most_relevant_chunk(answer: str, full_content: str, user_input: str = "", divider_number: int = 4) -> str:
    """
    Split `full_content` into `divider_number` chunks and return the single chunk
    most relevant to `answer` (augmented with `user_input`) using classical IR signals.
    
    No LLMs used. Fast and robust to paraphrases via TF-IDF + BM25 + Jaccard + fuzzy ratio.

    Args:
        answer: Model's answer text (the target we're matching to)
        full_content: Long source text to split and search within
        user_input: The original user query (helps guide relevance)
        divider_number: Number of chunks to split into (default: 4)

    Returns:
        str: The most relevant chunk (empty string if no content).
    """
    # --- helpers -------------------------------------------------------------
    def safe_normalize(s: str) -> str:
        # Use existing normalize_text if available; otherwise a minimal fallback.
        try:
            return normalize_text(s, remove_linebreaks=True)
        except NameError:
            s = re.sub(r'[\n\r]+', ' ', s)
            s = re.sub(r'\s+', ' ', s)
            s = s.replace('−', '-').replace('∼', '~')
            return s.strip()

    def split_text_into_chunks(text: str, n: int):
        """Length-aware split that prefers breaking on whitespace near boundaries."""
        text = text or ""
        n = max(1, int(n or 1))
        text_len = len(text)
        if text_len == 0:
            return [""]
        if n == 1 or text_len < n * 400:  # small texts: just return as one or few simple splits
            approx = max(1, text_len // n)
        else:
            approx = text_len // n

        cut_points = [0]
        for i in range(1, n):
            target = i * text_len // n
            # look for whitespace to the right, then to the left, within a small window
            window = 120
            right = re.search(r'\s', text[target:min(text_len, target + window)])
            left = None if right else re.search(r'\s(?=\S*$)', text[max(0, target - window):target])
            if right:
                cut_points.append(target + right.start())
            elif left:
                cut_points.append(target - (window - left.start()))
            else:
                cut_points.append(target)
        cut_points.append(text_len)
        chunks = []
        for i in range(len(cut_points) - 1):
            chunk = text[cut_points[i]:cut_points[i+1]].strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [""]

    STOPWORDS = {
        # small, hand-rolled English stopword list (expandable)
        "the","a","an","and","or","but","if","then","else","for","to","in","of","on","at","by","as",
        "is","am","are","was","were","be","been","being","with","from","that","this","these","those",
        "it","its","into","over","under","between","through","about","above","below","up","down",
        "we","you","they","he","she","i","me","my","our","your","their","them","his","her","ours",
        "yours","theirs","not","no","yes","can","could","may","might","should","would","will","shall",
        "do","does","did","done","doing","than","such","via","per"
    }

    def tokenize(s: str):
        s = s.lower()
        tokens = re.findall(r"\w+", s, flags=re.UNICODE)
        return [t for t in tokens if t and t not in STOPWORDS]

    def jaccard(a_tokens, b_tokens):
        A, B = set(a_tokens), set(b_tokens)
        if not A and not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def build_idf(chunks_tokens):
        N = len(chunks_tokens)
        df = Counter()
        for toks in chunks_tokens:
            for t in set(toks):
                df[t] += 1
        idf = {}
        for t, d in df.items():
            # BM25-style IDF with +1 to keep positive
            idf[t] = math.log((N - d + 0.5) / (d + 0.5) + 1.0)
        return idf

    def tfidf_cosine(doc_tokens, query_tokens, idf):
        if not doc_tokens or not query_tokens:
            return 0.0
        tf_doc = Counter(doc_tokens)
        tf_q = Counter(query_tokens)
        # log-scaled tf
        doc_vec = {}
        for t, c in tf_doc.items():
            doc_vec[t] = (1.0 + math.log(c)) * idf.get(t, 0.0)
        q_vec = {}
        for t, c in tf_q.items():
            q_vec[t] = (1.0 + math.log(c)) * idf.get(t, 0.0)
        # cosine
        dot = 0.0
        for t, w in q_vec.items():
            dot += w * doc_vec.get(t, 0.0)
        norm_d = math.sqrt(sum(v*v for v in doc_vec.values()))
        norm_q = math.sqrt(sum(v*v for v in q_vec.values()))
        if norm_d == 0.0 or norm_q == 0.0:
            return 0.0
        return dot / (norm_d * norm_q)

    def bm25_score(doc_tokens, query_tokens, idf, avgdl, k1=1.5, b=0.75):
        if not doc_tokens or not query_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for t in set(query_tokens):
            f = tf.get(t, 0)
            if f == 0:
                continue
            denom = f + k1 * (1.0 - b + b * (dl / (avgdl if avgdl > 0 else 1.0)))
            score += idf.get(t, 0.0) * ((f * (k1 + 1.0)) / (denom if denom > 0 else 1.0))
        return score

    def minmax_norm(scores):
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-12:
            return [0.0 for _ in scores]
        return [(s - mn) / (mx - mn) for s in scores]

    # --- main ---------------------------------------------------------------
    content = safe_normalize(full_content or "")
    if not content.strip():
        return ""

    # split
    chunks = split_text_into_chunks(content, divider_number)
    # tokens
    chunks_tokens = [tokenize(c) for c in chunks]
    query_text = safe_normalize((answer or "") + " " + (user_input or ""))
    query_tokens = tokenize(query_text)

    # if everything is empty, return the longest chunk
    if all(len(t) == 0 for t in chunks_tokens) or not query_tokens:
        return max(chunks, key=len, default="")

    # idf and stats
    idf = build_idf(chunks_tokens)
    avgdl = sum(len(toks) for toks in chunks_tokens) / max(1, len(chunks_tokens))

    # compute signals
    tfidf_scores = [tfidf_cosine(toks, query_tokens, idf) for toks in chunks_tokens]
    bm25_scores_ = [bm25_score(toks, query_tokens, idf, avgdl) for toks in chunks_tokens]
    jaccard_scores_ = [jaccard(toks, query_tokens) for toks in chunks_tokens]
    # SequenceMatcher on raw strings (cheap since few chunks)
    seq_scores = []
    ans_norm = query_text.lower()
    for c in chunks:
        try:
            seq_scores.append(SequenceMatcher(None, ans_norm, c.lower()).ratio())
        except Exception:
            seq_scores.append(0.0)

    # normalize for stable combination
    tfidf_n = minmax_norm(tfidf_scores)
    bm25_n = minmax_norm(bm25_scores_)
    jacc_n = minmax_norm(jaccard_scores_)
    seq_n = minmax_norm(seq_scores)

    # weighted blend: emphasize lexical/semantic overlap (tfidf/bm25),
    # keep set-similarity and fuzzy as light signals.
    weights = (0.5, 0.35, 0.1, 0.05)
    blended = [
        weights[0]*tfidf_n[i] + weights[1]*bm25_n[i] + weights[2]*jacc_n[i] + weights[3]*seq_n[i]
        for i in range(len(chunks))
    ]

    # pick best; tie-break by longer chunk (more context)
    best_idx = max(range(len(chunks)), key=lambda i: (blended[i], len(chunks[i])))
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
            content = find_most_relevant_chunk(answer, full_content=context_data["content"], user_input=user_input, divider_number=4)
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