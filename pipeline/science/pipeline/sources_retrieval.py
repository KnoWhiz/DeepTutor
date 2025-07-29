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


def locate_chunk_in_pdf(chunk: str, pdf_path: str, similarity_threshold: float = 0.8, remove_linebreaks: bool = False) -> dict:
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
    
    try:
        # Normalize the search chunk
        normalized_chunk = normalize_text(chunk, remove_linebreaks)
        chunk_words = normalized_chunk.split()
        min_match_length = min(100, len(normalized_chunk))  # For long chunks, we'll use word-based search
        
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # First try exact matching
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            normalized_text = normalize_text(text, remove_linebreaks)
            
            # Try exact match first
            start_pos = normalized_text.find(normalized_chunk)
            if start_pos != -1:
                end_pos = start_pos + len(normalized_chunk)
                result["page_num"] = page_num
                result["start_char"] = start_pos
                result["end_char"] = end_pos
                result["success"] = True
                result["similarity"] = 1.0
                break
            
            # If chunk is long, try matching the first N words
            if len(chunk_words) > 10:
                first_words = " ".join(chunk_words[:10])
                start_pos = normalized_text.find(first_words)
                
                if start_pos != -1:
                    # Found beginning of the chunk, now check similarity
                    potential_match = normalized_text[start_pos:start_pos + len(normalized_chunk)]
                    
                    # Check if lengths are comparable
                    if abs(len(potential_match) - len(normalized_chunk)) < 0.2 * len(normalized_chunk):
                        # Calculate similarity
                        similarity = SequenceMatcher(None, normalized_chunk, potential_match).ratio()
                        
                        if similarity > similarity_threshold:
                            result["page_num"] = page_num
                            result["start_char"] = start_pos
                            result["end_char"] = start_pos + len(potential_match)
                            result["success"] = True
                            result["similarity"] = similarity
                            break
        
        # If not found, try sliding window approach on pages
        if not result["success"]:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                normalized_text = normalize_text(text, remove_linebreaks)
                
                # For very long chunks, we'll use a sliding window approach
                # to find the most similar section
                if len(normalized_chunk) > min_match_length:
                    best_similarity = 0
                    best_start = -1
                    
                    # Try to match beginning of chunk with sliding windows
                    first_words = " ".join(chunk_words[:10])
                    for match in re.finditer(re.escape(chunk_words[0]), normalized_text):
                        start_pos = match.start()
                        
                        # Skip if there's not enough text left
                        if start_pos + len(normalized_chunk) > len(normalized_text):
                            continue
                        
                        # Extract a section of text of similar length
                        window_size = min(len(normalized_chunk) + 100, len(normalized_text) - start_pos)
                        window_text = normalized_text[start_pos:start_pos + window_size]
                        
                        # Compare beginning of window with beginning of chunk
                        window_start = window_text[:len(first_words)]
                        similarity = SequenceMatcher(None, first_words, window_start).ratio()
                        
                        if similarity > 0.8:  # If beginning matches well, check the whole chunk
                            chunk_similarity = SequenceMatcher(None, normalized_chunk, 
                                                             window_text[:len(normalized_chunk)]).ratio()
                            if chunk_similarity > best_similarity:
                                best_similarity = chunk_similarity
                                best_start = start_pos
                    
                    if best_similarity > similarity_threshold:
                        result["page_num"] = page_num
                        result["start_char"] = best_start
                        result["end_char"] = best_start + len(normalized_chunk)
                        result["success"] = True
                        result["similarity"] = best_similarity
                        break
        
        doc.close()
    except Exception as e:
        print(f"Error processing PDF: {e}")
    
    logger.info(f"TEST: result: {result}")
    logger.info(f"Format of result: {type(result)}")

    if isinstance(result, dict):
        return result
    else:
        return {
        "page_num": 1,
        "start_char": 1,
        "end_char": 10,
        "success": False,
        "similarity": 0.0
    }


def get_response_source_complex(chat_session: ChatSession, file_path_list, user_input, answer, chat_history, embedding_folder_list):
    """
    Retrieves and processes source references for AI-generated responses in a tutoring system.
    
    This function performs semantic similarity search across embedded document chunks to identify
    relevant source content that supports the generated response. It handles both textual content
    and image references, providing normalized relevance scores and metadata for source attribution.
    
    Args:
        chat_session (ChatSession): Active chat session containing conversation context and settings
        file_path_list (List[str]): Paths to the uploaded document files being referenced
        user_input (str): The original user query that prompted the response
        answer (str): The AI-generated response content to find sources for
        chat_history (List): Historical conversation context for improved source matching
        embedding_folder_list (List[str]): Paths to directories containing pre-computed embeddings
    
    Returns:
        Tuple[Dict, Dict, Dict, Dict]: A 4-tuple containing:
            - sources_with_scores: Dictionary mapping source content to normalized relevance scores (0-1)
            - source_pages: Dictionary mapping source content to original page numbers in documents
            - refined_source_pages: Dictionary mapping validated sources to 1-indexed page numbers
            - refined_source_index: Dictionary mapping sources to their corresponding file indices
    
    Processing Pipeline:
        1. Loads image context mappings and URL references from embedding metadata
        2. Performs similarity search on chat context and response content
        3. Extracts and normalizes relevance scores across all retrieved chunks
        4. Maps image descriptions to URLs while preserving source attribution
        5. Validates sources against original documents and filters results
        6. Returns structured source data for citation and reference display
    
    Note:
        - Relevance scores are inverted distance metrics (lower distance = higher relevance)
        - Image sources are mapped from descriptions to actual URLs when available
        - Source validation ensures cited content can be located in original documents
        - Page numbers are converted to 1-indexed format for user display
    """
    mode = chat_session.mode
    config = load_config()
    para = config['llm']

    # Load image context, mapping from image file name to image descriptions
    logger.info(f"TEST: loading from embedding_folder_list: {embedding_folder_list}")
    image_context_path_list = [os.path.join(embedding_folder, "markdown/image_context.json") for embedding_folder in embedding_folder_list]
    image_context_list = []
    for image_context_path in image_context_path_list:
        if os.path.exists(image_context_path):
            with open(image_context_path, 'r') as f:
                image_context = json.loads(f.read())
        else:
            logger.info(f"image_context_path: {image_context_path} does not exist")
            image_context = {}
            with open(image_context_path, 'w') as f:
                json.dump(image_context, f)
        image_context_list.append(image_context)

    # Load images URL list mapping from images file name to image URL
    image_url_mapping_list = []
    for embedding_folder in embedding_folder_list:
        image_url_path = os.path.join(embedding_folder, "markdown/image_urls.json")
        if os.path.exists(image_url_path):
            with open(image_url_path, 'r') as f:
                image_url_mapping = json.load(f)
        else:
            logger.info("Image URL mapping file not found. Creating a new one.")
            image_url_mapping = {}
            with open(image_url_path, 'w') as f:
                json.dump(image_url_mapping, f)
        image_url_mapping_list.append(image_url_mapping)

    # Create reverse mapping from description to image name
    image_mapping_list = []
    for image_context in image_context_list:
        image_mapping = {}
        for image_name, descriptions in image_context.items():
            for desc in descriptions:
                image_mapping[desc] = image_name
        image_mapping_list.append(image_mapping)

    # Create a single mapping from description to image URL, mapping from image description to image URL
    image_url_mapping_merged = {}
    for idx, (image_mapping, image_url_mapping_merged) in enumerate(zip(image_mapping_list, image_url_mapping_list)):
        for description, image_name in image_mapping.items():
            if image_name in image_url_mapping_merged.keys():
                # If the image name exists in URL mapping, link the description directly to the URL
                image_url_mapping_merged[description] = image_url_mapping_merged[image_name]
            else:
                # If image URL not found, log a warning but don't break the process
                logger.warning(f"Image URL not found for {image_name} in embedding folder {idx}")
    logger.info(f"Created image URL mapping with {len(image_url_mapping_merged)} entries")

    # Create a reverse mapping based on image_url_mapping_merged
    image_url_mapping_merged_reverse = {v: k for k, v in image_url_mapping_merged.items()}

    db_merged = load_embeddings(embedding_folder_list, 'default')

    # Get relevant chunks for both question and answer with scores
    question_chunks_with_scores = []
    for key, value in chat_session.formatted_context.items():
        source_chunk = db_merged.similarity_search_with_score(str(value["content"]), k=1)
        question_chunks_with_scores.append(source_chunk[0])

    # answer_chunks_with_scores = db_merged.similarity_search_with_score(answer, k=config['sources_retriever']['k'])
    answer_chunks_with_scores = []

    # The total list of sources chunks from question and answer
    sources_chunks = []
    for chunk in question_chunks_with_scores:
        sources_chunks.append(chunk[0])
    for chunk in answer_chunks_with_scores:
        sources_chunks.append(chunk[0])

    logger.info(f"TEST: sources_chunks: {sources_chunks}")

    # sources_chunks_text = [chunk.page_content for chunk in sources_chunks]

    # # logger.info(f"TEST: sources_chunks: {sources_chunks}")
    # logger.info(f"TEST: sources_chunks_text: {sources_chunks_text}")

    # Get source pages dictionary, which maps each source to the page number it is found in. the page number is in the metadata of the document chunks
    source_pages = {}
    source_file_index = {}
    for chunk in sources_chunks:
        try:
            source_pages[chunk.page_content] = chunk.metadata['page']
        except KeyError:
            logger.exception(f"Error getting source pages for {chunk.page_content}")
            logger.info(f"Chunk metadata: {chunk.metadata}")
            source_pages[chunk.page_content] = 1
        try:
            # FIXME: KeyError: 'file_index' is not in the metadata
            source_file_index[chunk.page_content] = chunk.metadata['file_index']
            # source_file_index[chunk.page_content] = 1
        except KeyError:
            logger.exception(f"Error getting source file index for {chunk.page_content}")
            logger.info(f"Chunk metadata: {chunk.metadata}")
            source_file_index[chunk.page_content] = 1

    # Extract page content and scores, normalize scores to 0-1 range
    if question_chunks_with_scores and answer_chunks_with_scores:
        max_score = max(max(score for _, score in question_chunks_with_scores),
                       max(score for _, score in answer_chunks_with_scores))
        min_score = min(min(score for _, score in question_chunks_with_scores),
                       min(score for _, score in answer_chunks_with_scores))
    elif question_chunks_with_scores:
        max_score = max(score for _, score in question_chunks_with_scores)
        min_score = min(score for _, score in question_chunks_with_scores)
    elif answer_chunks_with_scores:
        max_score = max(score for _, score in answer_chunks_with_scores)
        min_score = min(score for _, score in answer_chunks_with_scores)
    else:
        # If both lists are empty, set default values
        max_score = 1.0
        min_score = 0.0
    
    score_range = max_score - min_score if max_score != min_score else 1

    # Get sources_with_scores dictionary, which maps each source to the score it has
    sources_with_scores = {}
    # Process question chunks
    for chunk, score in question_chunks_with_scores:
        normalized_score = 1 - (score - min_score) / score_range  # Invert because lower distance = higher relevance
        sources_with_scores[chunk.page_content] = max(normalized_score, sources_with_scores.get(chunk.page_content, 0))
    # Process answer chunks
    for chunk, score in answer_chunks_with_scores:
        normalized_score = 1 - (score - min_score) / score_range  # Invert because lower distance = higher relevance
        sources_with_scores[chunk.page_content] = max(normalized_score, sources_with_scores.get(chunk.page_content, 0))

    # Replace matching sources with image names while preserving scores
    #     Source_pages: a dictionary that maps each source to the page number it is found in
    #     Source_file_index: a dictionary that maps each source to the file index it is found in
    #     Sources_with_scores: a dictionary that maps each source to the score it has
    sources_with_scores = {image_url_mapping_merged.get(source, source): score
                         for source, score in sources_with_scores.items()}
    source_pages = {image_url_mapping_merged.get(source, source): page
                         for source, page in source_pages.items()}
    source_file_index = {image_url_mapping_merged.get(source, source): file_index
                         for source, file_index in source_file_index.items()}

    # # TEST
    # logger.info(f"TEST: sources before refine: sources_with_scores - {sources_with_scores}")
    logger.info(f"TEST: length of sources before refine: {len(sources_with_scores)}")

    # Refine and limit sources while preserving scores
    markdown_dir_list = [os.path.join(embedding_folder, "markdown") for embedding_folder in embedding_folder_list]
    # sources_with_scores = refine_sources_complex(sources_with_scores, file_path_list, markdown_dir_list, user_input, image_url_mapping_merged, source_pages, source_file_index, image_url_mapping_merged_reverse)
    # FIXME: Temporary use simple version and remove filtering logic
    # sources_with_scores = refine_sources_simple(sources_with_scores, file_path_list)

    # Refine source pages while preserving scores
    refined_source_pages = {}
    refined_source_index = {}
    for source, page in source_pages.items():
        if source in sources_with_scores:
            refined_source_pages[source] = page + 1
            refined_source_index[source] = source_file_index[source]

    # # TEST
    # logger.info(f"TEST: sources after refine: sources_with_scores - {sources_with_scores}")
    logger.info(f"TEST: length of sources after refine: {len(sources_with_scores)}")


    # # TEST
    # logger.info("TEST: refined source index:")
    # for source, index in refined_source_index.items():
    #     logger.info(f"{source}: {index}")
    # logger.info(f"TEST: length of refined source index: {len(refined_source_index)}")

    # # TEST
    # logger.info(f"TEST: sources after refine: sources_with_scores - {sources_with_scores}")
    # logger.info(f"TEST: length of sources after refine: {len(sources_with_scores)}")


    # # TEST
    # logger.info("TEST: refined source index:")
    # for source, index in refined_source_index.items():
    #     logger.info(f"{source}: {index}")
    # logger.info(f"TEST: length of refined source index: {len(refined_source_index)}")

    # # TEST
    # logger.info("TEST: refined source pages:")
    # for source, page in refined_source_pages.items():
    #     logger.info(f"{source}: {page}")
    # logger.info(f"TEST: length of refined source pages: {len(refined_source_pages)}")

    # Memory cleanup
    db = None

    return sources_with_scores, source_pages, refined_source_pages, refined_source_index


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
            content = context_data["content"]
            score = context_data["score"] 
            page_num = context_data["page_num"]  # 1-indexed from context
            source_index = context_data["source_index"]  # 1-indexed from context
            
            # Store the content as key with its score
            sources_with_scores[content] = float(score)
            
            # Store 0-indexed page number for source_pages and refined_source_pages (converting from 1-indexed)
            source_pages[content] = page_num - 1
            refined_source_pages[content] = page_num - 1
            
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