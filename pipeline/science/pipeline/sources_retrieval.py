import os
import io
import re
import fitz
# import pprint
import json
import asyncio
import concurrent.futures
from typing import List, Dict, Tuple, Any

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


async def locate_chunk_in_pdf_async(chunk: str, pdf_path: str, similarity_threshold: float = 0.8, remove_linebreaks: bool = False) -> dict:
    """
    Async version of locate_chunk_in_pdf for concurrent processing.
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            locate_chunk_in_pdf, 
            chunk, pdf_path, similarity_threshold, remove_linebreaks
        )
    return result


async def process_pdf_file_async(file_path: str) -> Tuple[Any, Any]:
    """
    Async version of process_pdf_file for concurrent processing.
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            process_pdf_file, 
            file_path
        )
    return result


async def robust_search_for_async(page: Any, source: str) -> List[Any]:
    """
    Async version of robust_search_for for concurrent processing.
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            robust_search_for, 
            page, source
        )
    return result


async def evaluate_image_relevance_async(
    image_url: str, 
    score: float, 
    user_input: str, 
    image_url_mapping_merged_reverse: Dict[str, str],
    llm: Any,
    error_parser: Any,
    chain: Any,
    idx: int
) -> Tuple[str, float, str, str] | None:
    """
    Async version of image relevance evaluation for concurrent processing.
    """
    if image_url not in image_url_mapping_merged_reverse:
        logger.warning(f"Image URL {image_url} not found in image_url_mapping_merged_reverse")
        return None
    
    logger.info(f"TEST: No. {idx} evaluating image with image_url: {image_url}")
    
    # Get all context descriptions for this image
    descriptions = image_url_mapping_merged_reverse[image_url]
    descriptions_text = "\n".join([f"- {desc}" for desc in descriptions])
    
    try:
        # Run LLM evaluation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: chain.invoke({
                    "question": user_input,
                    "descriptions": descriptions_text
                })
            )
        
        if result["is_relevant"]:
            # Combine vector similarity score with LLM relevance score
            combined_score = (score + result["relevance_score"]) / 2
            return (image_url, combined_score, result["actual_figure_number"], result["explanation"])
    except Exception as e:
        logger.exception(f"Error evaluating image {image_url}: {e}")
    
    return None


async def search_source_in_docs_async(
    docs: List[Any], 
    source: str, 
    score: float
) -> Dict[str, float]:
    """
    Async function to search for a source across multiple document pages concurrently.
    """
    search_tasks = []
    
    # Create tasks for searching across all pages of all documents
    for doc in docs:
        for page in doc:
            task = robust_search_for_async(page, source)
            search_tasks.append(task)
    
    # Execute all searches concurrently
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # Check if any search found the source
    for result in search_results:
        if not isinstance(result, Exception) and result:
            return {source: score}
    
    return {}


async def process_text_sources_concurrently(
    text_sources: Dict[str, float], 
    file_path_list: List[str]
) -> Dict[str, float]:
    """
    Process text sources concurrently for improved performance.
    """
    # First, load all documents concurrently
    doc_tasks = [process_pdf_file_async(file_path) for file_path in file_path_list]
    doc_results = await asyncio.gather(*doc_tasks, return_exceptions=True)
    
    # Extract successful document loads
    docs = []
    for result in doc_results:
        if not isinstance(result, Exception):
            _, doc = result
            docs.append(doc)
        else:
            logger.exception(f"Error loading document: {result}")
    
    # Now search for all sources concurrently
    search_tasks = []
    for source, score in text_sources.items():
        task = search_source_in_docs_async(docs, source, score)
        search_tasks.append(task)
    
    # Execute all source searches concurrently
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # Combine results
    refined_sources = {}
    for result in search_results:
        if not isinstance(result, Exception) and result:
            refined_sources.update(result)
    
    return refined_sources


async def process_image_sources_concurrently(
    image_sources: Dict[str, float],
    user_input: str,
    image_url_mapping_merged_reverse: Dict[str, str],
    llm: Any,
    error_parser: Any,
    chain: Any
) -> Dict[str, float]:
    """
    Process image sources concurrently for improved performance.
    """
    # Create tasks for all image evaluations
    evaluation_tasks = []
    for idx, (image_url, score) in enumerate(image_sources.items()):
        task = evaluate_image_relevance_async(
            image_url, score, user_input, image_url_mapping_merged_reverse,
            llm, error_parser, chain, idx
        )
        evaluation_tasks.append(task)
    
    # Execute all evaluations concurrently
    evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
    
    # Process results
    image_scores = []
    for result in evaluation_results:
        if not isinstance(result, Exception) and result is not None:
            image_scores.append(result)
    
    # Sort images by relevance score
    image_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Filter images with high relevance score (score > 0.5)
    filtered_images = {img_url: score for img_url, score, fig_num, expl in image_scores if score > 0.5}
    
    if filtered_images:
        # Get the highest score
        highest_score = max(filtered_images.values())
        # Keep images with scores within 10% of the highest score
        score_threshold = highest_score * 0.9
        filtered_images = {img_url: score for img_url, score in filtered_images.items() if score >= score_threshold}
    
    return filtered_images


def get_response_source(chat_session: ChatSession, file_path_list, user_input, answer, chat_history, embedding_folder_list):
    """
    Get the sources for the response
    Return a dictionary of sources with scores and metadata
    The scores are normalized to 0-1 range
    The metadata includes the page number, chunk index, and block bounding box coordinates
    The sources are refined by checking if they can be found in the document
    Only get first 20 sources
    Show them in the order they are found in the document
    Preserve image filenames but filter them based on context relevance using LLM
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

    logger.info(f"TEST: length of sources after refine: {len(sources_with_scores)}")

    # Memory cleanup
    db = None

    return sources_with_scores, source_pages, refined_source_pages, refined_source_index


async def refine_sources_simple_async(sources_with_scores: Dict[str, float], file_path_list: List[str]) -> Dict[str, float]:
    """
    Async simplified version of refine_sources_complex that only checks if text chunks can be found in the document.
    Uses concurrent processing for improved performance.
    
    Args:
        sources_with_scores: A dictionary mapping text chunks to their relevance scores
        file_path_list: List of PDF file paths to search in
        
    Returns:
        Dictionary of refined sources that were found in the original documents
    """
    return await process_text_sources_concurrently(sources_with_scores, file_path_list)


def refine_sources_simple(sources_with_scores, file_path_list):
    """
    Synchronous wrapper for the async concurrent version of refine_sources_simple.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use a thread pool
            import concurrent.futures
            
            def run_async():
                # Create new event loop in thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        refine_sources_simple_async(sources_with_scores, file_path_list)
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
        else:
            # No loop running, we can run directly
            return loop.run_until_complete(
                refine_sources_simple_async(sources_with_scores, file_path_list)
            )
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(
            refine_sources_simple_async(sources_with_scores, file_path_list)
        )


async def refine_sources_complex_async(sources_with_scores, file_path_list, markdown_dir_list, user_input, image_url_mapping_merged, source_pages, source_file_index, image_url_mapping_merged_reverse):
    """
    Async version of refine_sources_complex with concurrent processing for improved performance.
    """
    config = load_config()
    
    # Separate image sources from text sources
    image_sources = {}
    text_sources = {}
    for source, score in sources_with_scores.items():
        if source.startswith('https://knowhiztutorrag.blob'):
            image_sources[source] = score
        else:
            text_sources[source] = score

    logger.info(f"TEST: length of image sources before refine: {len(image_sources)}")
    logger.info(f"TEST: length of text sources before refine: {len(text_sources)}")

    # Process both image and text sources concurrently
    tasks = []
    
    # Task 1: Process image sources if they exist
    if image_sources:
        # Initialize LLM for relevance evaluation
        para = config['llm']
        llm = get_llm('basic', para)
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        
        # Create prompt for image relevance evaluation
        system_prompt = """
        You are an expert at evaluating the relevance between a user's question and image descriptions.
        Given a user's question and descriptions of an image, determine if the image is relevant and provide a relevance score.

        First, analyze the image descriptions to identify the actual figure number in the document (e.g., "Figure 1", "Fig. 2", etc.).
        Then evaluate the relevance considering both the actual figure number and the content.

        Organize your response in the following JSON format:
        ```json
        {{
            "actual_figure_number": "<extracted figure number from descriptions, e.g. 'Figure 1', 'Fig. 2', etc.>",
            "is_relevant": <Boolean, True/False>,
            "relevance_score": <float between 0 and 1>,
            "explanation": "<brief explanation including actual figure number and why this image is or isn't relevant>"
        }}
        ```

        Pay special attention to:
        1. If the user asks about a specific figure number (e.g., "Figure 1"), prioritize matching the ACTUAL figure number from descriptions, NOT the filename
        2. The semantic meaning and context of both the question and image descriptions
        3. Whether the image would help answer the user's question
        4. Look for figure number mentions in the descriptions like "Figure X", "Fig. X", "Figure-X", etc.
        """

        human_prompt = """
        User's question: {question}
        Image descriptions:
        {descriptions}

        Note: The image filename may not reflect the actual figure number in the document. Please extract the actual figure number from the descriptions.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])

        chain = prompt | llm | error_parser
        
        image_task = process_image_sources_concurrently(
            image_sources, user_input, image_url_mapping_merged_reverse,
            llm, error_parser, chain
        )
        tasks.append(image_task)
    else:
        # Add empty result for images if no image sources
        async def empty_images():
            return {}
        tasks.append(empty_images())

    # Task 2: Process text sources
    if text_sources:
        text_task = process_text_sources_concurrently(text_sources, file_path_list)
        tasks.append(text_task)
    else:
        # Add empty result for text if no text sources
        async def empty_text():
            return {}
        tasks.append(empty_text())

    # Execute both tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Extract results
    filtered_images = results[0] if not isinstance(results[0], Exception) else {}
    refined_text_sources = results[1] if not isinstance(results[1], Exception) else {}
    
    # Combine filtered image sources with refined text sources
    final_sources = {**filtered_images, **refined_text_sources}

    # Sort by score
    sorted_sources = dict(sorted(final_sources.items(), key=lambda x: x[1], reverse=True))

    # Keep only the top 50% of sources by score
    num_sources_to_keep = max(1, len(sorted_sources) // 2)  # Keep at least 1 source
    sorted_sources = dict(list(sorted_sources.items())[:num_sources_to_keep])

    # Further limit to top 20 if needed
    sorted_sources = dict(list(sorted_sources.items())[:20])

    logger.info("TEST: sorted sources after concurrent refine:")
    for source, score in sorted_sources.items():
        logger.info(f"{source}: {score}")
    logger.info(f"TEST: length of sorted sources after concurrent refine: {len(sorted_sources)}")

    return sorted_sources


def refine_sources_complex(sources_with_scores, file_path_list, markdown_dir_list, user_input, image_url_mapping_merged, source_pages, source_file_index, image_url_mapping_merged_reverse):
    """
    Synchronous wrapper for the async concurrent version.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use a thread pool
            import threading
            import concurrent.futures
            
            def run_async():
                # Create new event loop in thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        refine_sources_complex_async(
                            sources_with_scores, file_path_list, markdown_dir_list, 
                            user_input, image_url_mapping_merged, source_pages, 
                            source_file_index, image_url_mapping_merged_reverse
                        )
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
        else:
            # No loop running, we can run directly
            return loop.run_until_complete(
                refine_sources_complex_async(
                    sources_with_scores, file_path_list, markdown_dir_list, 
                    user_input, image_url_mapping_merged, source_pages, 
                    source_file_index, image_url_mapping_merged_reverse
                )
            )
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(
            refine_sources_complex_async(
                sources_with_scores, file_path_list, markdown_dir_list, 
                user_input, image_url_mapping_merged, source_pages, 
                source_file_index, image_url_mapping_merged_reverse
            )
        )


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0


class PageAwareTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter that respects page boundaries"""

    def split_document(self, document):
        """Split document while respecting page boundaries"""
        final_chunks = []

        for doc in document:
            # Get the page number from the metadata
            page_num = doc.metadata.get("page", 0)
            text = doc.page_content

            # Use parent class's splitting logic first
            chunks = super().split_text(text)

            # Create new document for each chunk with original metadata
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                # Update metadata to indicate chunk position
                metadata["chunk_index"] = i
                final_chunks.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))

        # Sort chunks by page number and then by chunk index
        final_chunks.sort(key=lambda x: (
            x.metadata.get("page", 0),
            x.metadata.get("chunk_index", 0)
        ))

        return final_chunks