import os
import io
import hashlib
# import shutil
import fitz
import tiktoken
import json
import langid
import requests
import base64
import time
from datetime import datetime, UTC

from dotenv import load_dotenv
from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler

import logging

logger = logging.getLogger("tutorpipeline.science.utils")

load_dotenv()
# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


if SKIP_MARKER_API:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.settings import settings


def format_time_tracking(time_tracking: Dict[str, float]) -> str:
    """
    Format time tracking dictionary into a readable string with appropriate units.

    Args:
        time_tracking: Dictionary of operation names and their durations in seconds

    Returns:
        Formatted string with times converted to appropriate units
    """
    formatted_times = []
    for key, value in time_tracking.items():
        if key == "0. session_id" or key == "0. session_type":
            formatted_times.append(f"{key}: {value}")
            continue
        if key == "0. start_time":
            datetime_str = datetime.fromtimestamp(value, UTC).strftime("%Y-%m-%d %H:%M:%S")
            formatted_times.append(f"{key}: {datetime_str}")
            continue
        if key == "0. end_time" and "0. start_time" in time_tracking:
            total_time_cost = value - time_tracking["0. start_time"]
            if total_time_cost >= 60:
                minutes = int(total_time_cost // 60)
                remaining_seconds = total_time_cost % 60
                formatted_times.append(f"0. total time cost: {minutes}m {remaining_seconds:.2f}s, session_id: {time_tracking.get('0. session_id', 'none')}, mode: {time_tracking.get('0. session_type', 'default')}")
            else:
                formatted_times.append(f"0. total time cost: {total_time_cost:.2f}s, session_id: {time_tracking.get('0. session_id', 'none')}, mode: {time_tracking.get('0. session_type', 'default')}")
            continue
        if key == "0. metrics_time" and "0. end_time" in time_tracking:
            metrics_time_cost = value - time_tracking["0. end_time"]
            formatted_times.append(f"0. metrics record time cost: {metrics_time_cost:.2f}s")
            continue


        if value >= 60:
            minutes = int(value // 60)
            remaining_seconds = value % 60
            formatted_times.append(f"{key}: {minutes}m {remaining_seconds:.2f}s")
        else:
            formatted_times.append(f"{key}: {value:.2f}s")

    return "\n".join(formatted_times)


def file_check_list(embedding_folder) -> list:
    GraphRAG_embedding_folder = os.path.join(embedding_folder, "GraphRAG/")
    create_final_community_reports_path = GraphRAG_embedding_folder + "output/create_final_community_reports.parquet"
    create_final_covariates_path = GraphRAG_embedding_folder + "output/create_final_covariates.parquet"
    create_final_document_path = GraphRAG_embedding_folder + "output/create_final_documents.parquet"
    create_final_entities_path = GraphRAG_embedding_folder + "output/create_final_entities.parquet"
    create_final_nodes_path = GraphRAG_embedding_folder + "output/create_final_nodes.parquet"
    create_final_relationships_path = GraphRAG_embedding_folder + "output/create_final_relationships.parquet"
    create_final_text_units_path = GraphRAG_embedding_folder + "output/create_final_text_units.parquet"
    create_final_communities_path = GraphRAG_embedding_folder + "output/create_final_communities.parquet"
    lancedb_path = GraphRAG_embedding_folder + "output/lancedb/"
    path_list = [
        create_final_community_reports_path,
        create_final_covariates_path,
        create_final_document_path,
        create_final_entities_path,
        create_final_nodes_path,
        create_final_relationships_path,
        create_final_text_units_path,
        create_final_communities_path,
        lancedb_path
    ]
    return GraphRAG_embedding_folder, path_list


def robust_search_for(page, text, chunk_size=512):
    """
    A more robust search routine:
    1) Splits text into chunks of up to 'chunk_size' tokens
       so that extremely large strings don't fail.
    2) Uses PyMuPDF search flags to handle hyphens/spaces
       (still an exact match, but more flexible about formatting).
    """
    flags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_INHIBIT_SPACES

    # Remove leading/trailing whitespace
    text = text.strip()
    if not text:
        return []

    # Tokenize by whitespace. Adjust to your needs if you want a different definition of "token."
    tokens = text.split()

    # If the text is short, just do a direct search
    if len(tokens) <= chunk_size:
        return page.search_for(text, flags=flags)

    # Otherwise, split into smaller chunks and search for each
    results = []
    for i in range(0, len(tokens), chunk_size):
        sub_tokens = tokens[i : i + chunk_size]
        chunk_text = " ".join(sub_tokens).strip()
        found_instances = page.search_for(chunk_text, flags=flags)
        results.extend(found_instances)

    return results


# Generate a unique course ID for the uploaded file
def generate_file_id(file_path):
    # logger.info(f"Generating course ID for file: {file_path}")
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
    file_id = hashlib.md5(file_bytes).hexdigest()
    return file_id


# Add new helper functions
def count_tokens(text, model_name='gpt-4o'):
    """Count tokens in text using tiktoken"""
    try:
        logger.info(f"Counting tokens for text: {text}")
        encoding = tiktoken.encoding_for_model(model_name)
        # tokens = encoding.encode(text)
        tokens = encoding.encode(text, disallowed_special=(encoding.special_tokens_set - {'<|endoftext|>'}))
        length = len(tokens)
        logger.info(f"Length of tokens: {length}")
        return length
    except Exception as e:
        logger.exception(f"Error counting tokens: {str(e)}")
        length = len(text.split())
        logger.info(f"Length of text: {length}")
        return length


def truncate_chat_history(chat_history, model_name='gpt-4o', token_limit=None):
    """Only keep messages that fit within token limit"""
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advanced'
    else:
        model_level = 'basic'
    if token_limit is None:
        max_tokens = int(api.models[model_level]['context_window']/3)
    else:
        max_tokens = token_limit

    logger.info(f"max_tokens: {max_tokens}")
    
    total_tokens = 0
    truncated_history = []
    for message in (chat_history[::-1])[1:]:
        if(message['role'] == 'assistant' or message['role'] == 'user'):
            temp_message = {}
            temp_message['role'] = message['role']
            temp_message['content'] = message['content']
            temp_message = str(temp_message)
            message_tokens = count_tokens(temp_message, model_name)
            if total_tokens + message_tokens > max_tokens:
                break
            truncated_history.insert(0, temp_message)
            total_tokens += message_tokens
    
    # Reverse the order of the truncated history
    # truncated_history = truncated_history[::-1]
    # logger.info(f"truncated_history: {truncated_history}")
    return str(truncated_history)


def truncate_document(_document, model_name='gpt-4o'):
    """Only keep beginning of document that fits token limit"""
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advanced'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)

    # # TEST
    # print(f"max_tokens: {max_tokens}")
    # print(f"model_name: {model_name}")
    # print(f"model_level: {model_level}")
    # print(f"api.models[model_level]: {api.models[model_level]}")
    # print(f"api.models[model_level]['context_window']: {api.models[model_level]['context_window']}")
    # print(f"api.models[model_level]['context_window']/1.2: {api.models[model_level]['context_window']/1.2}")
    # print(f"int(api.models[model_level]['context_window']/1.2): {int(api.models[model_level]['context_window']/1.2)}")
    # # TEST

    _document = str(_document)
    document_tokens = count_tokens(_document, model_name)
    if document_tokens > max_tokens:
        _document = _document[:max_tokens]
    return _document


def get_llm(llm_type, para):
    para = para
    api = ApiHandler(para)
    llm_basic = api.models['basic']['instance']
    llm_advanced = api.models['advanced']['instance']
    llm_creative = api.models['creative']['instance']
    llm_backup = api.models['backup']['instance']
    if llm_type == 'basic':
        return llm_basic
    elif llm_type == 'advanced':
        return llm_advanced
    elif llm_type == 'creative':
        return llm_creative
    elif llm_type == 'backup':
        return llm_backup
    return llm_basic


def get_translation_llm(para):
    """Get LLM instance specifically for translation"""
    api = ApiHandler(para)
    return api.models['basic']['instance']


def detect_language(text):
    """Detect language of the text"""
    # Load languages from config
    config = load_config()
    language_dict = config['languages']
    language_options = list(language_dict.values())
    language_short_dict = config['languages_short']
    language_short_options = list(language_short_dict.keys())

    language = langid.classify(text)[0]
    if language not in language_short_options:
        language = "English"
    else:
        language = language_short_dict[language]
    return language


def translate_content(content: str, target_lang: str) -> str:
    """
    Translates content from source language to target language using the LLM.

    Args:
        content (str): The text content to translate
        target_lang (str): Target language code (e.g. "en", "zh") 

    Returns:
        str: Translated content
    """
    language = detect_language(content)
    if language == target_lang:
        return content

    # Load config and get LLM
    config = load_config()
    para = config['llm']
    llm = get_translation_llm(para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # Create translation prompt
    system_prompt = """
    You are a professional translator.
    Translate the following entire content to {target_lang} language.
    Maintain the meaning and original formatting, including markdown syntax, LaTeX formulas, and emojis.
    Only translate the text content - do not modify any formatting, code, or special syntax.
    """

    system_prompt = """
    你是一个专业的英文到中文的翻译者。
    """

    human_prompt = """
    你是一个专业的英文到中文的翻译者。
    将以下内容全部翻译成中文，包括标题在内的每一句话
    并返回翻译后的中文版本内容：
    {content}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    # Create and execute translation chain
    translation_chain = prompt | llm | error_parser

    translated_content = translation_chain.invoke({
        "target_lang": target_lang,
        "content": content
    })

    return translated_content


def extract_images_from_pdf(
    pdf_doc: fitz.Document,
    output_dir: str | Path,
    min_size: Tuple[int, int] = (50, 50)  # Minimum size to filter out tiny images
) -> List[str]:
    """
    Extract images from a PDF file and save them to the specified directory.

    Args:
        pdf_doc: fitz.Document object
        output_dir: Directory where images will be saved
        min_size: Minimum dimensions (width, height) for images to be extracted

    Returns:
        List of paths to the extracted image files. But currently only supports png or jpg.

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        PermissionError: If output directory can't be created/accessed
        ValueError: If PDF file is invalid
    """
    # Convert paths to Path objects for better handling
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # List to store paths of extracted images
    extracted_images: List[str] = []

    try:
        # Open the PDF
        pdf_document = pdf_doc

        # Counter for naming images
        image_counter = 1

        # Iterate through each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Get images from page
            images = page.get_images()

            # Process each image
            for image_index, image in enumerate(images):
                # Get image data
                base_image = pdf_document.extract_image(image[0])
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Get image size
                img_xref = image[0]
                pix = fitz.Pixmap(pdf_document, img_xref)

                # Skip images smaller than min_size
                if pix.width < min_size[0] or pix.height < min_size[1]:
                    continue

                # Generate image filename
                image_filename = f"{image_counter}.{image_ext}"
                image_path = output_dir / image_filename

                # Save image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                extracted_images.append(str(image_path))
                image_counter += 1

        return extracted_images

    except Exception as e:
        logger.exception(f"Error processing images from PDF: {str(e)}")
        raise ValueError(f"Error processing images from PDF: {str(e)}")

    finally:
        if 'pdf_document' in locals():
            pdf_document.close()


def create_searchable_chunks(doc, chunk_size: int) -> list:
    """
    Create searchable chunks from a PDF document.

    Args:
        doc: The PDF document object
        chunk_size: Maximum size of each text chunk in characters

    Returns:
        list: A list of Document objects containing the chunks
    """
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_blocks = []

        # Get text blocks that can be found via search
        for block in page.get_text("blocks"):
            text = block[4]  # The text content is at index 4
            # Clean up the text
            clean_text = text.strip()

            if clean_text:
                # Remove hyphenation at line breaks
                clean_text = clean_text.replace("-\n", "")
                # Normalize spaces
                clean_text = " ".join(clean_text.split())
                # Replace special characters that might cause issues
                replacements = {
                    # "−": "-",  # Replace unicode minus with hyphen
                    # "⊥": "_|_",  # Replace perpendicular symbol
                    # "≫": ">>",  # Replace much greater than
                    # "%": "",     # Remove percentage signs that might be formatting artifacts
                    # "→": "->",   # Replace arrow
                }
                for old, new in replacements.items():
                    clean_text = clean_text.replace(old, new)

                # Split into chunks of specified size
                while len(clean_text) > 0:
                    # Find a good break point near chunk_size characters
                    end_pos = min(chunk_size, len(clean_text))
                    if end_pos < len(clean_text):
                        # Try to break at a sentence or period
                        last_period = clean_text[:end_pos].rfind(". ")
                        if last_period > 0:
                            end_pos = last_period + 1
                        else:
                            # If no period, try to break at a space
                            last_space = clean_text[:end_pos].rfind(" ")
                            if last_space > 0:
                                end_pos = last_space

                    chunk_text = clean_text[:end_pos].strip()
                    if chunk_text:
                        text_blocks.append(Document(
                            page_content=chunk_text,
                            metadata={
                                "page": page_num,
                                "source": f"page_{page_num + 1}",
                                "chunk_index": len(text_blocks),  # Track position within page
                                "block_bbox": block[:4],  # Store block bounding box coordinates
                                "total_blocks_in_page": len(page.get_text("blocks")),
                                "relative_position": len(text_blocks) / len(page.get_text("blocks"))
                            }
                        ))
                    clean_text = clean_text[end_pos:].strip()

        chunks.extend(text_blocks)

    # Sort chunks by page number and then by chunk index
    chunks.sort(key=lambda x: (
        x.metadata.get("page", 0),
        x.metadata.get("chunk_index", 0)
    ))

    return chunks


def responses_refine(answer, reference=''):
    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    system_prompt = (
        """
        You are a skilled academic content editor specializing in making complex scientific topics accessible and engaging.
        
        # ROLE AND CAPABILITIES
        - You transform complex academic content into clear, engaging educational material
        - You maintain factual accuracy while improving presentation and accessibility
        - You apply proper academic formatting while enhancing readability
        
        # PERSISTENT GUIDELINES
        
        ## Formatting Standards
        - Mathematical expressions use LaTeX syntax:
          * Inline formulas: $formula$ (e.g., $a/b = c/d$)
          * Block formulas: $$formula$$ (e.g., $$\\frac{{a}}{{b}} = \\frac{{c}}{{d}}$$)
        - Use **bold text** for key concepts and important points
        - Use markdown headings (## Heading) for section breaks
        - Use numbered or bulleted lists for sequential steps or related items
        
        ## Content Principles
        - Educational but conversational tone
        - Second-person address ("you") to directly engage readers
        - Strategic use of emojis (1-2 per major section)
        - Concise language that eliminates unnecessary words
        
        ## Core Constraints
        - NEVER introduce new factual content not supported by original material
        - NEVER oversimplify to the point of inaccuracy
        - NEVER overuse formatting or emojis
        - ALWAYS prioritize clarity and educational value over style
        - NEVER include these formatting instructions or references to them in your output
        - NEVER include sections labeled "References:" that contain formatting guidelines
        
        ## Prioritization Hierarchy (When Conflicts Arise)
        1. Factual accuracy (preserve all original facts)
        2. Clarity of explanation (ensure comprehensibility)
        3. Logical structure (maintain coherent flow)
        4. Educational value (enhance learning potential)
        5. Engagement (make content interesting)
        
        ## Input Handling
        - FOR well-structured inputs: Preserve structure while enhancing clarity
        - FOR disorganized inputs: Impose clear structure while preserving all key points
        - FOR technical inputs: Maintain precision while adding explanatory elements
        - FOR verbose inputs: Condense while preserving all important information
        - FOR answers with formulas: Ensure formulas are correctly formatted and explained
        
        ## Self-Assessment Criteria
        Before finalizing your refinement, verify:
        1. All facts from the original are preserved
        2. The logical flow is improved or at least maintained
        3. Complex concepts are explained clearly
        4. Formatting enhances rather than distracts from content
        5. The refined answer would be more helpful to a student than the original
        """
    )
    human_prompt = (
        """
        # REFINEMENT TASK
        Refine the following educational answer to make it more effective, engaging, and accessible:
        
        {answer}
        
        # REFERENCE MATERIAL (if available)
        {reference}
        
        # SPECIFIC REFINEMENT GOALS
        
        1. CONTENT STRUCTURE:
           - Create a clear introduction, main body, and conclusion
           - Break down complex ideas into digestible sections
           - Add transitions between concepts for smooth reading
        
        2. ACADEMIC ACCURACY:
           - Verify factual statements against reference (if provided)
           - Maintain scientific precision while improving clarity
           - If reference is empty, focus on clarity without changing facts
        
        3. EDUCATIONAL EFFECTIVENESS:
           - Clarify complex concepts with clearer explanations
           - Highlight key points with appropriate formatting
           - Add examples or analogies where helpful
        
        4. ENGAGEMENT:
           - Incorporate rhetorical questions or thought-provoking elements
           - Phrase difficult concepts in multiple ways
           - Add encouraging language to motivate the learner
        
        5. ACCESSIBILITY:
           - Simplify overly complex language without losing meaning
           - Break up dense paragraphs into manageable sections
           - Use concrete examples to illustrate abstract concepts
        
        # REFINEMENT PROCESS
        1. First, analyze the input to identify its type (well-structured, disorganized, technical, verbose)
        2. Identify key facts, concepts, and logical flow that must be preserved
        3. Determine areas for improvement based on the specific refinement goals
        4. Apply appropriate formatting enhancements consistently throughout
        5. Verify the refined content meets all self-assessment criteria
        
        # IMPORTANT OUTPUT INSTRUCTIONS
        - Your output should ONLY contain the refined educational content
        - Do NOT include meta-instructions like "References:" followed by formatting guidelines
        - Do NOT include any instructions about LaTeX syntax, markdown, or emojis in your output
        - Do NOT include any labels or sections that explain how to format text
        - If the original answer contains formatting instructions at the end, remove them completely
        - Preserve all LaTeX formulas in the original answer but ensure they are correctly formatted
        - Maintain consistent formatting throughout the entire response
        
        Remember: Your primary goal is to maintain the original meaning and factual accuracy while making the content more effective for learning.
        """
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    chain = prompt | llm | error_parser
    response = chain.invoke({"answer": answer, "reference": reference})
    return response