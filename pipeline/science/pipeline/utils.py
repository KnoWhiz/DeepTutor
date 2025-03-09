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
import re

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
        # logger.info(f"Counting tokens for text: {text}")
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
    # logger.info(f"max_tokens: {max_tokens}")
    # logger.info(f"model_name: {model_name}")
    # logger.info(f"model_level: {model_level}")
    # logger.info(f"api.models[model_level]: {api.models[model_level]}")
    # logger.info(f"api.models[model_level]['context_window']: {api.models[model_level]['context_window']}")
    # logger.info(f"api.models[model_level]['context_window']/1.2: {api.models[model_level]['context_window']/1.2}")
    # logger.info(f"int(api.models[model_level]['context_window']/1.2): {int(api.models[model_level]['context_window']/1.2)}")
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


def replace_latex_formulas(text):
    """
    Replace LaTeX formulas in various formats with $ formula $ format
    
    Args:
        text (str): Text containing LaTeX formulas
        
    Returns:
        str: Text with replaced LaTeX formula format
    """
    if not text:
        return text
    
    # Replace \( formula \) with $ formula $
    result = re.sub(r'\\[\(](.+?)\\[\)]', r'$ \1 $', text)
    
    # Replace complex mathematical formulas in square brackets 
    # This pattern specifically targets mathematical formulas containing the combination of:
    # 1. Complex LaTeX structures like g^{(2)} 
    # 2. LaTeX commands that start with backslash like \frac, \langle, etc.
    # The comma at the end is optional (from the user's example)
    result = re.sub(r'\[\s*([^\[\]]*?(?:\^\{.*?\}|\\\w+\{).*?)\s*,?\s*\]', r'$ \1 $', result)
    
    # Special case for the exact pattern from the user's example:
    # [ g^{(2)}(\tau) = \frac{\langle \rho_1(\tau) \rho_2(\tau + \delta T) \rangle}{\langle \rho_1(\tau) \rangle \langle \rho_2(\tau + \delta T) \rangle}, ]
    pattern = r'\[\s*(g\^\{.*?\}.*?\\frac\{.*?\}\{.*?\}),?\s*\]'
    result = re.sub(pattern, r'$ \1 $', result)
    
    return result


def responses_refine(answer, reference=''):
    # return answer
    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    system_prompt = (
        """
        You are an expert at refining educational content while maintaining its original meaning and language.
        Your task is to improve the display of content by:
        
        1. Properly formatting all mathematical formulas with LaTeX syntax:
           - Surround inline formulas with single dollar signs ($formula$)
           - Surround block/display formulas with double dollar signs ($$formula$$)
           - Ensure all mathematical symbols, equations, and expressions use proper LaTeX notation
        
        2. Improving readability by:
           - Adding **bold text** to important terms, concepts, and key points
           - Using proper markdown formatting for headings, lists, and other structural elements
           - Maintaining paragraph structure and flow
        
        3. Making the content more engaging by:
           - Adding relevant emojis at appropriate places (section headings, important points, examples)
           - Using emojis that match the educational context and subject matter
        
        4. Cleaning up irrelevant content:
           - Remove any code blocks containing only data reports like ```[Data: Reports (19, 22, 21)]``` or ```[39]```
           - Remove any debugging information, log outputs, or system messages not relevant to the educational content
           - Remove any metadata markers or tags that aren't meant for the end user
        
        IMPORTANT RULES:
        - DO NOT add any new information or change the actual content
        - DO NOT alter the meaning of any statements
        - DO NOT change the language of the content
        - DO NOT remove any information from the original answer that is relevant to the educational content
        - DO remove irrelevant technical artifacts like data reports, debug logs, or system messages
        - DO NOT start the answer with wording like "Here is the answer to your question:" or "Here is ..." or "Enhanced Answer" or "Refined Answer"
        """
    )
    human_prompt = (
        """
        # Original answer
        {answer}
        
        # Reference material (if available)
        {reference}
        
        Please refine the original answer by:
        1. Properly formatting all mathematical formulas with LaTeX syntax (add $ or $$ for all possible formulas as appropriate)
        2. Adding **bold text** to important terms and concepts for better readability
        3. Including relevant emojis to make the content more engaging
        4. Removing any irrelevant content like data reports (e.g., ```[Data: Reports (19, 22, 21)]```), debug logs, or system messages
        5. Avoid starting the answer with wording like "Here is ..." or "Enhanced Answer" or "Refined Answer"
        
        Do not change the actual educational information or add new content.
        """
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    chain = prompt | llm | error_parser
    response = chain.invoke({"answer": answer, "reference": reference})
    return response