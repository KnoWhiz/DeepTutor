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
from pipeline.science.pipeline.embeddings import get_embedding_models

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


class Question:
    """
    Represents a question with its text, language, and type information.

    Attributes:
        text (str): The text content of the question
        language (str): The detected language of the question (e.g., "English")
        question_type (str): The type of question (e.g., "local" or "global" or "image")
        special_context (str): Special context for the question
        answer_planning (dict): Planning information for constructing the answer
    """

    def __init__(self, text="", language="English", question_type="global", special_context="", answer_planning=None, image_url=None):
        """
        Initialize a Question object.

        Args:
            text (str): The text content of the question
            language (str): The language of the question
            question_type (str): The type of the question (local or global or image)
            special_context (str): Special context for the question
            answer_planning (dict): Planning information for constructing the answer
            image_url (str): The image url for the image question
        """
        self.text = text
        self.language = language
        if question_type not in ["local", "global", "image"]:
            self.question_type = "global"
        else:
            self.question_type = question_type

        self.special_context = special_context
        self.answer_planning = answer_planning or {}
        self.image_url = image_url   # This is the image url for the image question

    def __str__(self):
        """Return string representation of the Question."""
        return f"Question(text='{self.text}', language='{self.language}', type='{self.question_type}', image_url='{self.image_url}')"

    def to_dict(self):
        """Convert Question object to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "question_type": self.question_type,
            "special_context": self.special_context,
            "answer_planning": self.answer_planning,
            "image_url": str(self.image_url)
        }


def generators_list_stream_response(generators_list):
    """
    Stream response from a list of generators.
    """
    for generator in generators_list:
        for chunk in generator:
            yield chunk


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
        if key == "0. session_id" or key == "0. session_type" or key == "0. new_message_id":
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
        max_tokens = int(api.models[model_level]['context_window'] / 3)
    else:
        max_tokens = token_limit / 3

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


def get_llm(llm_type, para, stream=False):
    para = para
    api = ApiHandler(para, stream)
    llm_basic = api.models['basic']['instance']
    llm_advanced = api.models['advanced']['instance']
    llm_creative = api.models['creative']['instance']
    llm_backup = api.models['backup']['instance']
    if llm_type == 'basic':
        logger.info(f"Using basic LLM: {llm_basic}")
        return llm_basic
    elif llm_type == 'advanced':
        logger.info(f"Using advanced LLM: {llm_advanced}")
        return llm_advanced
    elif llm_type == 'creative':
        logger.info(f"Using creative LLM: {llm_creative}")
        return llm_creative
    elif llm_type == 'backup':
        logger.info(f"Using backup LLM: {llm_backup}")
        return llm_backup
    return llm_basic


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


# def responses_refine(answer, reference=''):
#     # return answer
#     return answer


def responses_refine(answer, reference='', stream=True):
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
           - Surround block/display formulas with double dollar signs (\n$$formula\n$$)
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
    if stream:
        response = chain.stream({"answer": answer, "reference": reference})
    else:
        response = chain.invoke({"answer": answer, "reference": reference})
    return response


def clean_translation_prefix(text):
    """
    Clean translation prefixes from text in all supported languages.
    
    Args:
        text: The text to clean
        
    Returns:
        The cleaned text with translation prefixes removed
    """
    if not text:
        return text
        
    # First, handle the specific case that's still occurring
    if "当然可以！以下是翻译内容：" in text:
        text = text.replace("当然可以！以下是翻译内容：", "").strip()
    
    # Common starter words in different languages
    starters = (
        # English starters
        r"Sure|Certainly|Of course|Here|Yes|Okay|"
        # Chinese starters
        r"当然|好的|是的|这是|以下是|"
        # Spanish starters
        r"Claro|Seguro|Por supuesto|Aquí|Sí|"
        # French starters
        r"Bien sûr|Certainement|Oui|Voici|Voilà|"
        # German starters
        r"Natürlich|Sicher|Klar|Hier ist|Ja|"
        # Japanese starters
        r"もちろん|はい|ここに|"
        # Korean starters
        r"물론|네|여기|"
        # Hindi starters
        r"ज़रूर|हां|यहां|निश्चित रूप से|"
        # Portuguese starters
        r"Claro|Certamente|Sim|Aqui|"
        # Italian starters
        r"Certo|Sicuro|Sì|Ecco"
    )
    
    # Translation-related terms in different languages
    translation_terms = (
        # English
        r"translation|translated|"
        # Chinese
        r"翻译|译文|"
        # Spanish
        r"traducción|traducido|"
        # French
        r"traduction|traduit|"
        # German
        r"Übersetzung|übersetzt|"
        # Japanese
        r"翻訳|"
        # Korean
        r"번역|"
        # Hindi
        r"अनुवाद|"
        # Portuguese
        r"tradução|traduzido|"
        # Italian
        r"traduzione|tradotto"
    )
    
    # More aggressive regex pattern for translation prefixes that handles newlines better
    pattern = rf'^({starters})[^A-Za-z0-9]*?({translation_terms})[^:]*?:[ \n]*'
    
    # Apply the cleanup with a more permissive DOTALL flag to handle newlines
    result = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)
    
    # Additional cleanup for common patterns with newlines
    result = re.sub(r'^[^:]+:[ \n]+', '', result, flags=re.DOTALL)
    
    # Remove any leading newlines after cleanup
    result = result.lstrip('\n')
    
    return result.strip()


def extract_answer_content(message_content):
    sources = {}    # {source_string: source_score}
    source_pages = {}    # {source_page_string: source_page_score}
    source_annotations = {}    # {source_annotation_string: source_annotation_data}
    refined_source_pages = {}    # {refined_source_page_string: refined_source_page_score}
    refined_source_index = {}    # {refined_source_index_string: refined_source_index_score}
    follow_up_questions = []

    # Extract the main answer (content between <response> tags)
    # The logic is: if we have <response> tags, we extract the content between them
    # Otherwise, we extract the content between <original_response> and </original_response> tags
    # If we have neither, we extract the content between <thinking> and </thinking> tags
    # If we have none of the above, we return an empty string
    answer = ""
    thinking = ""
    response_match = re.search(r'<response>(.*?)</response>', message_content, re.DOTALL)
    original_response_match = re.search(r'<original_response>(.*?)</original_response>', message_content, re.DOTALL)
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', message_content, re.DOTALL)
    if response_match:
        answer = response_match.group(1).strip()
    elif original_response_match:
        answer = original_response_match.group(1).strip()
    elif thinking_match:
        answer = thinking_match.group(1).strip()
    else:
        answer = ""

    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Extract follow-up questions (content between <followup_question> tags)
    followup_matches = re.finditer(r'<followup_question>(.*?)</followup_question>', message_content, re.DOTALL)
    for match in followup_matches:
        question = match.group(1).strip()
        if question:
            # Remove any residual XML tags
            question = re.sub(r'<followup_question>.*?</followup_question>', '', question)

            # Apply the clean_translation_prefix function
            question = clean_translation_prefix(question)

            follow_up_questions.append(question)

    # Extract sources (content between <source> tags)
    source_matches = re.finditer(r'<source>(.*?)</source>', message_content, re.DOTALL)
    for match in source_matches:
        source_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', source_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                sources[key] = float(value)
            except ValueError:
                # If conversion fails, use default value of 0.0 and log the issue
                logger.warning(f"Failed to parse source value '{value}' for key '{key[:50]}...', using default 0.0")
                sources[key] = 0.0

    # Extract source pages (content between <source_page> tags)
    source_page_matches = re.finditer(r'<source_page>(.*?)</source_page>', message_content, re.DOTALL)
    for match in source_page_matches:
        source_page_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', source_page_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                source_pages[key] = float(value)
            except ValueError:
                # If conversion fails, use default value of 1.0 and log the issue
                logger.warning(f"Failed to parse source_page value '{value}' for key '{key[:50]}...', using default 1.0")
                source_pages[key] = 1.0
    
    # Extract source annotations (content between <source_annotations> tags)
    source_annotation_matches = re.finditer(r'<source_annotations>(.*?)</source_annotations>', message_content, re.DOTALL)
    # logger.info(f"Starting source annotations extraction")
    annotation_count = 0
    
    for match in source_annotation_matches:
        annotation_count += 1
        source_annotation_content = match.group(1).strip()
        # logger.info(f"Processing source annotation #{annotation_count}: {source_annotation_content[:100]}...")
        
        # Find the last occurrence of "}{"
        last_separator_index = source_annotation_content.rfind("}{")
        
        if last_separator_index != -1:
            # Extract text content (everything before the last "}{")
            text_content = source_annotation_content[:last_separator_index].strip()
            # Extract metadata string (everything after the last "}{" plus an opening brace)
            metadata_str = "{" + source_annotation_content[last_separator_index+1:].strip()
            logger.info(f"Extracted text content (first 50 chars): {text_content[:50]}...")
            logger.info(f"Extracted metadata string: {metadata_str[:100]}...")
            
            try:
                # Clean up any potential trailing/leading characters for JSON parsing
                metadata_str = metadata_str.rstrip("}")
                if not metadata_str.endswith("}"):
                    metadata_str += "}"
                
                # Convert the metadata string to a dictionary
                metadata = json.loads(metadata_str)
                source_annotations[text_content] = metadata
                logger.info(f"Successfully parsed metadata: page_num={metadata.get('page_num', 'N/A')}, success={metadata.get('success', False)}, similarity={metadata.get('similarity', 0)}")
                
                # Also extract page information for source_pages if successful
                if 'page_num' in metadata and metadata.get('success', False) and metadata.get('similarity', 0) > 0:
                    page_num = metadata['page_num']
                    # Only add if page_num is valid (not -1)
                    if page_num >= 0:
                        page_key = f"page_{page_num + 1}"
                        source_pages[page_key] = metadata['similarity']
                        logger.info(f"Added to source_pages: {page_key} with similarity {metadata['similarity']}")
                    else:
                        logger.warning(f"Skipped invalid page_num: {page_num}")
            except json.JSONDecodeError as e:
                # If JSON parsing fails, log it and store the raw string
                # logger.info(f"Failed to parse source annotation metadata: {e}")
                logger.info(f"Problematic JSON string: {metadata_str}")
                source_annotations[text_content] = metadata_str
        else:
            logger.info(f"Could not find text/metadata separator in annotation: {source_annotation_content[:50]}...")
    
    logger.info(f"Completed source annotations extraction. Found {annotation_count} annotations, extracted {len(source_annotations)} valid entries")

    # Extract source annotations (content between <source_annotations> tags)
    source_annotation_matches = re.finditer(r'<source_annotations>(.*?)</source_annotations>', message_content, re.DOTALL)
    for match in source_annotation_matches:
        source_annotation_content = match.group(1).strip()
        # Extract the content and JSON data
        # The format is {text content}{JSON data}
        annotation_match = re.match(r'\{(.*?)\}\{(.*)\}', source_annotation_content, re.DOTALL)
        if annotation_match:
            text = annotation_match.group(1)
            json_str = annotation_match.group(2)
            
            # Make sure the JSON string is valid by checking for proper closing brace
            if not json_str.endswith('}'):
                json_str += '}'
                
            try:
                # Parse the JSON data directly from the string
                # Replace single quotes with double quotes for valid JSON
                # And handle Python bool literals
                processed_json = json_str.replace("'", '"').replace("True", "true").replace("False", "false")
                data = {
                    "page_num": int(re.search(r'"page_num":\s*(\d+)', processed_json).group(1)),
                    "start_char": int(re.search(r'"start_char":\s*(\d+)', processed_json).group(1)),
                    "end_char": int(re.search(r'"end_char":\s*(\d+)', processed_json).group(1)),
                    "success": True if "true" in processed_json else False,
                    "similarity": float(re.search(r'"similarity":\s*([\d\.]+)', processed_json).group(1))
                }
                source_annotations[text] = data
            except Exception as e:
                # logger.error(f"Failed to extract source annotation data: {e}")
                # Fall back to storing as string if parsing fails
                source_annotations[text] = json_str

    # Extract refined source pages (content between <refined_source_page> tags)
    refined_source_page_matches = re.finditer(r'<refined_source_page>(.*?)</refined_source_page>', message_content, re.DOTALL)
    for match in refined_source_page_matches:
        refined_source_page_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', refined_source_page_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                refined_source_pages[key] = float(value)
            except ValueError:
                # If conversion fails, use default value of 1.0 and log the issue
                logger.warning(f"Failed to parse refined_source_page value '{value}' for key '{key[:50]}...', using default 1.0")
                refined_source_pages[key] = 1.0

    # Extract refined source index (content between <refined_source_index> tags)
    refined_source_index_matches = re.finditer(r'<refined_source_index>(.*?)</refined_source_index>', message_content, re.DOTALL)
    for match in refined_source_index_matches:
        refined_source_index_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', refined_source_index_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float or int
                refined_source_index[key] = float(value)
            except ValueError:
                try:
                    # Try converting to int if float conversion fails
                    refined_source_index[key] = int(value)
                except ValueError:
                    # If both conversions fail, use default value of 0 and log the issue
                    logger.warning(f"Failed to parse refined_source_index value '{value}' for key '{key[:50]}...', using default 0")
                    refined_source_index[key] = 0

    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking


def extract_lite_mode_content(message_content):
    return extract_answer_content(message_content)


def extract_basic_mode_content(message_content):
    return extract_answer_content(message_content)


def extract_advanced_mode_content(message_content):
    return extract_answer_content(message_content)


def create_truncated_db(db: FAISS, embedding_type: str = "default") -> FAISS:
    """
    Create a new FAISS database with only the first 3 sentences of each chunk from the original database.
    
    Parameters:
        db (FAISS): The original FAISS database to truncate
        embedding_type (str): Type of embedding model to use ("default", "lite", or "small")
        
    Returns:
        FAISS: A new FAISS database with truncated content (first 3 sentences only)
    """
    import re
    
    # Initialize required components
    config = load_config()
    para = config["llm"]
    embeddings = get_embedding_models(embedding_type, para)
    
    # Get all documents from the original database - more reliable method
    # First try using similarity_search_with_score which will return all docs
    try:
        documents = []
        # Use a dummy query that should return all documents with their scores
        all_docs = db.similarity_search_with_score("", k=10000)  # Large k to get all docs
        for doc, _ in all_docs:
            documents.append(doc)
        
        # If no documents found, try direct access as fallback
        if not documents:
            documents = list(db.docstore._dict.values())
            
        logger.info(f"Found {len(documents)} documents in the database")
    except Exception as e:
        # Fallback to direct access method
        logger.warning(f"Error using similarity_search: {str(e)}. Falling back to direct access.")
        documents = list(db.docstore._dict.values())
    
    # Create new documents with truncated content
    truncated_documents = []
    
    for doc in documents:
        # Extract the first 3 sentences using a more reliable method
        content = doc.page_content
        
        # More sophisticated sentence splitting using regex
        # This pattern handles periods, question marks, and exclamation points
        # followed by spaces or end of string, but ignores common abbreviations
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        sentences = sentence_endings.split(content)
        
        # Handle cases with fewer than 3 sentences
        if len(sentences) <= 3:
            truncated_content = content
        else:
            # Get the first 3 sentences
            first_three = sentences[:3]
            # Check if the last sentence already has ending punctuation
            if first_three[-1].rstrip().endswith(('.', '!', '?')):
                truncated_content = ''.join(first_three)
            else:
                # Add a period if the last sentence doesn't end with punctuation
                truncated_content = ''.join(first_three) + '.'
        
        # Create a new document with the truncated content
        truncated_doc = Document(
            page_content=truncated_content,
            metadata=doc.metadata
        )
        truncated_documents.append(truncated_doc)
    
    # Create a new FAISS database with the truncated documents
    if not truncated_documents:
        logger.warning("No documents to truncate. Returning empty FAISS database.")
        # Create an empty FAISS database
        new_db = FAISS.from_documents(
            [Document(page_content="Empty placeholder", metadata={"source": "placeholder"})],
            embeddings
        )
        return new_db
    
    try:
        # Create the FAISS database with the truncated documents
        new_db = FAISS.from_documents(truncated_documents, embeddings)
        logger.info(f"Created truncated FAISS database with {len(truncated_documents)} documents")
        return new_db
    except Exception as e:
        logger.error(f"Error creating truncated FAISS database: {str(e)}")
        # Fallback to smaller models if available
        try:
            logger.info("Trying with 'small' embedding model...")
            embeddings_small = get_embedding_models("small", para)
            new_db = FAISS.from_documents(truncated_documents, embeddings_small)
            return new_db
        except Exception as e2:
            logger.error(f"Error with small embedding model: {str(e2)}")
            logger.info("Trying with 'lite' embedding model...")
            embeddings_lite = get_embedding_models("lite", para)
            new_db = FAISS.from_documents(truncated_documents, embeddings_lite)
            return new_db