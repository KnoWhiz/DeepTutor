import os
import hashlib
import shutil
import fitz
import tiktoken
import json
import langid
import streamlit as st

from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from streamlit_float import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.config import load_config
from pipeline.api_handler import ApiHandler

# from config import load_config
# from api_handler import ApiHandler

os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'


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
def generate_course_id(file):
    file_hash = hashlib.md5(file).hexdigest()
    return file_hash


# Custom function to extract document objects from uploaded file
def extract_documents_from_file(file, filename):
    input_dir = './input_files/'

    # Create the input_files directory if it doesn't exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # Save the file to the input_files directory with the original filename
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(file)

    # Load the document
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # file path
    file_paths = os.path.join(input_dir, filename)

    return documents, file_paths


# Find pages with the given excerpts in the document
def find_pages_with_excerpts(doc, excerpts):
    pages_with_excerpts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for excerpt in excerpts:
            text_instances = robust_search_for(page, excerpt)
            if text_instances:
                pages_with_excerpts.append(page_num)
                break
    return pages_with_excerpts if pages_with_excerpts else [0]


# Get the highlight information for the given excerpts
def get_highlight_info(doc, excerpts):
    annotations = []
    for page_num, page in enumerate(doc):
        for excerpt in excerpts:
            text_instances = robust_search_for(page, excerpt)
            if text_instances:
                for inst in text_instances:
                    annotations.append({
                        "page": page_num + 1,
                        "x": inst.x0,
                        "y": inst.y0,
                        "width": inst.x1 - inst.x0,
                        "height": inst.y1 - inst.y0,
                        "color": "red",
                    })
    return annotations


# Add new helper functions
def count_tokens(text, model_name='gpt-4o'):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def truncate_chat_history(chat_history, model_name='gpt-4o'):
    """Only keep messages that fit within token limit"""
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advance'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)
    total_tokens = 0
    truncated_history = []
    for message in reversed(chat_history):
        if(message['role'] == 'assistant' or message['role'] == 'user'):
            message = str(message)
            message_tokens = count_tokens(message, model_name)
            if total_tokens + message_tokens > max_tokens:
                break
            truncated_history.insert(0, message)
            total_tokens += message_tokens
    return truncated_history


def truncate_document(_document, model_name='gpt-4o'):
    """Only keep beginning of document that fits token limit"""
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advance'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)
    _document = str(_document)
    document_tokens = count_tokens(_document, model_name)
    if document_tokens > max_tokens:
        _document = _document[:max_tokens]
    return _document


def get_embedding_models(embedding_model_type, para):
    para = para
    api = ApiHandler(para)
    embedding_model_default = api.embedding_models['default']['instance']
    if embedding_model_type == 'default':
        return embedding_model_default
    else:
        return embedding_model_default


def get_llm(llm_type, para):
    para = para
    api = ApiHandler(para)
    llm_basic = api.models['basic']['instance']
    llm_advance = api.models['advance']['instance']
    llm_creative = api.models['creative']['instance']
    if llm_type == 'basic':
        return llm_basic
    elif llm_type == 'advance':
        return llm_advance
    elif llm_type == 'creative':
        return llm_creative
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
    Translate the following content to {target_lang} language.
    Maintain the meaning and original formatting, including markdown syntax, LaTeX formulas, and emojis.
    Only translate the text content - do not modify any formatting, code, or special syntax.
    """
    
    human_prompt = """
    Content to translate:
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
        raise ValueError(f"Error processing PDF: {str(e)}")
    
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()


def extract_pdf_content_to_markdown(
    pdf_path: str | Path,
    output_dir: str | Path,
) -> Tuple[str, Dict[str, Image.Image]]:
    """
    Extract text and images from a PDF file and save them to the specified directory.
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory where images and markdown will be saved
    
    Returns:
        Tuple containing:
        - Path to the saved markdown file
        - Dictionary of image names and their PIL Image objects
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        OSError: If output directory cannot be created
        Exception: For other processing errors
    """
    # Validate input PDF exists
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize converter and process PDF
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        rendered = converter(str(pdf_path))
        text, _, images = text_from_rendered(rendered)

        # Save markdown content
        md_path = output_dir / f"{pdf_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved markdown to: {md_path}")

        # Save images
        saved_images = {}
        if images:
            print(f"Saving {len(images)} images to {output_dir}")
            for img_name, img in images.items():
                try:
                    # Create a valid filename from the image name
                    safe_filename = "".join(c for c in img_name if c.isalnum() or c in ('-', '_', '.'))
                    output_path = output_dir / safe_filename
                    
                    # Save the image
                    img.save(output_path)
                    saved_images[img_name] = img
                    print(f"Saved image: {output_path}")
                except Exception as e:
                    print(f"Error saving image {img_name}: {str(e)}")
        else:
            print("No images found in the PDF")

        return str(md_path), saved_images

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")