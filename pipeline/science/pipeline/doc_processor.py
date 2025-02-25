import os
import io
import hashlib
import fitz
import json
import requests
import base64
import time
from datetime import datetime, UTC

from dotenv import load_dotenv
from typing import Tuple, Dict
from pathlib import Path
from PIL import Image

from langchain_community.document_loaders import PyMuPDFLoader

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.images_understanding import (
    extract_image_context,
    upload_markdown_to_azure,
    upload_images_to_azure,
)
from pipeline.science.pipeline.utils import robust_search_for

import logging
logger = logging.getLogger("tutorpipeline.science.doc_processor")
load_dotenv()
# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


# Custom function to extract document objects from uploaded file
def extract_document_from_file(file_path):
    # Load the document
    loader = PyMuPDFLoader(file_path)
    document = loader.load()
    return document


# Function to process the PDF file
def process_pdf_file(file_path):
    # Process the document
    file = open(file_path, 'rb')
    file_bytes = file.read()
    document = extract_document_from_file(file_path)
    doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    return document, doc


# Function to save the file locally as a text file
def save_file_txt_locally(file_path, filename, embedding_folder):
    """
    Save the file (e.g., PDF) loaded as text into the GraphRAG_embedding_input_folder.
    """
    file = open(file_path, 'rb')
    file_bytes = file.read()

    # Define folder structure
    GraphRAG_embedding_folder = os.path.join(embedding_folder, "GraphRAG")
    GraphRAG_embedding_input_folder = os.path.join(GraphRAG_embedding_folder, "input")

    # Create folders if they do not exist
    os.makedirs(GraphRAG_embedding_input_folder, exist_ok=True)

    # Generate a shorter filename using hash, and it should be unique and consistent for the same file
    base_name = os.path.splitext(filename)[0]
    hashed_name = hashlib.md5(file_bytes).hexdigest()[:8]  # Use first 8 chars of hash
    output_file_path = os.path.join(GraphRAG_embedding_input_folder, f"{hashed_name}.txt")

    # Extract text from the PDF using the provided utility function
    document = extract_document_from_file(file_path)

    # Write the extracted text into a .txt file
    # If the file does not exist, it will be created
    if os.path.exists(output_file_path):
        print(f"File already exists: {output_file_path}")
        return

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for doc in document:
                # Each doc is expected to have a `page_content` attribute if it's a Document object
                if hasattr(doc, 'page_content') and doc.page_content:
                    # Write the text, followed by a newline for clarity
                    f.write(doc.page_content.strip() + "\n")
        print(f"Text successfully saved to: {output_file_path}")
    except OSError as e:
        print(f"Error saving file: {e}")
        # Create a mapping file to track original filenames if needed
        mapping_file = os.path.join(GraphRAG_embedding_folder, "filename_mapping.json")
        try:
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
            else:
                mapping = {}
            mapping[hashed_name] = base_name
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
        except Exception as e:
            logger.exception(f"Error saving filename mapping: {e}")
        raise
    return


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
    react_annotations = []
    for page_num, page in enumerate(doc):
        for excerpt in excerpts:
            text_instances = robust_search_for(page, excerpt)
            # print(f"text_instances: {text_instances}")
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
                    react_annotations.append(
                        {
                            "content": {
                                "text": excerpt,
                            },
                            "rects": [
                                {
                                    "x1": inst.x0,
                                    "y1": inst.y0,
                                    "x2": inst.x1,
                                    "y2": inst.y1,
                                    "width": inst.x1 - inst.x0,
                                    "height": inst.y1 - inst.y0,
                                }
                            ],
                            "pageNumber": page_num + 1,
                        }
                    )
    # print(f"annotations: {annotations}")
    return annotations, react_annotations


class mdDocumentProcessor:
    """
    Class to handle markdown document extraction processing and maintain document state without ST dependency.
    """
    def __init__(self):
        self.md_document = ""

    def set_md_document(self, content: str):
        """Set the markdown document content."""
        self.md_document = content

    def append_md_document(self, content: str):
        """Append content to the markdown document."""
        self.md_document += content.strip() + "\n"

    def get_md_document(self) -> str:
        """Get the current markdown document content."""
        return self.md_document


def extract_pdf_content_to_markdown_via_api(
    file_path: str | Path,
    output_dir: str | Path,
) -> Tuple[str, Dict[str, Image.Image], str]:
    """
    Extract text and images from a PDF file using the Marker API and save them to the specified directory.

    Args:
        file_path: Path to the input PDF file
        output_dir: Directory where images and markdown will be saved

    Returns:
        Tuple containing:
        - Path to the saved markdown file
        - Dictionary of image names and their PIL Image objects
        - Markdown content (str)

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        OSError: If output directory cannot be created
        Exception: For API errors or processing failures
    """
    # Load environment variables and validate input
    load_dotenv()
    API_KEY = os.getenv("MARKER_API_KEY")
    if not API_KEY:
        raise ValueError("MARKER_API_KEY not found in environment variables")

    # Validate input PDF exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Generate hash ID for the file
    with open(file_path, 'rb') as f:
        file_id = hashlib.md5(f.read()).hexdigest()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    API_URL = "https://www.datalab.to/api/v1/marker"

    # Submit the file to API
    with open(file_path, "rb") as f:
        form_data = {
            "file": (str(file_path), f, "application/pdf"),
            "langs": (None, "English"),
            "force_ocr": (None, False),
            "paginate": (None, False),
            "output_format": (None, "markdown"),
            "use_llm": (None, False),
            "strip_existing_ocr": (None, False),
            "disable_image_extraction": (None, False),
        }
        headers = {"X-Api-Key": API_KEY}
        response = requests.post(API_URL, files=form_data, headers=headers)

    # Check initial response
    data = response.json()
    if not data.get("success"):
        raise Exception(f"API request failed: {data.get('error')}")

    request_check_url = data.get("request_check_url")
    print("Submitted request. Polling for results...")

    # Poll until processing is complete
    max_polls = 300
    poll_interval = 2
    result = None

    for i in range(max_polls):
        time.sleep(poll_interval)
        poll_response = requests.get(request_check_url, headers=headers)
        result = poll_response.json()
        status = result.get("status")
        print(f"Poll {i+1}: status = {status}")
        if status == "complete":
            break
    else:
        raise Exception("The request did not complete within the expected time.")

    # Process and save results
    if not result.get("success"):
        raise Exception(f"Processing failed: {result.get('error')}")

    # Save markdown content with hash ID
    markdown = result.get("markdown", "")
    md_path = output_dir / f"{file_id}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown)
    logger.info(f"Saved markdown to: {md_path}")

    # Process and save images
    saved_images: Dict[str, Image.Image] = {}
    images = result.get("images", {})

    if images:
        print(f"Processing {len(images)} images...")
        for filename, b64data in images.items():
            try:
                # Create PIL Image from base64 data
                image_data = base64.b64decode(b64data)
                img = Image.open(io.BytesIO(image_data))

                # Create a valid filename
                safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.'))
                output_path = output_dir / safe_filename

                # Save the image
                img.save(output_path)
                saved_images[filename] = img
                print(f"Saved image: {output_path}")
            except Exception as e:
                logger.exception(f"Error saving image {filename}: {e}")
    else:
        print("No images were returned with the result")

    # Save markdown file and images to Azure Blob Storage
    try:
        upload_markdown_to_azure(output_dir, file_path)
        upload_images_to_azure(output_dir, file_path)
    except Exception as e:
        logger.exception(f"Error uploading markdown and images to Azure Blob Storage: {e}")

    # Extract image context
    try:
        config = load_config()
        chunk_size = config['embedding']['chunk_size']
        extract_image_context(output_dir, file_path=file_path)
    except Exception as e:
        logger.exception(f"Error extracting image context: {e}")
        raise Exception(f"Error extracting image context: {e}")

    return str(md_path), saved_images, markdown


def extract_pdf_content_to_markdown(
    file_path: str | Path,
    output_dir: str | Path,
) -> Tuple[str, Dict[str, Image.Image]]:
    """
    Extract text and images from a PDF file and save them to the specified directory.

    Args:
        file_path: Path to the input PDF file
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
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.settings import settings

    # Validate input PDF exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate hash ID for the file
        with open(file_path, 'rb') as f:
            file_id = hashlib.md5(f.read()).hexdigest()

        # Initialize converter and process PDF
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        rendered = converter(str(file_path))
        text, _, images = text_from_rendered(rendered)

        # Save markdown content with hash ID
        md_path = output_dir / f"{file_id}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Saved markdown to: {md_path}")

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
                    logger.exception(f"Error saving image {img_name}: {str(e)}")
        else:
            print("No images found in the PDF")

        # Save markdown file and images to Azure Blob Storage
        try:
            upload_markdown_to_azure(output_dir, file_path)
            upload_images_to_azure(output_dir, file_path)
        except Exception as e:
            logger.exception(f"Error uploading markdown and images to Azure Blob Storage: {e}")

        # Extract image context
        try:
            config = load_config()
            chunk_size = config['embedding']['chunk_size']
            extract_image_context(output_dir, file_path=file_path)
        except Exception as e:
            logger.exception(f"Error extracting image context: {e}")
            raise Exception(f"Error extracting image context: {e}")

        return str(md_path), saved_images, text

    except Exception as e:
        logger.exception(f"Error processing PDF: {str(e)}")
        raise Exception(f"Error processing PDF: {str(e)}")
