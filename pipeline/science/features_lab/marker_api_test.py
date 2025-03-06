import os
import io
import time
import requests
import base64
import logging
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def extract_pdf_content_to_markdown_via_api(
    file_path: str | Path,
    output_dir: str | Path,
) -> Tuple[str, Dict[str, Image.Image]]:
    """
    Extract text and images from a PDF file using the Marker API and save them to the specified directory.

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
    logger.info(f"Submitted request. Polling for results...")

    # Poll until processing is complete
    max_polls = 300
    poll_interval = 2
    result = None

    for i in range(max_polls):
        time.sleep(poll_interval)
        poll_response = requests.get(request_check_url, headers=headers)
        result = poll_response.json()
        status = result.get("status")
        logger.info(f"Poll {i+1}: status = {status}")
        if status == "complete":
            break
    else:
        raise Exception("The request did not complete within the expected time.")

    # Process and save results
    if not result.get("success"):
        raise Exception(f"Processing failed: {result.get('error')}")

    # Save markdown content
    markdown = result.get("markdown", "")
    md_path = output_dir / f"{file_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown)
    logger.info(f"Saved markdown to: {md_path}")

    # Process and save images
    saved_images: Dict[str, Image.Image] = {}
    images = result.get("images", {})

    if images:
        logger.info(f"Processing {len(images)} images...")
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
                logger.info(f"Saved image: {output_path}")
            except Exception as e:
                logger.info(f"Error saving image {filename}: {e}")
    else:
        logger.info("No images were returned with the result")

    return str(md_path), saved_images

if __name__ == "__main__":
    # Example usage
    file_path = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/science.1189075.pdf"
    output_dir = "markdown_output"

    try:
        md_path, saved_images = extract_pdf_content_to_markdown_via_api(file_path, output_dir)
        logger.info(f"Successfully processed PDF. Markdown saved to: {md_path}")
        logger.info(f"Number of images extracted: {len(saved_images)}")
    except Exception as e:
        logger.info(f"Error: {str(e)}")