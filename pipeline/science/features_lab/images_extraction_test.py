import os
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings

import logging
logger = logging.getLogger("images_extraction_test.py")

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
    # Validate input PDF exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize converter and process PDF
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        rendered = converter(str(file_path))
        text, _, images = text_from_rendered(rendered)

        # Save markdown content
        md_path = output_dir / f"{file_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Saved markdown to: {md_path}")

        # Save images
        saved_images = {}
        if images:
            logger.info(f"Saving {len(images)} images to {output_dir}")
            for img_name, img in images.items():
                try:
                    # Create a valid filename from the image name
                    safe_filename = "".join(c for c in img_name if c.isalnum() or c in ('-', '_', '.'))
                    output_path = output_dir / safe_filename

                    # Save the image
                    img.save(output_path)
                    saved_images[img_name] = img
                    logger.info(f"Saved image: {output_path}")
                except Exception as e:
                    logger.exception(f"Error saving image {img_name}: {str(e)}")
        else:
            logger.info("No images found in the PDF")

        return str(md_path), saved_images

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    # Example usage
    # file_path = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/science.1189075.pdf"
    file_path = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/RankRAG- Unifying Context Ranking with  Retrieval-Augmented Generation in LLMs.pdf"
    output_dir = "markdown_output"

    try:
        md_path, saved_images = extract_pdf_content_to_markdown(file_path, output_dir)
        logger.info(f"Successfully processed PDF. Markdown saved to: {md_path}")
        logger.info(f"Number of images extracted: {len(saved_images)}")
    except Exception as e:
        logger.exception(f"Error: {str(e)}")