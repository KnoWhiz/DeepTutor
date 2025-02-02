import os
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings

def extract_pdf_content(
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

if __name__ == "__main__":
    # Example usage
    pdf_path = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/science.1189075.pdf"
    output_dir = "output"
    
    try:
        md_path, saved_images = extract_pdf_content(pdf_path, output_dir)
        print(f"Successfully processed PDF. Markdown saved to: {md_path}")
        print(f"Number of images extracted: {len(saved_images)}")
    except Exception as e:
        print(f"Error: {str(e)}")