import fitz  # PyMuPDF
import os
from typing import List, Tuple
from pathlib import Path


def extract_images_from_pdf(
    pdf_doc: fitz.Document,
    output_dir: str | Path,
    min_size: Tuple[int, int] = (200, 200),  # Increased minimum size
    max_size: Tuple[int, int] = (4000, 4000),  # Maximum size to filter out too large images
    min_aspect_ratio: float = 0.2,  # Minimum width/height ratio
    max_aspect_ratio: float = 5.0,  # Maximum width/height ratio
) -> List[str]:
    """
    Extract meaningful figures from a PDF file while filtering out small icons and decorative elements.

    Args:
        pdf_doc: fitz.Document object
        output_dir: Directory where images will be saved
        min_size: Minimum dimensions (width, height) for images to be extracted
        max_size: Maximum dimensions (width, height) for images to be extracted
        min_aspect_ratio: Minimum width/height ratio to filter out too narrow images
        max_aspect_ratio: Maximum width/height ratio to filter out too wide images

    Returns:
        List of paths to the extracted image files

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
        pdf_document = pdf_doc
        image_counter = 1
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            images = page.get_images()
            
            for image_index, image in enumerate(images):
                # Get image data
                base_image = pdf_document.extract_image(image[0])
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Get image size and properties
                img_xref = image[0]
                pix = fitz.Pixmap(pdf_document, img_xref)
                width, height = pix.width, pix.height
                
                # Calculate aspect ratio
                aspect_ratio = width / height if height != 0 else 0
                
                # Skip images that don't meet the criteria
                if any([
                    # Size checks
                    width < min_size[0] or height < min_size[1],
                    width > max_size[0] or height > max_size[1],
                    # Aspect ratio checks
                    aspect_ratio < min_aspect_ratio,
                    aspect_ratio > max_aspect_ratio,
                    # Area check - filter out very small images
                    (width * height) < (min_size[0] * min_size[1]),
                    # Additional check for common icon sizes
                    (width in [16, 32, 48, 64] and height in [16, 32, 48, 64])
                ]):
                    continue
                
                # Generate simple sequential filename
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


if __name__ == "__main__":
    # Example usage
    # pdf_path = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/science.1189075.pdf"
    pdf_path = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/RankRAG- Unifying Context Ranking with  Retrieval-Augmented Generation in LLMs.pdf"
    pdf_doc = fitz.open(pdf_path)
    output_dir = "extracted_images"
    
    try:
        extracted_files = extract_images_from_pdf(
            pdf_doc,
            output_dir,
            min_size=(100, 100),  # Increased minimum size
            max_size=(4000, 4000),  # Maximum size limit
            min_aspect_ratio=0.2,
            max_aspect_ratio=5.0
        )
        print(f"Successfully extracted {len(extracted_files)} images:")
        for image_path in extracted_files:
            print(f"- {image_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
