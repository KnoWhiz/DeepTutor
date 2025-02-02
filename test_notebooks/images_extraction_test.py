import fitz  # PyMuPDF
import os
from typing import List, Tuple
from pathlib import Path


def extract_images_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    min_size: Tuple[int, int] = (50, 50)  # Minimum size to filter out tiny images
) -> List[str]:
    """
    Extract images from a PDF file and save them to the specified directory.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory where images will be saved
        min_size: Minimum dimensions (width, height) for images to be extracted
    
    Returns:
        List of paths to the extracted image files
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        PermissionError: If output directory can't be created/accessed
        ValueError: If PDF file is invalid
    """
    # Convert paths to Path objects for better handling
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List to store paths of extracted images
    extracted_images: List[str] = []
    
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
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


if __name__ == "__main__":
    # Example usage
    pdf_path = "/Users/bingran_you/Downloads/Toward Keyword Generation through Large Language Models.pdf"
    output_dir = "extracted_images"
    
    try:
        # Set minimum size to 100x100 pixels
        extracted_files = extract_images_from_pdf(
            pdf_path, 
            output_dir,
            min_size=(100, 100)  # Adjust these values as needed
        )
        print(f"Successfully extracted {len(extracted_files)} images:")
        for image_path in extracted_files:
            print(f"- {image_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
