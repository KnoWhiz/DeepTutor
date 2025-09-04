"""
PDF to Text Converter utility.

This module provides functionality to convert PDF files to text format
for processing with the Gemini CLI.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PDFConverter:
    """Utility class for converting PDF files to text."""
    
    @staticmethod
    def pdf_to_text(pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert a PDF file to text format.
        
        Args:
            pdf_path: Path to the input PDF file
            output_path: Optional path for the output text file. If None, 
                        creates a .txt file with the same name as the PDF
        
        Returns:
            Path to the created text file
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If there's an error during conversion
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        # Determine output path
        if output_path is None:
            output_path = pdf_path.with_suffix(".txt")
        else:
            output_path = Path(output_path)
        
        try:
            # Open the PDF file
            doc = fitz.open(str(pdf_path))
            text_content = []
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Add page separator for multi-page documents
                if page_num > 0:
                    text_content.append(f"\n--- Page {page_num + 1} ---\n")
                
                text_content.append(text)
            
            # Close the document
            doc.close()
            
            # Combine all text
            full_text = "".join(text_content)
            
            # Write to output file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            logger.info(f"Successfully converted PDF to text: {pdf_path} -> {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error converting PDF to text: {str(e)}")
            raise Exception(f"Failed to convert PDF to text: {str(e)}")
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text content from a PDF file without saving to disk.
        
        Args:
            pdf_path: Path to the input PDF file
            
        Returns:
            Extracted text content as a string
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If there's an error during extraction
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        try:
            # Open the PDF file
            doc = fitz.open(str(pdf_path))
            text_content = []
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Add page separator for multi-page documents
                if page_num > 0:
                    text_content.append(f"\n--- Page {page_num + 1} ---\n")
                
                text_content.append(text)
            
            # Close the document
            doc.close()
            
            # Combine all text
            full_text = "".join(text_content)
            
            logger.info(f"Successfully extracted text from PDF: {pdf_path}")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")


def convert_uploaded_pdf(uploaded_file, working_dir: str) -> str:
    """
    Convert an uploaded PDF file (from Streamlit) to text format.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        working_dir: Directory to save the converted text file
        
    Returns:
        Path to the created text file
    """
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded PDF temporarily
    pdf_path = working_dir / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert to text
    converter = PDFConverter()
    text_path = converter.pdf_to_text(str(pdf_path))
    
    # Remove the temporary PDF file
    pdf_path.unlink()
    
    logger.info(f"Converted uploaded PDF to text: {text_path}")
    return text_path


if __name__ == "__main__":
    # Test the converter
    test_pdf = "/path/to/test.pdf"  # Replace with actual test PDF path
    converter = PDFConverter()
    
    try:
        text_file = converter.pdf_to_text(test_pdf)
        print(f"Successfully converted to: {text_file}")
    except Exception as e:
        print(f"Error: {e}") 