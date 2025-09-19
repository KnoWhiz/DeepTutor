#!/usr/bin/env python3
"""
Test script for the _is_pdf_image_only function.

This script tests the _is_pdf_image_only function against all PDF files
in the specified directory to determine which PDFs are image-only and
which contain extractable text.

Usage:
    python pdf_image_only_test.py

The script will:
1. Scan all PDF files in the target directory
2. Test each PDF with the _is_pdf_image_only function
3. Generate a detailed report with statistics
4. Save results to a log file and text file
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add the parent directory to the path to import from pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pipeline.science.pipeline.doc_processor import _is_pdf_image_only

# Configure logging
def setup_logging():
    """Set up logging configuration for the test script."""
    log_file = "pdf_image_only_test.log"
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files from the specified directory.
    
    Args:
        directory: Path to the directory containing PDF files
        
    Returns:
        List of full paths to PDF files
    """
    pdf_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all PDF files recursively
    for pdf_file in directory_path.rglob("*.pdf"):
        if pdf_file.is_file():
            pdf_files.append(str(pdf_file))
    
    return sorted(pdf_files)

def test_pdf_file(pdf_path: str, logger: logging.Logger) -> Tuple[bool, str, float]:
    """
    Test a single PDF file with the _is_pdf_image_only function.
    
    Args:
        pdf_path: Path to the PDF file to test
        logger: Logger instance for logging
        
    Returns:
        Tuple of (is_image_only, status_message, processing_time)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Testing: {os.path.basename(pdf_path)}")
        
        # Test the function
        is_image_only = _is_pdf_image_only(pdf_path)
        
        processing_time = time.time() - start_time
        
        if is_image_only:
            status = "IMAGE-ONLY (requires OCR)"
        else:
            status = "TEXT-BASED (extractable text found)"
        
        logger.info(f"Result: {status} (processed in {processing_time:.2f}s)")
        
        return is_image_only, status, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"ERROR: {str(e)}"
        logger.error(f"Failed to process {os.path.basename(pdf_path)}: {error_msg}")
        return None, error_msg, processing_time

def generate_report(results: List[Tuple[str, bool, str, float]], logger: logging.Logger) -> str:
    """
    Generate a detailed report of the test results.
    
    Args:
        results: List of (file_path, is_image_only, status, processing_time) tuples
        logger: Logger instance for logging
        
    Returns:
        Formatted report string
    """
    total_files = len(results)
    image_only_count = sum(1 for _, is_image_only, _, _ in results if is_image_only is True)
    text_based_count = sum(1 for _, is_image_only, _, _ in results if is_image_only is False)
    error_count = sum(1 for _, is_image_only, _, _ in results if is_image_only is None)
    
    total_time = sum(processing_time for _, _, _, processing_time in results)
    avg_time = total_time / total_files if total_files > 0 else 0
    
    report = []
    report.append("=" * 80)
    report.append("PDF IMAGE-ONLY DETECTION TEST REPORT")
    report.append("=" * 80)
    report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total PDF Files Tested: {total_files}")
    report.append(f"Image-Only PDFs: {image_only_count} ({image_only_count/total_files*100:.1f}%)")
    report.append(f"Text-Based PDFs: {text_based_count} ({text_based_count/total_files*100:.1f}%)")
    report.append(f"Errors: {error_count} ({error_count/total_files*100:.1f}%)")
    report.append(f"Total Processing Time: {total_time:.2f} seconds")
    report.append(f"Average Processing Time: {avg_time:.2f} seconds per file")
    report.append("")
    
    # Detailed results
    report.append("DETAILED RESULTS:")
    report.append("-" * 80)
    report.append(f"{'Filename':<50} {'Status':<25} {'Time (s)':<10}")
    report.append("-" * 80)
    
    for file_path, is_image_only, status, processing_time in results:
        filename = os.path.basename(file_path)
        # Truncate filename if too long
        if len(filename) > 47:
            filename = filename[:44] + "..."
        
        report.append(f"{filename:<50} {status:<25} {processing_time:<10.2f}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def save_results_to_file(report: str, results: List[Tuple[str, bool, str, float]]):
    """
    Save the test results to a text file.
    
    Args:
        report: The formatted report string
        results: List of detailed results
    """
    output_file = "pdf_image_only_test_results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n")
        f.write("RAW DATA (for further analysis):\n")
        f.write("-" * 40 + "\n")
        f.write("file_path,is_image_only,status,processing_time\n")
        
        for file_path, is_image_only, status, processing_time in results:
            # Escape commas in file paths and status messages
            escaped_path = file_path.replace(",", ";")
            escaped_status = status.replace(",", ";")
            f.write(f"{escaped_path},{is_image_only},{escaped_status},{processing_time}\n")
    
    print(f"Results saved to: {output_file}")

def main():
    """Main function to run the PDF image-only detection test."""
    # Set up logging
    logger = setup_logging()
    
    # Target directory
    target_directory = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers"
    
    logger.info("Starting PDF Image-Only Detection Test")
    logger.info(f"Target directory: {target_directory}")
    
    try:
        # Get all PDF files
        logger.info("Scanning for PDF files...")
        pdf_files = get_pdf_files(target_directory)
        
        if not pdf_files:
            logger.warning("No PDF files found in the target directory")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to test")
        
        # Test each PDF file
        results = []
        start_time = time.time()
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Progress: {i}/{len(pdf_files)}")
            
            is_image_only, status, processing_time = test_pdf_file(pdf_path, logger)
            results.append((pdf_path, is_image_only, status, processing_time))
            
            # Add a small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        logger.info(f"All tests completed in {total_time:.2f} seconds")
        
        # Generate and display report
        report = generate_report(results, logger)
        print("\n" + report)
        
        # Save results to file
        save_results_to_file(report, results)
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
