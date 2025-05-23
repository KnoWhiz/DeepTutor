import os
import sys
from pathlib import Path
import fitz
import pytest

# Add project root to Python path to allow absolute imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline.science.pipeline.rag_agent import get_page_content_from_file


def _create_sample_pdf(file_path: str | Path, page_texts: list[str]):
    """Utility function to create a sample PDF with the given page texts."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        # Insert text at (72, 72) points (1 inch margin)
        page.insert_text((72, 72), text)
    doc.save(str(file_path))
    doc.close()


def test_get_page_content_success(tmp_path):
    """Test that the function returns correct page content for valid inputs."""
    # Create sample PDF
    pdf_path = tmp_path / "sample.pdf"
    pages = ["Hello Page 1", "Second page content", "Third page"]
    _create_sample_pdf(pdf_path, pages)

    file_path_list = [str(pdf_path)]

    # Verify each page content
    for page_num, expected_text in enumerate(pages):
        content = get_page_content_from_file(file_path_list, 0, page_num)
        # We only assert that expected text is a substring (PyMuPDF may add newlines)
        assert expected_text in content


def test_file_index_out_of_range(tmp_path):
    """Test that an IndexError is raised when file_index is invalid."""
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path, ["Only one page"])

    file_path_list = [str(pdf_path)]
    with pytest.raises(IndexError):
        get_page_content_from_file(file_path_list, 5, 0)  # invalid file_index


def test_page_number_out_of_range(tmp_path):
    """Test that a ValueError is raised when page_number is invalid."""
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path, ["Only one page"])

    file_path_list = [str(pdf_path)]
    with pytest.raises(ValueError):
        get_page_content_from_file(file_path_list, 0, 10)  # invalid page_number


def test_file_not_found(tmp_path):
    """Test that a FileNotFoundError is raised when the file does not exist."""
    # Provide a non-existent file path
    fake_path = tmp_path / "nonexistent.pdf"
    file_path_list = [str(fake_path)]
    with pytest.raises(FileNotFoundError):
        get_page_content_from_file(file_path_list, 0, 0) 