"""Test script to verify source search functionality by loading a PDF and searching for its content."""

import os
import io
import fitz
import argparse
import sys
from pathlib import Path

import logging
logger = logging.getLogger("sources_search_test.py")

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import robust_search_for
from pipeline.science.pipeline.embeddings import get_embedding_models
# from frontend.state import state_process_pdf_file, extract_document_from_file

def create_searchable_chunks(doc, chunk_size: int) -> list:
    """
    Create searchable chunks from a PDF document.

    Args:
        doc: The PDF document object
        chunk_size: Maximum size of each text chunk in characters

    Returns:
        list: A list of Document objects containing the chunks
    """
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_blocks = []

        # Get text blocks that can be found via search
        for block in page.get_text("blocks"):
            text = block[4]  # The text content is at index 4
            # Clean up the text
            clean_text = text.strip()

            if clean_text:
                # Remove hyphenation at line breaks
                clean_text = clean_text.replace("-\n", "")
                # Normalize spaces
                clean_text = " ".join(clean_text.split())
                # Replace special characters that might cause issues
                replacements = {
                    # "−": "-",  # Replace unicode minus with hyphen
                    # "⊥": "_|_",  # Replace perpendicular symbol
                    # "≫": ">>",  # Replace much greater than
                    # "%": "",     # Remove percentage signs that might be formatting artifacts
                    # "→": "->",   # Replace arrow
                }
                for old, new in replacements.items():
                    clean_text = clean_text.replace(old, new)

                # Split into chunks of specified size
                while len(clean_text) > 0:
                    # Find a good break point near chunk_size characters
                    end_pos = min(chunk_size, len(clean_text))
                    if end_pos < len(clean_text):
                        # Try to break at a sentence or period
                        last_period = clean_text[:end_pos].rfind(". ")
                        if last_period > 0:
                            end_pos = last_period + 1
                        else:
                            # If no period, try to break at a space
                            last_space = clean_text[:end_pos].rfind(" ")
                            if last_space > 0:
                                end_pos = last_space

                    chunk_text = clean_text[:end_pos].strip()
                    if chunk_text:
                        text_blocks.append(Document(
                            page_content=chunk_text,
                            metadata={
                                "page": page_num,
                                "source": f"page_{page_num + 1}",
                                "chunk_index": len(text_blocks),  # Track position within page
                                "block_bbox": block[:4],  # Store block bounding box coordinates
                                "total_blocks_in_page": len(page.get_text("blocks")),
                                "relative_position": len(text_blocks) / len(page.get_text("blocks"))
                            }
                        ))
                    clean_text = clean_text[end_pos:].strip()

        chunks.extend(text_blocks)

    # Sort chunks by page number and then by chunk index
    chunks.sort(key=lambda x: (
        x.metadata.get("page", 0),
        x.metadata.get("chunk_index", 0)
    ))

    return chunks

def test_source_search(file_path: str, chunk_size: int = 512) -> None:
    """
    Test source search functionality by:
    1. Loading a PDF file
    2. Processing it into document and doc objects
    3. Splitting content into chunks
    4. Embedding the chunks
    5. For each chunk, verify it can be found in the source PDF

    Args:
        file_path: Path to the PDF file to test
        chunk_size: Maximum size of each text chunk in characters (default: 512)
    """
    logger.info(f"Testing source search for PDF: {file_path}")

    # Load the PDF file
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Process the PDF file
    document, doc = state_process_pdf_file(file_path)

    logger.info(f"Loaded PDF with {len(document)} document chunks")

    # Create chunks directly from the PDF content to ensure better matching
    chunks = create_searchable_chunks(doc, chunk_size)

    logger.info(f"Created {len(chunks)} searchable chunks from PDF")

    # Create temporary embedding folder
    temp_embedding_folder = "temp_embeddings"
    os.makedirs(temp_embedding_folder, exist_ok=True)

    # Load config and get embedding model
    config = load_config()
    embeddings = get_embedding_models("default", config["llm"])

    # Create vector store
    db = FAISS.from_documents(chunks, embeddings)

    # Get all chunks from the vector store by doing a similarity search with each chunk
    retrieved_chunks = []
    logger.info("\nRetrieving chunks from vector store...")
    for chunk in chunks:
        # Use the chunk's content to find similar chunks
        similar_chunks = db.similarity_search(chunk.page_content, k=1)
        if similar_chunks:
            retrieved_chunks.append(similar_chunks[0])

    # Test each retrieved chunk
    total_chunks = len(retrieved_chunks)
    found_chunks = 0
    not_found_chunks = []

    logger.info("\nTesting each chunk...")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\rTesting chunk {i}/{total_chunks}", end="", flush=True)

        # Get the page number from metadata
        page_num = chunk.metadata.get("page", 0)  # Already 0-based
        if page_num < 0 or page_num >= len(doc):
            logger.info(f"\nWarning: Invalid page number {page_num + 1} for chunk {i}")
            not_found_chunks.append((i, chunk.page_content))
            continue

        # Get the page
        page = doc[page_num]

        # Search for the chunk content in the page
        search_results = robust_search_for(page, chunk.page_content)

        if search_results:
            found_chunks += 1
        else:
            not_found_chunks.append((i, chunk.page_content))

    logger.info("\n\nResults:")
    logger.info(f"Total chunks tested: {total_chunks}")
    logger.info(f"Chunks found in source: {found_chunks}")
    logger.info(f"Chunks not found in source: {len(not_found_chunks)}")

    if not_found_chunks:
        logger.info("\nChunks that couldn't be found in source:")
        for i, content in not_found_chunks:
            print(f"\nChunk {i}:")
            print("-" * 80)
            print("Content:")
            print(content[:200] + "..." if len(content) > 200 else content)
            print("\nMetadata:")
            print(retrieved_chunks[i-1].metadata)
            print("-" * 80)

    # Cleanup
    if os.path.exists(temp_embedding_folder):
        import shutil
        shutil.rmtree(temp_embedding_folder)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test source search functionality")
    # parser.add_argument("file_path", help="Path to the PDF file to test")
    # args = parser.parse_args()

    test_source_search(
        "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/input_files/science.1189075.pdf",
        chunk_size=2048
    )
