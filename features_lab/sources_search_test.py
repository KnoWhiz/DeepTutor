"""Test script to verify source search functionality by loading a PDF and searching for its content."""

import os
import io
import fitz
import argparse
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from pipeline.config import load_config
from pipeline.utils import get_embedding_models, robust_search_for
from frontend.state import process_pdf_file, extract_documents_from_file

def test_source_search(pdf_path: str) -> None:
    """
    Test source search functionality by:
    1. Loading a PDF file
    2. Processing it into documents and doc objects
    3. Splitting content into chunks
    4. Embedding the chunks
    5. For each chunk, verify it can be found in the source PDF
    
    Args:
        pdf_path: Path to the PDF file to test
    """
    print(f"Testing source search for PDF: {pdf_path}")
    
    # Load the PDF file
    with open(pdf_path, "rb") as f:
        file_content = f.read()
    
    # Process the PDF file
    filename = os.path.basename(pdf_path)
    documents, doc, file_paths = process_pdf_file(file_content, filename)
    
    print(f"Loaded PDF with {len(documents)} document chunks")
    
    # Load config and get embedding model
    config = load_config()
    embeddings = get_embedding_models("default", config["llm"])
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=config["embedding"]["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunks for testing")
    
    # Create temporary embedding folder
    temp_embedding_folder = "temp_embeddings"
    os.makedirs(temp_embedding_folder, exist_ok=True)
    
    # Create vector store
    db = FAISS.from_documents(chunks, embeddings)
    
    # Test each chunk
    total_chunks = len(chunks)
    found_chunks = 0
    not_found_chunks = []
    
    print("\nTesting each chunk...")
    for i, chunk in enumerate(chunks, 1):
        print(f"\rTesting chunk {i}/{total_chunks}", end="", flush=True)
        
        # Get the page number from metadata
        page_num = chunk.metadata.get("page", 0) - 1  # Convert to 0-based index
        if page_num < 0 or page_num >= len(doc):
            print(f"\nWarning: Invalid page number {page_num + 1} for chunk {i}")
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
    
    print("\n\nResults:")
    print(f"Total chunks tested: {total_chunks}")
    print(f"Chunks found in source: {found_chunks}")
    print(f"Chunks not found in source: {len(not_found_chunks)}")
    
    if not_found_chunks:
        print("\nChunks that couldn't be found in source:")
        for i, content in not_found_chunks:
            print(f"\nChunk {i}:")
            print("-" * 80)
            print(content[:200] + "..." if len(content) > 200 else content)
            print("-" * 80)
    
    # Cleanup
    if os.path.exists(temp_embedding_folder):
        import shutil
        shutil.rmtree(temp_embedding_folder)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test source search functionality")
    # parser.add_argument("pdf_path", help="Path to the PDF file to test")
    # args = parser.parse_args()
    
    test_source_search("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/input_files/science.1189075.pdf")
