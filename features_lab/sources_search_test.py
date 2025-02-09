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
    
    # Create chunks directly from the PDF content to ensure better matching
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
                
                # Split into chunks of 512 characters
                while len(clean_text) > 0:
                    # Find a good break point near 512 characters
                    end_pos = min(512, len(clean_text))
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
                            metadata={"page": page_num, "source": f"page_{page_num + 1}"}
                        ))
                    clean_text = clean_text[end_pos:].strip()
        
        chunks.extend(text_blocks)
    
    print(f"Created {len(chunks)} searchable chunks from PDF")
    
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
    print("\nRetrieving chunks from vector store...")
    for chunk in chunks:
        # Use the chunk's content to find similar chunks
        similar_chunks = db.similarity_search(chunk.page_content, k=1)
        if similar_chunks:
            retrieved_chunks.append(similar_chunks[0])
    
    # Test each retrieved chunk
    total_chunks = len(retrieved_chunks)
    found_chunks = 0
    not_found_chunks = []
    
    print("\nTesting each chunk...")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\rTesting chunk {i}/{total_chunks}", end="", flush=True)
        
        # Get the page number from metadata
        page_num = chunk.metadata.get("page", 0)  # Already 0-based
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
    # parser.add_argument("pdf_path", help="Path to the PDF file to test")
    # args = parser.parse_args()
    
    test_source_search("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/input_files/RankRAG- Unifying Context Ranking with  Retrieval-Augmented Generation in LLMs.pdf")
