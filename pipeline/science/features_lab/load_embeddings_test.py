from pathlib import Path
import logging
import os
import sys
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import FAISS

# Setup logging
logger = logging.getLogger(__name__)

# Import necessary functions
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.embeddings import get_embedding_models

def load_embeddings(embedding_folder_list: list[str | Path], embedding_type: str = 'default'):
    """
    Load embeddings from the specified folder.
    Adds a file_index metadata field to each document indicating which folder it came from.
    """
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models(embedding_type, para)
    
    # Load each database separately and add file_index to metadata
    all_docs = []
    for i, embedding_folder in enumerate(embedding_folder_list):
        db = FAISS.load_local(embedding_folder, embeddings, allow_dangerous_deserialization=True)
        # Get the documents and add file_index to their metadata
        docs = db.docstore._dict.values()
        for doc in docs:
            doc.metadata["file_index"] = i
            all_docs.append(doc)
    
    # Create a new database with all the documents that have updated metadata
    db_merged = FAISS.from_documents(all_docs, embeddings)
    
    # # Log the first 5 chunks for testing
    # logger.info(f"Total chunks in merged database: {len(all_docs)}")
    # for i, doc in enumerate(all_docs[:5]):
    #     logger.info(f"Chunk {i+1} - Content preview: {doc.page_content[:50]}...")
    #     logger.info(f"Chunk {i+1} - Metadata: {doc.metadata}")
    #     logger.info(f"Chunk {i+1} - From embedding folder index: {doc.metadata['file_index']} (corresponds to {embedding_folder_list[doc.metadata['file_index']]})")
    
    return db_merged

def list_all_chunks(db):
    """
    List all chunks in the database with their content and metadata.
    
    Args:
        db: FAISS database containing the documents
    """
    # logger.info("=== LISTING ALL CHUNKS IN DATABASE ===")
    all_docs = list(db.docstore._dict.values())
    # for i, doc in enumerate(all_docs):
    #     logger.info(f"Chunk {i+1}/{len(all_docs)}")
    #     logger.info(f"Content: {doc.page_content}")
    #     logger.info(f"Metadata: {doc.metadata}")
    #     logger.info("-" * 50)
    # logger.info(f"Total chunks: {len(all_docs)}")
    return all_docs

def test_load_embeddings():
    """
    Test the load_embeddings function to ensure all chunks are loaded correctly.
    """
    # Define embedding paths
    embedding_paths = [
        "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/embedded_content/lite_mode/1eecd3da808d385834c966650074b676/lite_embedding",
        "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/embedded_content/lite_mode/a22abad3dc9862c41d41f79d59318780/lite_embedding",
        "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/embedded_content/lite_mode/d12c88bcdeb77d8f40ae04506c0d75f1/lite_embedding"
    ]
    
    # Check if paths exist
    for path in embedding_paths:
        assert os.path.exists(path), f"Embedding path does not exist: {path}"
    
    # Load embeddings
    db_merged = load_embeddings(embedding_paths)
    
    # List all chunks in the database
    all_docs = list_all_chunks(db_merged)
    
    # Verify the database was created successfully
    assert db_merged is not None, "Failed to create merged database"
    
    # Check that the database contains documents
    doc_count = len(db_merged.docstore._dict)
    logger.info(f"Total documents in merged database: {doc_count}")
    assert doc_count > 0, "Merged database contains no documents"
    
    # Count documents per source folder
    file_index_counts = {}
    for doc_id, doc in db_merged.docstore._dict.items():
        file_index = doc.metadata.get("file_index")
        if file_index is not None:
            file_index_counts[file_index] = file_index_counts.get(file_index, 0) + 1
    
    # Log counts by source
    logger.info("Document count by source folder:")
    for idx, count in file_index_counts.items():
        folder_path = embedding_paths[idx]
        folder_name = os.path.basename(os.path.dirname(folder_path))
        logger.info(f"  Source {idx} ({folder_name}): {count} documents")
    
    # Verify all embedding folders contributed documents
    assert len(file_index_counts) == len(embedding_paths), f"Expected documents from {len(embedding_paths)} folders, but found {len(file_index_counts)}"
    for i in range(len(embedding_paths)):
        assert i in file_index_counts, f"No documents found from source folder index {i}"
        assert file_index_counts[i] > 0, f"Source folder index {i} has 0 documents"
    
    # Test simple similarity search to ensure the vector store works
    if doc_count > 0:
        # Get the first document's content as a query
        sample_doc = next(iter(db_merged.docstore._dict.values()))
        query_text = sample_doc.page_content[:50]  # Use first 50 chars
        
        # Perform a similarity search
        search_results = db_merged.similarity_search(query_text, k=3)
        
        # Verify search results
        assert len(search_results) > 0, "Similarity search returned no results"
        logger.info(f"Similarity search with query '{query_text}' returned {len(search_results)} results")
        logger.info(f"First result source folder: {search_results[0].metadata.get('file_index')}")
    
    # Print confirmation
    logger.info("All tests passed! Embeddings loaded successfully from all folders.")
    
    return db_merged

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    test_load_embeddings()