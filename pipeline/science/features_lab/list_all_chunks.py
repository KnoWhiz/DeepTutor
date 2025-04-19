from pathlib import Path
import logging
import os
import sys
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import FAISS

# Setup logging
logging.basicConfig(level=logging.INFO)
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
    
    return db_merged, all_docs

def list_all_chunks(docs, output_file=None):
    """
    List all chunks with their content and metadata.
    
    Args:
        docs: List of documents to display
        output_file: Optional file path to save the results
    """
    logger.info(f"Listing all {len(docs)} chunks:")
    
    # Prepare output data
    chunks_data = []
    for i, doc in enumerate(docs):
        chunk_info = {
            "chunk_number": i + 1,
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        chunks_data.append(chunk_info)
        
        # Print to console
        logger.info(f"Chunk {i+1}/{len(docs)}:")
        logger.info(f"Content: {doc.page_content[:100]}...")
        logger.info(f"Metadata: {doc.metadata}")
        logger.info("-" * 50)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        logger.info(f"Saved all chunks to {output_file}")
    
    logger.info(f"Total chunks: {len(docs)}")
    return chunks_data

def main():
    # Define embedding paths
    embedding_paths = [
        "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/embedded_content/lite_mode/1eecd3da808d385834c966650074b676/lite_embedding",
        "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/embedded_content/lite_mode/a22abad3dc9862c41d41f79d59318780/lite_embedding",
        "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/embedded_content/lite_mode/d12c88bcdeb77d8f40ae04506c0d75f1/lite_embedding"
    ]
    
    # Check if paths exist
    for path in embedding_paths:
        if not os.path.exists(path):
            logger.error(f"Embedding path does not exist: {path}")
            return
    
    # Load embeddings
    _, all_docs = load_embeddings(embedding_paths)
    
    # List all chunks and save to file
    output_file = "all_chunks.json"
    list_all_chunks(all_docs, output_file)
    
    logger.info(f"Document count by source folder:")
    file_index_counts = {}
    for doc in all_docs:
        file_index = doc.metadata.get("file_index")
        if file_index is not None:
            file_index_counts[file_index] = file_index_counts.get(file_index, 0) + 1
    
    for idx, count in file_index_counts.items():
        folder_path = embedding_paths[idx]
        folder_name = os.path.basename(os.path.dirname(folder_path))
        logger.info(f"  Source {idx} ({folder_name}): {count} documents")

if __name__ == "__main__":
    main() 