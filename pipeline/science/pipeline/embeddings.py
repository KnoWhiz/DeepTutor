#!/usr/bin/env python3
"""
Embeddings module for the DeepTutor pipeline.

This module provides functionality for creating, loading, and managing embeddings
for various document types and chat modes. It includes robust error handling and
automatic regeneration of missing embedding index files.

Key Features:
- Document embedding creation using various embedding models
- FAISS vector store management for efficient similarity search
- Automatic validation of embedding index files before loading
- Automatic regeneration of missing embeddings when possible
- Support for different embedding types (lite, default, etc.)

New Functionality (Added for robustness):
- validate_embedding_index_files(): Checks for missing FAISS index files
- load_embeddings_with_regeneration(): Loads embeddings with automatic regeneration
- regenerate_missing_lite_embeddings(): Regenerates missing lite embeddings from source PDFs

Error Handling:
The module now gracefully handles missing embedding index files by:
1. Validating all embedding folders before attempting to load
2. Providing detailed logging about missing files
3. Attempting automatic regeneration when possible
4. Falling back to partial loading with valid folders only
5. Raising informative errors when no valid embeddings can be loaded

This addresses the RuntimeError that occurred when FAISS index files were missing:
"Error: 'f' failed: could not open ... for reading: No such file or directory"
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler
# from pipeline.science.pipeline.doc_processor import extract_document_from_file
# from pipeline.science.pipeline.utils import create_searchable_chunks

import logging
logger = logging.getLogger("tutorpipeline.science.embeddings")

load_dotenv()

# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")

# Custom function to extract document objects from uploaded file
def extract_document_from_file(file_path):
    # Load the document
    loader = PyMuPDFLoader(file_path)
    document = loader.load()
    return document

# Define create_searchable_chunks here to avoid circular import
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








def get_embedding_models(embedding_type, para):
    para = para
    api = ApiHandler(para)
    embedding_model_default = api.embedding_models['default']['instance']
    embedding_model_lite = api.embedding_models['default']['instance']
    embedding_model_small = api.embedding_models['default']['instance']
    # embedding_model_lite = api.embedding_models['lite']['instance']
    # embedding_model_small = api.embedding_models['small']['instance']
    if embedding_type == 'default':
        return embedding_model_default
    elif embedding_type == 'lite':
        return embedding_model_lite
    elif embedding_type == 'small':
        return embedding_model_small
    else:
        return embedding_model_default


# Create markdown embeddings
def create_markdown_embeddings(md_document: str, output_dir: str | Path, chunk_size: int = 3000, chunk_overlap: int = 300, page_stats=None):
    """
    Create markdown embeddings from a markdown document and save them to the specified directory.

    Args:
        md_document: Markdown document
        output_dir: Directory where embeddings will be saved
        chunk_size: Size of each chunk for regular embeddings
        chunk_overlap: Overlap between chunks for regular embeddings
        page_stats: List of page statistics for accurate page attribution (BASIC mode only)

    Returns:
        None
    """
    # Load the markdown file
    # Create and save markdown embeddings
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)
    
    page_based_embedding_folder = os.path.join(output_dir, 'page_based_index')
    page_based_faiss_path = os.path.join(page_based_embedding_folder, "index.faiss")
    page_based_pkl_path = os.path.join(page_based_embedding_folder, "index.pkl")
    logger.info(f"MMARK: create_markdown_embeddings")

    logger.info("Creating markdown embeddings ...")

    logger.info(f"page_stats: {page_stats}")
    if md_document:
        # Create markdown directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Split markdown content into chunks with page attribution
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(md_document)
        
        # Create documents with accurate page attribution
        markdown_texts = []
        
        if page_stats:
            # NEW APPROACH: Use page statistics for both main and page-based embeddings
            logger.info("Using page statistics for unified embedding approach...")
            
            # Split entire markdown into chunks
            chunks = text_splitter.split_text(md_document)
            
            for chunk_index, chunk in enumerate(chunks):
                clean_chunk = chunk.replace('<|endoftext|>', '')
                
                if clean_chunk.strip():
                    # Calculate chunk position using page statistics proportion
                    chunk_size_actual = len(clean_chunk)
                    chunk_position = chunk_index * chunk_size  # Estimate position
                    
                    # Find which page this chunk belongs to using proportion
                    total_md_length = len(md_document)
                    char_proportion = chunk_position / total_md_length if total_md_length > 0 else 0
                    
                    # Find the page that contains this proportion
                    page_num = 0  # Default to first page
                    for i, page_stat in enumerate(page_stats):
                        page_start_proportion = page_stat["start_char"] / total_md_length if total_md_length > 0 else 0
                        page_end_proportion = page_stat["end_char"] / total_md_length if total_md_length > 0 else 0
                        
                        if page_start_proportion <= char_proportion <= page_end_proportion:
                            page_num = i  # 0-indexed page number
                            break
                    
                    # Add 10% overflow for chunk boundaries
                    overflow_size = int(chunk_size_actual * 0.1)
                    start_pos = max(0, chunk_position - overflow_size)
                    end_pos = min(len(md_document), chunk_position + chunk_size_actual + overflow_size)
                    
                    markdown_texts.append(Document(
                        page_content=clean_chunk,
                        metadata={
                            "source": "markdown",
                            "page": page_num,  # 0-indexed page number
                            "chunk_index": chunk_index,
                            "char_position": chunk_position,
                            "start_char": start_pos,
                            "end_char": end_pos,
                            "attribution_method": "page_stat_proportion"
                        }
                    ))
                else:
                    logger.warning(f"Skipping chunk {chunk_index} due to empty content")
        else:
            # Fallback to old method when no page statistics available
            logger.info("Using fallback chunking method (no page statistics)...")
            for chunk_index, chunk in enumerate(chunks):
                clean_chunk = chunk.replace('<|endoftext|>', '')
                
                # Fallback estimation method
                chunk_position = md_document.find(clean_chunk)
                estimated_page = max(1, (chunk_position // (chunk_size * 2)) + 1) if chunk_position != -1 else 1
                page_num = estimated_page - 1  # Convert to 0-indexed
                
                markdown_texts.append(Document(
                    page_content=clean_chunk,
                    metadata={
                        "source": "markdown",
                        "page": page_num,  # 0-indexed for consistency
                        "chunk_index": chunk_index,
                        "char_position": chunk_position if chunk_position != -1 else (chunk_index * chunk_size)
                    }
                ))


        # COMMENTED OUT: Original markdown embeddings generation
        # for text in markdown_texts:
        #     logger.info(f"markdown text after text splitter: {text}")
        # 
        # # Create and save markdown embeddings
        # db_markdown = FAISS.from_documents(markdown_texts, embeddings)
        # db_markdown.save_local(output_dir)
        # logger.info(f"Saved {len(markdown_texts)} markdown chunks to {output_dir}")
        
        # Create unified page-based embeddings for markdown (will be saved to both locations)
        if os.path.exists(page_based_faiss_path) and os.path.exists(page_based_pkl_path):
            logger.info("Markdown page-based embedding already exists. We can load existing embeddings...")
        else:
            logger.info("Creating unified page-based embeddings...")
            page_documents = []
            
            if page_stats:
                # Use char_proportion with cum_prop for refined page-based chunking
                logger.info("Using char_proportion with cum_prop for page-based embeddings...")
                cum_prop = 0.0  # Track cumulative proportion
                
                for page_stat in page_stats:
                    # Use char_proportion to estimate actual start/end positions
                    char_proportion = page_stat['char_proportion']
                    total_md_length = len(md_document)
                    
                    # Estimate chunk boundaries using proportion
                    estimated_start = int(cum_prop * total_md_length)
                    estimated_end = int((cum_prop + char_proportion) * total_md_length)
                    
                    # Apply 10% overflow (front and back)
                    chunk_size_actual = estimated_end - estimated_start
                    overflow_size = int(chunk_size_actual * 0.1)
                    start_pos = max(0, estimated_start - overflow_size)
                    end_pos = min(len(md_document), estimated_end + overflow_size)
                    logger.info(f"=== EMBEDDINGS DEBUG: start_pos = {start_pos}, end_pos = {end_pos} ===")
                    logger.info(f"=== EMBEDDINGS DEBUG: char_proportion = {char_proportion} ===")
                    logger.info(f"=== EMBEDDINGS DEBUG: cum_prop = {cum_prop} ===")
                    
                    # Graceful bounds checking
                    if start_pos >= len(md_document):
                        logger.warning(f"Page {page_stat['page_num']} start position {start_pos} exceeds markdown length {len(md_document)}")
                        cum_prop += char_proportion  # Update cum_prop even if skipping
                        continue
                    
                    page_text = md_document[start_pos:end_pos]
                    
                    # Clean up the text
                    clean_text = page_text.strip()
                    if clean_text:
                        clean_text = clean_text.replace('<|endoftext|>', '')
                        clean_text = " ".join(clean_text.split())
                        logger.info(f"=== EMBEDDINGS DEBUG: clean_text = {clean_text} and page_stat['page_num'] = {page_stat['page_num']} ===")
                        # Only add if we have meaningful content
                        if len(clean_text) > 10:  # Minimum content threshold
                            page_documents.append(Document(
                                page_content=clean_text,
                                metadata={
                                    "page": page_stat["page_num"],  # Add 1 to index (0-indexed to 1-indexed)
                                    "source": f"page_{page_stat['page_num']}",
                                    "page_type": "full_page",
                                    "total_pages": len(page_stats),
                                    "file_path": str(output_dir),  # Add file path like RAG
                                    "start_char": start_pos,
                                    "end_char": end_pos,
                                    "total_chars": len(md_document),
                                    "original_char_count": page_stat["char_count"],
                                    "char_proportion": char_proportion,
                                    "cum_prop": cum_prop
                                }
                            ))
                        else:
                            logger.warning(f"Page {page_stat['page_num']} has insufficient content ({len(clean_text)} chars), skipping")
                    else:
                        logger.warning(f"Page {page_stat['page_num']} has no content after cleaning")
                    
                    # Update cumulative proportion for next iteration
                    cum_prop += char_proportion
            else:
                # Fallback to old method: Use 10 times the chunk size as "page size"
                logger.info("Using fallback method for page-based embeddings...")
                page_size = chunk_size * 10
                
                for page_num, start_pos in enumerate(range(0, len(md_document), page_size)):
                    end_pos = min(start_pos + page_size, len(md_document))
                    page_text = md_document[start_pos:end_pos]
                    
                    # Clean up the text
                    clean_text = page_text.strip()
                    if clean_text:
                        clean_text = clean_text.replace('<|endoftext|>', '')
                        clean_text = " ".join(clean_text.split())
                        
                        page_documents.append(Document(
                            page_content=clean_text,
                            metadata={
                                "page": page_num,  
                                "source": f"markdown_page_{page_num}",
                                "page_type": "markdown_page",
                                "page_size": page_size,
                                "start_char": start_pos,
                                "end_char": end_pos,
                                "total_chars": len(md_document)
                            }
                        ))
            
            if page_documents:
                # Create FAISS database from page documents
                logger.info(f"=== EMBEDDINGS DEBUG: page_documents count = {len(page_documents)} ===")
                
                # Debug: Show detailed content of page_documents
                for i, doc in enumerate(page_documents):
                    logger.info(f"=== EMBEDDINGS DEBUG: Document {i+1}/{len(page_documents)} ===")
                    logger.info(f"=== EMBEDDINGS DEBUG: Content preview = {doc.page_content[:200]}... ===")
                    logger.info(f"=== EMBEDDINGS DEBUG: Metadata = {doc.metadata} ===")
                
                # Show sample of first 2 documents for quick overview
                if len(page_documents) > 0:
                    logger.info(f"=== EMBEDDINGS DEBUG: First document content = {page_documents[0].page_content[:300]}... ===")
                    if len(page_documents) > 1:
                        logger.info(f"=== EMBEDDINGS DEBUG: Second document content = {page_documents[1].page_content[:300]}... ===")
                
                page_db = FAISS.from_documents(page_documents, embeddings)
                # Create directory if it doesn't exist
                os.makedirs(page_based_embedding_folder, exist_ok=True)
                # Save the page-based embeddings to BOTH locations (unified approach)
                page_db.save_local(page_based_embedding_folder)
                page_db.save_local(output_dir)  # Save to main output_dir as well
                logger.info(f"=== EMBEDDINGS DEBUG: Saved {len(page_documents)} unified page-based documents ===")
                logger.info(f"=== EMBEDDINGS DEBUG: Saved to {page_based_embedding_folder} and {output_dir} ===")
            else:
                logger.warning("=== EMBEDDINGS DEBUG: No page_documents found - page_documents is empty ===")
    else:
        logger.info("No markdown content available to create markdown embeddings")


async def generate_LiteRAG_embedding(_doc, file_path, embedding_folder):
    """
    Generate RAG embeddings for the document using page-level chunks.
    Each page becomes one chunk in the embedding.
    """
    config = load_config()
    para = config['llm']
    lite_embedding_folder = os.path.join(embedding_folder, 'lite_embedding')
    
    # Check if all necessary files exist to load the embeddings
    faiss_path = os.path.join(lite_embedding_folder, "index.faiss")
    pkl_path = os.path.join(lite_embedding_folder, "index.pkl")
    
    embeddings = get_embedding_models('lite', para)
    
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        logger.info("RAG embedding already exists. We can load existing embeddings...")
    else:
        logger.info("Creating RAG embeddings using page-level chunks...")
        
        # Extract document using the doc_processor function
        document = extract_document_from_file(file_path)
        
        # Create page-based documents (one document per page)
        page_documents = []
        for page_num, page_doc in enumerate(document):
            page_text = page_doc.page_content
            
            # Clean up the text
            clean_text = page_text.strip()
            if clean_text:
                # Remove hyphenation at line breaks
                clean_text = clean_text.replace("-\n", "")
                # Normalize spaces
                clean_text = " ".join(clean_text.split())
                
                page_documents.append(Document(
                    page_content=clean_text,
                    metadata={
                        "page": page_num,
                        "source": f"page_{page_num + 1}",
                        "page_type": "full_page",
                        "total_pages": len(document),
                        "file_path": file_path
                    }
                ))
        
        if page_documents:
            # Create FAISS database from page documents – retry with backup
            try:
                db = FAISS.from_documents(page_documents, embeddings)
            except Exception as e:
                logger.warning(
                    "generate_LiteRAG_embedding: primary embedding attempt failed – %s."
                    " Retrying with backup Azure endpoint.",
                    repr(e),
                )

                from langchain_openai import AzureOpenAIEmbeddings

                # Determine a reasonable deployment/model to mirror the requested one
                # If the provided embeddings instance has a `model` attribute, reuse it.
                model_name = getattr(embeddings, "model", "text-embedding-3-large")
                deployment_name = getattr(embeddings, "deployment", model_name)

                fallback_embeddings = AzureOpenAIEmbeddings(
                    deployment=deployment_name,
                    model=model_name,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
                    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
                    openai_api_type="azure",
                    chunk_size=2000,
                )

                try:
                    db = FAISS.from_documents(page_documents, fallback_embeddings)
                except Exception as e2:
                    logger.error(
                        "generate_LiteRAG_embedding: fallback embedding attempt failed – %s.",
                        repr(e2),
                    )
                    raise  # Re‑raise so upstream can handle
            # Create directory if it doesn't exist
            os.makedirs(lite_embedding_folder, exist_ok=True)
            # Save the embeddings
            db.save_local(lite_embedding_folder)
            logger.info(f"Saved {len(page_documents)} page-level chunks to {lite_embedding_folder}")
        else:
            logger.warning("No page content found to create RAG embeddings")


def validate_embedding_index_files(embedding_folder_list: list[str | Path], embedding_type: str = 'lite') -> tuple[list[str | Path], list[str | Path]]:
    """
    Validate that all required index files exist for each embedding folder.
    
    Args:
        embedding_folder_list: List of embedding folder paths
        embedding_type: Type of embedding ('lite', 'default', etc.)
    
    Returns:
        tuple: (valid_folders, invalid_folders) - Lists of folder paths
    """
    valid_folders = []
    invalid_folders = []
    
    for embedding_folder in embedding_folder_list:
        embedding_folder_path = Path(embedding_folder)
        
        if embedding_type == 'lite':
            # For lite embeddings, check for index.faiss and index.pkl files
            faiss_path = embedding_folder_path / "index.faiss"
            pkl_path = embedding_folder_path / "index.pkl"
            
            if faiss_path.exists() and pkl_path.exists():
                valid_folders.append(embedding_folder)
                logger.debug(f"Valid embedding folder: {embedding_folder}")
            else:
                invalid_folders.append(embedding_folder)
                logger.warning(f"Invalid embedding folder (missing index files): {embedding_folder}")
                if not faiss_path.exists():
                    logger.warning(f"Missing file: {faiss_path}")
                if not pkl_path.exists():
                    logger.warning(f"Missing file: {pkl_path}")
        else:
            # For other embedding types, check if the folder exists and has content
            if embedding_folder_path.exists() and any(embedding_folder_path.iterdir()):
                valid_folders.append(embedding_folder)
                logger.debug(f"Valid embedding folder: {embedding_folder}")
            else:
                invalid_folders.append(embedding_folder)
                logger.warning(f"Invalid embedding folder (empty or missing): {embedding_folder}")
    
    return valid_folders, invalid_folders


async def load_embeddings(embedding_folder_list: list[str | Path], embedding_type: str = 'default', file_path_list: list[str | Path] | None = None):
    """
    Load embeddings from the specified folder.
    Adds a file_index metadata field to each document indicating which folder it came from.
    
    Args:
        embedding_folder_list: List of paths to embedding folders
        embedding_type: Type of embedding ('lite', 'default', etc.)
    
    Returns:
        FAISS database with merged embeddings
        
    Raises:
        RuntimeError: If no valid embedding folders are found after regeneration attempts
    """
    
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models(embedding_type, para)
    
    # Validate all embedding folders before attempting to load
    # logger.info(f"Validating {len(embedding_folder_list)} embedding folders...")
    valid_folders, invalid_folders = validate_embedding_index_files(embedding_folder_list, embedding_type)
    
    # If there are invalid folders, attempt to regenerate them immediately
    if invalid_folders:
        logger.info(f"Found {len(invalid_folders)} invalid embedding folders. Attempting regeneration...")
        for folder in invalid_folders:
            logger.info(f"  - Invalid folder: {folder}")
        
        if embedding_type == 'lite':
            # Attempt to regenerate missing lite embeddings
            try:
                logger.info("Starting regeneration process for missing lite embeddings...")
                regenerated_folders = await regenerate_missing_lite_embeddings(invalid_folders, embedding_folder_list, file_path_list)
                
                if regenerated_folders:
                    logger.info(f"Successfully regenerated {len(regenerated_folders)} embedding folders")
                    # Re-validate after regeneration to update our valid/invalid lists
                    valid_folders, remaining_invalid = validate_embedding_index_files(embedding_folder_list, embedding_type)
                    if remaining_invalid:
                        logger.warning(f"Still have {len(remaining_invalid)} invalid folders after regeneration:")
                        for folder in remaining_invalid:
                            logger.warning(f"  - Still invalid: {folder}")
                else:
                    logger.error("No embedding folders were successfully regenerated")
                    
            except Exception as e:
                logger.error(f"Error during regeneration process: {str(e)}")
                # Continue with whatever valid folders we have
        else:
            logger.warning(f"Regeneration not supported for embedding type '{embedding_type}'. Proceeding with valid folders only.")
    
    # Final validation check
    if not valid_folders:
        error_msg = (
            f"No valid embedding folders found after regeneration attempts. "
            f"Total folders: {len(embedding_folder_list)}, "
            f"All folders are missing required index files."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"Proceeding to load embeddings from {len(valid_folders)} valid folders")
    
    # Load each valid database separately and add file_index to metadata
    all_docs = []
    original_folder_mapping = {valid_folder: embedding_folder_list.index(valid_folder) 
                              for valid_folder in valid_folders if valid_folder in embedding_folder_list}
    
    successfully_loaded = 0
    for i, embedding_folder in enumerate(valid_folders):
        try:
            logger.info(f"Loading embeddings from: {embedding_folder}")
            db = FAISS.load_local(embedding_folder, embeddings, allow_dangerous_deserialization=True)
            
            # Get the documents and add file_index to their metadata
            docs = db.docstore._dict.values()
            original_index = original_folder_mapping.get(embedding_folder, i)
            
            for doc in docs:
                doc.metadata["file_index"] = original_index
                doc.metadata["embedding_folder"] = str(embedding_folder)
                all_docs.append(doc)
                
            logger.info(f"Successfully loaded {len(docs)} pages of documents from {embedding_folder}")
            successfully_loaded += 1
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embedding_folder}: {str(e)}")
            # This shouldn't happen since we validated, but let's be safe
            continue
    
    if not all_docs:
        error_msg = (
            f"No documents could be loaded from any embedding folders. "
            f"Total folders attempted: {len(embedding_folder_list)}, "
            f"Successfully loaded: {successfully_loaded}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # # Create a new database with all the documents that have updated metadata
    # logger.info(f"Creating merged database from {len(all_docs)} documents...")

    try:
        db_merged = FAISS.from_documents(all_docs, embeddings)
    except Exception as e:
        logger.warning(
            "load_embeddings: primary FAISS build failed – %s. Retrying with backup Azure embeddings.",
            repr(e),
        )

        from langchain_openai import AzureOpenAIEmbeddings

        model_name = getattr(embeddings, "model", "text-embedding-3-large")
        deployment_name = getattr(embeddings, "deployment", model_name)

        fallback_embeddings = AzureOpenAIEmbeddings(
            deployment=deployment_name,
            model=model_name,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
            openai_api_type="azure",
            chunk_size=2000,
        )

        try:
            db_merged = FAISS.from_documents(all_docs, fallback_embeddings)
        except Exception as e2:
            logger.error(
                "load_embeddings: fallback FAISS build failed – %s.",
                repr(e2),
            )
            raise
    
    # # Log summary information
    # logger.info(f"Successfully created merged database:")
    # logger.info(f"  - Total chunks: {len(all_docs)}")
    # logger.info(f"  - Successfully loaded folders: {successfully_loaded}")
    # logger.info(f"  - Total folders requested: {len(embedding_folder_list)}")
    
    # # Log the first 5 chunks for testing
    # for i, doc in enumerate(all_docs[:5]):
    #     logger.info(f"Chunk {i+1} - Content preview: {doc.page_content[:50]}...")
    #     logger.info(f"Chunk {i+1} - Metadata: {doc.metadata}")
    #     if doc.metadata['file_index'] < len(embedding_folder_list):
    #         logger.info(f"Chunk {i+1} - From embedding folder index: {doc.metadata['file_index']} (corresponds to {embedding_folder_list[doc.metadata['file_index']]})")
    
    return db_merged


async def regenerate_missing_lite_embeddings(invalid_folders: list[str | Path], embedding_folder_list: list[str | Path], file_path_list: list[str | Path] | None) -> list[str | Path]:
    """
    Regenerate missing lite embeddings for invalid folders.
    
    This function will:
    1. Clean up any existing incomplete files in invalid folders
    2. Find the source PDF files for each invalid folder
    3. Regenerate the embeddings and overwrite the folder contents
    4. Verify that the regeneration was successful
    
    Args:
        invalid_folders: List of embedding folder paths that are missing index files
        embedding_folder_list: Original list of all embedding folder paths
    
    Returns:
        List of successfully regenerated embedding folder paths
    """
    import shutil
    
    regenerated_folders = []
    
    for embedding_folder in invalid_folders:
        try:
            logger.info(f"Attempting to regenerate embeddings for: {embedding_folder}")
            
            # Extract the embedding folder and its parent dir
            embedding_folder_path = Path(embedding_folder)
            parent_folder = embedding_folder_path.parent
            
            # Clean up any existing incomplete files in the embedding folder
            if embedding_folder_path.exists():
                logger.info(f"Cleaning up existing incomplete folder: {embedding_folder}")
                try:
                    shutil.rmtree(embedding_folder_path)
                    logger.info(f"Successfully removed incomplete folder: {embedding_folder}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up folder {embedding_folder}: {cleanup_error}")
                    # Continue anyway, the generation might still work
            
            # Determine the source PDF path from provided file_path_list using index mapping
            file_path = None
            if file_path_list is not None:
                try:
                    # Map invalid embedding folder back to its index within the original list
                    original_index = embedding_folder_list.index(embedding_folder)
                    candidate_path = Path(str(file_path_list[original_index]))
                    if candidate_path.suffix.lower() == ".pdf" and candidate_path.exists():
                        file_path = str(candidate_path)
                        logger.info(f"Resolved source PDF via file_path_list: {file_path}")
                    else:
                        logger.warning(f"Mapped file path is not an existing PDF: {candidate_path}")
                except ValueError:
                    logger.warning(f"Embedding folder not found in original list for index mapping: {embedding_folder}")
                except Exception as map_err:
                    logger.warning(f"Error mapping file path for {embedding_folder}: {map_err}")
            
            # Fallback: attempt to locate PDF in the parent folder if mapping failed
            if file_path is None:
                pdf_files = list(parent_folder.glob("*.pdf"))
                if not pdf_files:
                    logger.error(f"No PDF files found for regeneration. Parent folder checked: {parent_folder}")
                    continue
                file_path = str(pdf_files[0])
                logger.info(f"Fallback to parent folder PDF for regeneration: {file_path}")
            
            # Open the PDF file using fitz (PyMuPDF)
            import fitz
            _doc = None
            try:
                _doc = fitz.open(file_path)
                
                # Generate lite embeddings using the existing function
                # This will create the lite_embedding folder and save the index files
                logger.info(f"Generating lite embeddings for: {file_path}")
                await generate_LiteRAG_embedding(_doc, file_path, str(parent_folder))
                
            finally:
                # Ensure the document is closed even if an error occurs
                if _doc is not None:
                    _doc.close()
            
            # Verify that the embeddings were created successfully
            faiss_path = embedding_folder_path / "index.faiss"
            pkl_path = embedding_folder_path / "index.pkl"
            
            if faiss_path.exists() and pkl_path.exists():
                regenerated_folders.append(str(embedding_folder))
                logger.info(f"✅ Successfully regenerated embeddings for: {embedding_folder}")
                logger.info(f"   - Created: {faiss_path}")
                logger.info(f"   - Created: {pkl_path}")
            else:
                logger.error(f"❌ Failed to regenerate embeddings for: {embedding_folder}")
                logger.error(f"   - Missing: {faiss_path} (exists: {faiss_path.exists()})")
                logger.error(f"   - Missing: {pkl_path} (exists: {pkl_path.exists()})")
                
        except Exception as e:
            logger.error(f"❌ Error regenerating embeddings for {embedding_folder}: {str(e)}")
            logger.error(f"   - Exception type: {type(e).__name__}")
            # Continue with other folders
            continue
    
    if regenerated_folders:
        logger.info(f"🎉 Successfully regenerated {len(regenerated_folders)} out of {len(invalid_folders)} embedding folders")
        for folder in regenerated_folders:
            logger.info(f"   ✅ {folder}")
    else:
        logger.warning(f"⚠️  No embedding folders were successfully regenerated out of {len(invalid_folders)} attempts")
        for folder in invalid_folders:
            logger.warning(f"   ❌ {folder}")
    
    return regenerated_folders


async def load_embeddings_with_regeneration(embedding_folder_list: list[str | Path], embedding_type: str = 'default', allow_regeneration: bool = True, file_path_list: list[str | Path] | None = None):
    """
    Load embeddings from the specified folders with automatic regeneration of missing files.
    
    This function is now a wrapper around load_embeddings() since regeneration is handled
    automatically in the main function.
    
    Args:
        embedding_folder_list: List of paths to embedding folders
        embedding_type: Type of embedding ('lite', 'default', etc.)
        allow_regeneration: Whether to attempt regeneration (for backward compatibility)
    
    Returns:
        FAISS database with merged embeddings
        
    Raises:
        RuntimeError: If no valid embedding folders are found after regeneration attempts
    """
    if not allow_regeneration:
        logger.warning("Regeneration is disabled, but load_embeddings() will still validate and attempt regeneration for 'lite' type")
    
    # The main load_embeddings function now handles regeneration automatically
    return await load_embeddings(embedding_folder_list, embedding_type, file_path_list)
