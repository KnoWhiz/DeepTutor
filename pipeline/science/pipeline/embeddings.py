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

    logger.info("Creating markdown embeddings ...")
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
                        
                        # Only add if we have meaningful content
                        if len(clean_text) > 10:  # Minimum content threshold
                            page_documents.append(Document(
                                page_content=clean_text,
                                metadata={
                                    "page": page_stat["page_num"] + 1,  # Add 1 to index (0-indexed to 1-indexed)
                                    "source": f"page_{page_stat['page_num'] + 1}",
                                    "page_type": "full_page",
                                    "total_pages": len(page_stats),
                                    "file_path": str(output_dir),  # Add file path like LiteRAG
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
                                "page": page_num + 1,  # Add 1 to index
                                "source": f"markdown_page_{page_num + 1}",
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
    Generate LiteRAG embeddings for the document using page-level chunks.
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
        logger.info("LiteRAG embedding already exists. We can load existing embeddings...")
    else:
        logger.info("Creating LiteRAG embeddings using page-level chunks...")
        
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
            # Create FAISS database from page documents
            db = FAISS.from_documents(page_documents, embeddings)
            # Create directory if it doesn't exist
            os.makedirs(lite_embedding_folder, exist_ok=True)
            # Save the embeddings
            db.save_local(lite_embedding_folder)
            logger.info(f"Saved {len(page_documents)} page-level chunks to {lite_embedding_folder}")
        else:
            logger.warning("No page content found to create LiteRAG embeddings")


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
    
    # Log the first 5 chunks for testing
    logger.info(f"Total chunks in merged database: {len(all_docs)}")
    for i, doc in enumerate(all_docs[:5]):
        logger.info(f"Chunk {i+1} - Content preview: {doc.page_content[:50]}...")
        logger.info(f"Chunk {i+1} - Metadata: {doc.metadata}")
        logger.info(f"Chunk {i+1} - From embedding folder index: {doc.metadata['file_index']} (corresponds to {embedding_folder_list[doc.metadata['file_index']]})")
    
    return db_merged