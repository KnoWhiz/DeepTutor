import os
import json
import time
import fitz
from typing import Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    create_searchable_chunks,
    generate_file_id,
)
from pipeline.science.pipeline.images_understanding import initialize_image_files
from pipeline.science.pipeline.embeddings_graphrag import generate_GraphRAG_embedding
from pipeline.science.pipeline.session_manager import ChatMode, ChatSession
from pipeline.science.pipeline.get_doc_summary import generate_document_summary
from pipeline.science.pipeline.doc_processor import (
    mdDocumentProcessor,
    extract_pdf_content_to_markdown_via_api,
    extract_pdf_content_to_markdown,
    extract_pdf_content_to_markdown_via_api_streaming,
    save_file_txt_locally,
)
from pipeline.science.pipeline.embeddings import (
    get_embedding_models,
    create_markdown_embeddings,
    generate_LiteRAG_embedding,
)

import logging
logger = logging.getLogger("tutorpipeline.science.embeddings_agent")

load_dotenv()

# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


def calculate_page_length(_doc, file_path):
    """
    Calculate character count per page for accurate page attribution.
    Following the generate_LiteRAG_embedding approach but focusing on statistics.
    
    Args:
        _doc: The fitz.Document object
        file_path: Path to the PDF file
    
    Returns:
        List of dicts with page statistics
    """
    from pipeline.science.pipeline.embeddings import extract_document_from_file
    
    # Extract document using the same method as RAG
    document = extract_document_from_file(file_path)
    
    page_stats = []
    cumulative_chars = 0
    total_chars = 0
    
    for page_num, page_doc in enumerate(document):
        page_text = page_doc.page_content
        
        # Clean up the text (same as RAG)
        clean_text = page_text.strip()
        if clean_text:
            clean_text = clean_text.replace("-\n", "")
            clean_text = "".join(clean_text.split())
        
        char_count = len(clean_text)
        total_chars += char_count
        page_stats.append({
            "page_num": page_num,
            "char_count": char_count,
            "start_char": cumulative_chars,
            "end_char": cumulative_chars + char_count,
            "content_preview": clean_text[:100] + "..." if len(clean_text) > 100 else clean_text,
            "char_proportion": 0.0
        })
        cumulative_chars += char_count
    for i in range(len(page_stats)):
        if total_chars == 0:
            page_stats[i]['char_proportion'] = float(1/len(page_stats))
        else:
            page_stats[i]['char_proportion'] = page_stats[i]['char_count'] / total_chars
    return page_stats


async def embeddings_agent(
    _mode: ChatMode,
    _document: list,
    _doc: "fitz.Document",
    file_path: str,
    embedding_folder: str,
    time_tracking: Dict[str, float] = {},
    chat_session: ChatSession = None
):
    """
    Generate embeddings for the document
    If the embeddings already exist, load them
    Otherwise, extract content to markdown via API or local PDF extraction
    Then, initialize image files and try to append image context to texts with error handling
    Create the vector store to use as the index
    Save the embeddings to the specified folder
    Generate and save document summary using the texts we created
    """
    # yield "\n\n**Loading embeddings ...**"
    file_id = generate_file_id(file_path)
    # logger.info(f"Current mode: {_mode}")
    if _mode == ChatMode.ADVANCED:
        # GraphRAG is implemented in the following code
        logger.info("Mode: ChatMode.ADVANCED. Generating GraphRAG embeddings...")
        yield "\n\n**🗺️ Loading GraphRAG embeddings ...**"
    elif _mode == ChatMode.BASIC:
        # Basic mode is implemented in the following code
        logger.info("Mode: ChatMode.BASIC. Generating VectorRAG embeddings...")
        # yield "\n\n**🗂️ Loading VectorRAG embeddings ...**"
    elif _mode == ChatMode.LITE:
        logger.info("Mode: ChatMode.LITE. Generating RAG embeddings...")
        # yield f"\n\n**🔍 Loading RAG embeddings for file: {os.path.basename(file_path)} ...**"
        # yield "\n\n**🔍 Loading RAG embeddings for files ...**"
        lite_embedding_start_time = time.time()
        await generate_LiteRAG_embedding(_doc, file_path, embedding_folder)
        time_tracking['lite_embedding_total'] = time.time() - lite_embedding_start_time
        logger.info("lite_embedding_total for %s completed in %.2fs", file_id, time_tracking['lite_embedding_total'])

        # Memory cleanup
        db = None
        doc_processor = None
        texts = None

        # yield "\n\n**Embeddings generated successfully.**"
        logger.info("Embeddings generated successfully.")
        return
    else:
        raise ValueError("Invalid mode")

    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)
    embeddings_small = get_embedding_models('small', para)
    embeddings_lite = get_embedding_models('lite', para)
    doc_processor = mdDocumentProcessor()

    # Define the default filenames used by FAISS when saving
    logger.info("Initializing file paths ...")
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")
    document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")

    markdown_embedding_folder = os.path.join(embedding_folder, "markdown")
    markdown_faiss_path = os.path.join(markdown_embedding_folder, "index.faiss")
    markdown_pkl_path = os.path.join(markdown_embedding_folder, "index.pkl")

    # Calculate page statistics for BASIC mode (for accurate page attribution)
    page_stats = None
    if _mode == ChatMode.BASIC:
        logger.info("BASIC mode: Calculating page statistics for accurate page attribution...")
        yield "\n\n**📊 Calculating page statistics ...**"
        page_stats = calculate_page_length(_doc, file_path)
        logger.info(f"Calculated page statistics: {len(page_stats)} pages")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path) and os.path.exists(document_summary_path) \
        and os.path.exists(markdown_faiss_path) and os.path.exists(markdown_pkl_path):
        logger.info("Embedding already exists. We can load existing embeddings...")
        yield "\n\n**🔍 Embedding already exists. We can load existing embeddings...**"
    else:
        try:
            yield "\n\n**📝 Extracting markdown from the document ...**"
            markdown_extraction_start_time = time.time()
            # Extract content to markdown via API
            if not SKIP_MARKER_API:
                logger.info("Marker API is enabled. Using Marker API to extract content to markdown.")
                # yield "\n\n**Parsing PDF to markdown...**"
                logger.info("Parsing PDF to markdown...")
                markdown_dir = os.path.join(embedding_folder, "markdown")
                
                # Use the streaming version of extract_pdf_content_to_markdown_via_api
                # yield "\n\n**Starting PDF extraction...**"
                
                # Call the streaming version and get status updates
                md_path = None
                saved_images = None
                md_document = None
                
                try:
                    # Use the async streaming version that both yields progress and returns the final result
                    async for progress_update in extract_pdf_content_to_markdown_via_api_streaming(file_path, markdown_dir):
                        if isinstance(progress_update, tuple) and len(progress_update) == 3:
                            # This is the final return value - a tuple with (md_path, saved_images, md_document)
                            md_path, saved_images, md_document = progress_update
                            logger.info("Received final result tuple from streaming function")
                        else:
                            # # This is a progress update if there is no "**" in the string
                            # if "**" not in progress_update and "![" not in progress_update and len(progress_update) < 100:
                            #     yield f"\n\n**📑 PDF parsing progress: {progress_update}**"
                            # else:
                            #     yield f"\n\n{progress_update}"
                            #     # pass
                            yield progress_update
                    
                    # Verify we received the expected return values
                    if md_path is None or saved_images is None or md_document is None:
                        logger.error("Failed to extract PDF content - incomplete return values")
                        raise Exception("Failed to extract PDF content - incomplete return values")
                    
                    logger.info(f"PDF extraction completed. MD document length: {len(md_document) if md_document else 0}")
                    doc_processor.set_md_document(md_document)
                    yield "\n\n**📝 PDF extraction completed successfully ...**"
                except Exception as e:
                    logger.exception(f"Error during streaming PDF extraction: {str(e)}")
                    yield f"\n\n**❌ Error during PDF extraction: {str(e)}**"
                    
                    # Fall back to non-streaming version
                    yield "\n\n**❌ Falling back to standard extraction method...**"
                    md_path, saved_images, md_document = extract_pdf_content_to_markdown_via_api(file_path, markdown_dir)
                    doc_processor.set_md_document(md_document)
                    yield "\n\n**📝 PDF extraction completed with fallback method**"
            else:
                logger.info("Marker API is disabled. Using local PDF extraction.")
                yield "\n\n**🔍 Using local PDF extraction to extract content to markdown...**"
                markdown_dir = os.path.join(embedding_folder, "markdown")
                md_path, saved_images, md_document = extract_pdf_content_to_markdown(file_path, markdown_dir)
                doc_processor.set_md_document(md_document)
            time_tracking['markdown_extraction'] = time.time() - markdown_extraction_start_time
            logger.info("markdown_extraction for %s completed in %.2fs", file_id, time_tracking['markdown_extraction'])
        except Exception as e:
            logger.exception(f"Error extracting content to markdown, using _doc to extract searchable content as save as markdown file: {e}")
            yield "\n\n**❌ Error extracting content to markdown, using _doc to extract searchable content as save as markdown file...**"
            # Use _doc to extract searchable content as save as markdown file
            fake_markdown_extraction_start_time = time.time()
            yield "\n\n**🔍 Using PDF loader to extract searchable content as save as markdown file...**"
            doc_processor.set_md_document("")
            texts = []
            # Process each page in the PDF document
            for page_num in range(len(_doc)):
                yield f"\n\n**📑 Processing page {page_num + 1} of {len(_doc)}...**"
                page = _doc[page_num]
                # Get all text blocks that can be found via search
                text_blocks = []
                for block in page.get_text("blocks"):
                    text = block[4]  # The text content is at index 4
                    # Verify the text can be found via search
                    search_results = page.search_for(text.strip())
                    if search_results:
                        text_blocks.append(text)

                # Join the searchable text blocks
                page_content = "\n".join(text_blocks)
                doc_processor.append_md_document(page_content)
                texts.append(Document(
                    page_content=page_content,
                    metadata={"source": f"page_{page_num + 1}", "page": page_num}
                ))

            # Save to markdown_dir
            # yield "\n\n**Saving markdown to file...**"
            logger.info("Saving markdown to file...")
            markdown_dir = os.path.join(embedding_folder, "markdown")
            os.makedirs(markdown_dir, exist_ok=True)
            file_id = generate_file_id(file_path)
            md_path = os.path.join(markdown_dir, f"{file_id}.md")
            # yield f"\n\n**Saving markdown to file: {md_path}...**"
            logger.info(f"Saving markdown to file: {md_path}...")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(doc_processor.get_md_document())

            # Use the texts directly instead of splitting again
            logger.info(f"Number of pages processed: {len(texts)}")
            # yield f"\n\n**Number of pages processed: {len(texts)}**"
            time_tracking['fake_markdown_extraction'] = time.time() - fake_markdown_extraction_start_time
            logger.info("fake_markdown_extraction for %s completed in %.2fs", file_id, time_tracking['fake_markdown_extraction'])
        else:
            # Split the document into chunks when markdown extraction succeeded
            # yield "\n\n**📑 Splitting document into chunks ...**"
            create_searchable_chunks_start_time = time.time()
            
            # Calculate average page length, handling empty document case
            if len(_document) > 0:
                average_page_length = sum(len(doc.page_content) for doc in _document) / len(_document)
                chunk_size = int(average_page_length // 3)
            else:
                # For empty documents (like image-only PDFs), use a default chunk size
                average_page_length = 0.0
                chunk_size = 1000  # Default chunk size for image-only PDFs
            
            logger.info(f"Average page length: {average_page_length}")
            # yield f"\n\n**Average page length: {int(average_page_length)}**"
            logger.info(f"Chunk size: {chunk_size}")
            # yield f"\n\n**Chunk size: {int(chunk_size)}**"
            texts = create_searchable_chunks(_doc, chunk_size)
            # yield f"\n\n**length of document chunks generated for get_response_source: {len(texts)}**"
            logger.info(f"length of document chunks generated for get_response_source: {len(texts)}")
            
            # Handle image-only PDFs: if no chunks were created from text extraction,
            # but we have markdown content from OCR, divide it equally among pages
            if len(texts) == 0 and doc_processor.get_md_document().strip():
                logger.info("No text chunks found, but markdown content exists. Processing as image-only PDF...")
                yield "\n\n**🖼️ Processing image-only PDF: dividing OCR content by pages ...**"
                
                markdown_content = doc_processor.get_md_document().strip()
                total_pages = len(_doc)
                
                if total_pages > 0:
                    # Calculate content per page
                    content_per_page = len(markdown_content) // total_pages
                    
                    # Split content into pages
                    for page_num in range(total_pages):
                        start_pos = page_num * content_per_page
                        if page_num == total_pages - 1:
                            # Last page gets any remaining content
                            end_pos = len(markdown_content)
                        else:
                            end_pos = (page_num + 1) * content_per_page
                        
                        page_content = markdown_content[start_pos:end_pos].strip()
                        
                        if page_content:
                            texts.append(Document(
                                page_content=page_content,
                                metadata={
                                    "page": page_num,
                                    "source": f"page_{page_num + 1}",
                                    "chunk_index": 0,
                                    "is_ocr_content": True,
                                    "total_pages": total_pages,
                                    "content_start": start_pos,
                                    "content_end": end_pos
                                }
                            ))
                    
                    logger.info(f"Created {len(texts)} chunks for image-only PDF with {total_pages} pages")
                    yield f"\n\n**✅ Created {len(texts)} chunks for image-only PDF with {total_pages} pages**"
                else:
                    logger.warning("PDF has no pages, cannot process as image-only PDF")
                    yield "\n\n**⚠️ PDF has no pages, cannot process as image-only PDF**"
            
            time_tracking['create_searchable_chunks'] = time.time() - create_searchable_chunks_start_time
            logger.info("create_searchable_chunks for %s completed in %.2fs", file_id, time_tracking['create_searchable_chunks'])

        # Initialize image files and try to append image context to texts with error handling
        process_image_files_start_time = time.time()
        try:
            markdown_dir = os.path.join(embedding_folder, "markdown")
            image_context_path, _ = initialize_image_files(markdown_dir)
            with open(image_context_path, "r") as f:
                image_context = json.load(f)

            # Only process image context if there are actual images
            if image_context:
                logger.info(f"Found {len(image_context)} images with context")
                # yield f"\n\n**Found {len(image_context)} images with context**"

                # Create a temporary FAISS index for similarity search
                # Check if we have any texts to process
                if not texts:
                    logger.warning("No texts available for FAISS index creation, skipping image context processing")
                    yield "\n\n**⚠️ No texts available for FAISS index creation, skipping image context processing**"
                    temp_db = None
                else:
                    try:
                        temp_db = FAISS.from_documents(texts, embeddings)
                    except Exception as e:
                        try:
                            logger.exception(f"Error creating temporary FAISS index: {e}")
                            yield f"\n\n**❌ Error creating temporary FAISS index: {e}**"
                            logger.info("Continuing with small embeddings...")
                            yield "\n\n**🔍 Continuing with small embeddings...**"
                            temp_db = FAISS.from_documents(texts, embeddings_small)
                        except Exception as e:
                            logger.exception(f"Error creating temporary FAISS index with small embeddings: {e}")
                            yield f"\n\n**❌ Error creating temporary FAISS index with small embeddings: {e}**"
                            logger.info("Continuing with lite embeddings...")
                            yield "\n\n**🔍 Continuing with lite embeddings...**"
                            temp_db = FAISS.from_documents(texts, embeddings_lite)

                # Only process image context if we have a valid temp_db
                if temp_db is not None:
                    for image, context in image_context.items():
                        for c in context:
                            # Clean the context text for comparison
                            clean_context = c.replace(" <markdown>", "").strip()

                            # Use similarity search to find the most relevant chunk
                            similar_chunks = temp_db.similarity_search_with_score(clean_context, k=1)

                            if similar_chunks:
                                best_match_chunk, score = similar_chunks[0]
                                # Only use the page number if the similarity score is good enough
                                # (score is distance, so lower is better)
                                best_match_page = best_match_chunk.metadata.get("page", 0) if score < 1.0 else 0
                            else:
                                best_match_page = 0

                            texts.append(Document(
                                page_content=c, 
                                metadata={
                                    "source": image,
                                    "page": best_match_page
                                }
                            ))
                else:
                    logger.warning("Cannot process image context without valid FAISS index")
                    yield "\n\n**⚠️ Cannot process image context without valid FAISS index**"

            else:
                logger.info("No image context found to process")
                yield "\n\n**❌ No image context found to process**"
        except Exception as e:
            logger.exception(f"Error processing image context: {e}")
            yield "\n\n**❌ Error processing image context: {e}**"
            logger.info("Continuing without image context...")
            yield "\n\n**❌ Continuing without image context...**"
        time_tracking['process_image_files'] = time.time() - process_image_files_start_time
        logger.info("process_image_files for %s completed in %.2fs", file_id, time_tracking['process_image_files'])

        # Create the vector store to use as the index
        create_vector_store_start_time = time.time()
        # yield "\n\n**🗂️ Creating vector store ...**"
        
        # Check if we have any texts to create embeddings from
        if not texts:
            logger.error("No texts available for vector store creation")
            yield "\n\n**❌ No texts available for vector store creation**"
            
            # Create a minimal placeholder document to prevent complete failure
            logger.info("Creating minimal placeholder document to prevent complete failure")
            yield "\n\n**🔧 Creating minimal placeholder document to prevent complete failure**"
            texts = [Document(
                page_content="This document contains no extractable text content. It may be an image-only PDF or contain only non-text elements.",
                metadata={
                    "page": 0,
                    "source": "placeholder",
                    "chunk_index": 0,
                    "is_placeholder": True,
                    "warning": "No text content could be extracted from this document"
                }
            )]
        
        try:
            db = FAISS.from_documents(texts, embeddings)
            # Save the embeddings to the specified folder
            # yield "\n\n**Saving vector store to file...**"
            logger.info("Saving vector store to file...")
            db.save_local(embedding_folder)
        except Exception as e:
            logger.exception(f"Error creating vector store with default embeddings: {e}")
            yield f"\n\n**❌ Error creating vector store with default embeddings: {e}**"
            try:
                logger.info("Trying with small embeddings...")
                yield "\n\n**🔍 Trying with small embeddings...**"
                db = FAISS.from_documents(texts, embeddings_small)
                db.save_local(embedding_folder)
            except Exception as e2:
                logger.exception(f"Error creating vector store with small embeddings: {e2}")
                yield f"\n\n**❌ Error creating vector store with small embeddings: {e2}**"
                try:
                    logger.info("Trying with lite embeddings...")
                    yield "\n\n**🔍 Trying with lite embeddings...**"
                    db = FAISS.from_documents(texts, embeddings_lite)
                    db.save_local(embedding_folder)
                except Exception as e3:
                    logger.exception(f"Error creating vector store with lite embeddings: {e3}")
                    yield f"\n\n**❌ Error creating vector store with lite embeddings: {e3}**"
                    raise ValueError(f"Failed to create vector store with all embedding models: {e3}")
        
        time_tracking['vectorrag_create_vector_store'] = time.time() - create_vector_store_start_time
        logger.info("vectorrag_create_vector_store for %s completed in %.2fs", file_id, time_tracking['vectorrag_create_vector_store'])

        # Save the markdown embeddings to the specified folder
        create_markdown_embeddings_start_time = time.time()
        # yield "\n\n**📝 Creating markdown embeddings ...**"
        create_markdown_embeddings(
            doc_processor.get_md_document(), 
            markdown_embedding_folder, 
            chunk_size=config['embedding']['chunk_size'], 
            chunk_overlap=config['embedding']['chunk_overlap'],
            page_stats=page_stats  # Pass page statistics for accurate page attribution
        )
        time_tracking['vectorrag_create_markdown_embeddings'] = time.time() - create_markdown_embeddings_start_time
        logger.info("vectorrag_create_markdown_embeddings for %s completed in %.2fs", file_id, time_tracking['vectorrag_create_markdown_embeddings'])

        try:
            # Generate and save document summary using the texts we created
            logger.info("Generating document summary ...")
            # yield "\n\n**📚 Loading document summary ...**"
            generate_document_summary_start_time = time.time()
            # By default, use the markdown document to generate the summary
            await generate_document_summary(texts, embedding_folder, doc_processor.get_md_document())
            time_tracking['generate_document_summary'] = time.time() - generate_document_summary_start_time
            logger.info("generate_document_summary for %s completed in %.2fs", file_id, time_tracking['generate_document_summary'])
            logger.info("Document summary generated and saved successfully ...")
            # yield "\n\n**📚 Document summary generated and saved successfully ...**"
        except Exception as e:
            logger.exception(f"Error generating document summary: {e}")
            yield f"\n\n**❌ Error generating document summary: {e}**"
            logger.info("Continuing without document summary...")
            # yield "\n\n**❌ Continuing without document summary...**"

    graphrag_start_time = time.time()
    if _mode == ChatMode.ADVANCED:
        # GraphRAG_embedding_generator = await generate_GraphRAG_embedding(embedding_folder, time_tracking)
        # if GraphRAG_embedding_generator:
        #     for chunk in GraphRAG_embedding_generator:
        #         yield chunk
        yield "\n\n**🗺️ Building knowledge graph based on markdown ...**"
        save_file_txt_locally(file_path, filename=file_id[:8], embedding_folder=embedding_folder, chat_session=chat_session)
        async for chunk in generate_GraphRAG_embedding(embedding_folder, time_tracking):
            yield chunk
    time_tracking['graphrag_generate_embedding'] = time.time() - graphrag_start_time
    logger.info("graphrag_generate_embedding for %s completed in %.2fs", file_id, time_tracking['graphrag_generate_embedding'])

    # Memory cleanup
    db = None
    doc_processor = None
    texts = None

    # yield "\n\n**Embeddings generated successfully.**"
    logger.info("Embeddings generated successfully.")
    return
