import os
import json
import time
from typing import Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    create_searchable_chunks,
    format_time_tracking,
    generate_file_id,
)
from pipeline.science.pipeline.images_understanding import initialize_image_files
from pipeline.science.pipeline.embeddings_graphrag import generate_GraphRAG_embedding
from pipeline.science.pipeline.session_manager import ChatMode
from pipeline.science.pipeline.get_doc_summary import generate_document_summary
from pipeline.science.pipeline.doc_processor import (
    mdDocumentProcessor,
    extract_pdf_content_to_markdown_via_api,
    extract_pdf_content_to_markdown
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


async def embeddings_agent(_mode, _document, _doc, file_path, embedding_folder, time_tracking: Dict[str, float] = {}):
    """
    Generate embeddings for the document
    If the embeddings already exist, load them
    Otherwise, extract content to markdown via API or local PDF extraction
    Then, initialize image files and try to append image context to texts with error handling
    Create the vector store to use as the index
    Save the embeddings to the specified folder
    Generate and save document summary using the texts we created
    """
    yield "Generating embeddings ...\n\n"
    file_id = generate_file_id(file_path)
    graphrag_start_time = time.time()
    logger.info(f"Current mode: {_mode}")
    if _mode == ChatMode.ADVANCED:
        logger.info("Mode: ChatMode.ADVANCED. Generating GraphRAG embeddings...")
        time_tracking = await generate_GraphRAG_embedding(embedding_folder, time_tracking)
    elif _mode == ChatMode.BASIC:
        logger.info("Mode: ChatMode.BASIC. Generating VectorRAG embeddings...")
    elif _mode == ChatMode.LITE:
        logger.info("Mode: ChatMode.LITE. Generating LiteRAG embeddings...")
        lite_embedding_start_time = time.time()
        await generate_LiteRAG_embedding(_doc, file_path, embedding_folder)
        time_tracking['lite_embedding_total'] = time.time() - lite_embedding_start_time
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

        # Memory cleanup
        db = None
        doc_processor = None
        texts = None

        yield "Embeddings generated successfully.\n\n"
        return
    else:
        raise ValueError("Invalid mode")
    time_tracking['graphrag_generate_embedding'] = time.time() - graphrag_start_time
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)
    embeddings_small = get_embedding_models('small', para)
    embeddings_lite = get_embedding_models('lite', para)
    doc_processor = mdDocumentProcessor()

    # Define the default filenames used by FAISS when saving
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")
    document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")

    markdown_embedding_folder = os.path.join(embedding_folder, "markdown")
    markdown_faiss_path = os.path.join(markdown_embedding_folder, "index.faiss")
    markdown_pkl_path = os.path.join(markdown_embedding_folder, "index.pkl")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path) and os.path.exists(document_summary_path) \
        and os.path.exists(markdown_faiss_path) and os.path.exists(markdown_pkl_path):
        logger.info("Embedding already exists. We can load existing embeddings...")
    else:
        try:
            markdown_extraction_start_time = time.time()
            # Extract content to markdown via API
            if not SKIP_MARKER_API:
                logger.info("Marker API is enabled. Using Marker API to extract content to markdown.")
                markdown_dir = os.path.join(embedding_folder, "markdown")
                md_path, saved_images, md_document = extract_pdf_content_to_markdown_via_api(file_path, markdown_dir)
                doc_processor.set_md_document(md_document)
            else:
                logger.info("Marker API is disabled. Using local PDF extraction.")
                markdown_dir = os.path.join(embedding_folder, "markdown")
                md_path, saved_images, md_document = extract_pdf_content_to_markdown(file_path, markdown_dir)
                doc_processor.set_md_document(md_document)
            time_tracking['markdown_extraction'] = time.time() - markdown_extraction_start_time
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
        except Exception as e:
            logger.exception(f"Error extracting content to markdown, using _doc to extract searchable content as save as markdown file: {e}")
            # Use _doc to extract searchable content as save as markdown file
            fake_markdown_extraction_start_time = time.time()
            doc_processor.set_md_document("")
            texts = []
            # Process each page in the PDF document
            for page_num in range(len(_doc)):
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
                    metadata={"source": f"page_{page_num + 1}", "page": page_num + 1}
                ))

            # Save to markdown_dir
            markdown_dir = os.path.join(embedding_folder, "markdown")
            os.makedirs(markdown_dir, exist_ok=True)
            file_id = generate_file_id(file_path)
            md_path = os.path.join(markdown_dir, f"{file_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(doc_processor.get_md_document())

            # Use the texts directly instead of splitting again
            logger.info(f"Number of pages processed: {len(texts)}")
            time_tracking['fake_markdown_extraction'] = time.time() - fake_markdown_extraction_start_time
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
        else:
            # Split the document into chunks when markdown extraction succeeded
            create_searchable_chunks_start_time = time.time()
            average_page_length = sum(len(doc.page_content) for doc in _document) / len(_document)
            chunk_size = int(average_page_length // 3)
            logger.info(f"Average page length: {average_page_length}")
            logger.info(f"Chunk size: {chunk_size}")
            logger.info("Creating new embeddings...")
            texts = create_searchable_chunks(_doc, chunk_size)
            logger.info(f"length of document chunks generated for get_response_source:{len(texts)}")
            time_tracking['create_searchable_chunks'] = time.time() - create_searchable_chunks_start_time
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

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

                # Create a temporary FAISS index for similarity search
                try:
                    temp_db = FAISS.from_documents(texts, embeddings)
                except Exception as e:
                    try:
                        logger.exception(f"Error creating temporary FAISS index: {e}")
                        logger.info("Continuing with small embeddings...")
                        temp_db = FAISS.from_documents(texts, embeddings_small)
                    except Exception as e:
                        logger.exception(f"Error creating temporary FAISS index with small embeddings: {e}")
                        logger.info("Continuing with lite embeddings...")
                        temp_db = FAISS.from_documents(texts, embeddings_lite)

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
                logger.info("No image context found to process")
        except Exception as e:
            logger.exception(f"Error processing image context: {e}")
            logger.info("Continuing without image context...")
        time_tracking['process_image_files'] = time.time() - process_image_files_start_time
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

        # Create the vector store to use as the index
        create_vector_store_start_time = time.time()
        db = FAISS.from_documents(texts, embeddings)
        # Save the embeddings to the specified folder
        db.save_local(embedding_folder)
        time_tracking['vectorrag_create_vector_store'] = time.time() - create_vector_store_start_time
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

        # Save the markdown embeddings to the specified folder
        create_markdown_embeddings_start_time = time.time()
        create_markdown_embeddings(doc_processor.get_md_document(), markdown_embedding_folder)
        time_tracking['vectorrag_create_markdown_embeddings'] = time.time() - create_markdown_embeddings_start_time
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

        try:
            # Generate and save document summary using the texts we created
            logger.info("Generating document summary...")
            generate_document_summary_start_time = time.time()
            # By default, use the markdown document to generate the summary
            await generate_document_summary(texts, embedding_folder, doc_processor.get_md_document())
            time_tracking['generate_document_summary'] = time.time() - generate_document_summary_start_time
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            logger.info("Document summary generated and saved successfully.")
        except Exception as e:
            logger.exception(f"Error generating document summary: {e}")
            logger.info("Continuing without document summary...")

    # Memory cleanup
    db = None
    doc_processor = None
    texts = None

    yield "Embeddings generated successfully.\n\n"
    return
