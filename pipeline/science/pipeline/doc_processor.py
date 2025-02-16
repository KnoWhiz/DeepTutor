import os
import json
import fitz
import asyncio
import pandas as pd
import re

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    count_tokens,
    truncate_chat_history,
    truncate_document,
    get_llm,
    get_embedding_models,
    extract_images_from_pdf,
    extract_pdf_content_to_markdown,
    extract_pdf_content_to_markdown_via_api,
    create_searchable_chunks,
)
from pipeline.science.pipeline.images_understanding import initialize_image_files
from pipeline.science.pipeline.graphrag_doc_processor import generate_GraphRAG_embedding
from pipeline.science.pipeline.session_manager import ChatMode

import logging
logger = logging.getLogger("tutorpipeline.science.doc_processor")

load_dotenv()
# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
print(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


class DocumentProcessor:
    """
    Class to handle document processing and maintain document state without ST dependency.
    """
    def __init__(self):
        self.md_document = ""

    def set_md_document(self, content: str):
        """Set the markdown document content."""
        self.md_document = content

    def append_md_document(self, content: str):
        """Append content to the markdown document."""
        self.md_document += content.strip() + "\n"

    def get_md_document(self) -> str:
        """Get the current markdown document content."""
        return self.md_document


def generate_embedding(_mode, _document, _doc, pdf_path, embedding_folder):
    """
    Generate embeddings for the document
    If the embeddings already exist, load them
    Otherwise, extract content to markdown via API or local PDF extraction
    Then, initialize image files and try to append image context to texts with error handling
    Create the vector store to use as the index
    Save the embeddings to the specified folder
    Generate and save document summary using the texts we created
    """
    print("Current mode: ", _mode)
    if _mode == ChatMode.ADVANCED:
        print("Mode: Advanced. Generating GraphRAG embeddings...")
        asyncio.run(generate_GraphRAG_embedding(embedding_folder))
    elif _mode == ChatMode.BASIC:
        print("Mode: Basic. Generating VectorRAG embeddings...")
    else:
        raise ValueError("Invalid mode")

    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)
    doc_processor = DocumentProcessor()

    # Define the default filenames used by FAISS when saving
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")
    document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path) and os.path.exists(document_summary_path):
        # Load existing embeddings
        # print("Loading existing embeddings...")
        logger.info("Loading existing embeddings...")
        db = FAISS.load_local(
            embedding_folder, embeddings, allow_dangerous_deserialization=True
        )
    else:
        try:
            # Extract content to markdown via API
            if not SKIP_MARKER_API:
                print("Marker API is enabled. Using Marker API to extract content to markdown.")
                markdown_dir = os.path.join(embedding_folder, "markdown")
                md_path, saved_images, md_document = extract_pdf_content_to_markdown_via_api(pdf_path, markdown_dir)
                doc_processor.set_md_document(md_document)
            else:
                print("Marker API is disabled. Using local PDF extraction.")
                markdown_dir = os.path.join(embedding_folder, "markdown")
                md_path, saved_images, md_document = extract_pdf_content_to_markdown(pdf_path, markdown_dir)
                doc_processor.set_md_document(md_document)
        except Exception as e:
            print(f"Error extracting content to markdown via API: {e}")
            # Use _doc to extract searchable content
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
            md_path = os.path.join(markdown_dir, "content.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(doc_processor.get_md_document())

            # Use the texts directly instead of splitting again
            print(f"Number of pages processed: {len(texts)}")
        else:
            # Split the document into chunks when markdown extraction succeeded
            average_page_length = sum(len(doc.page_content) for doc in _document) / len(_document)
            chunk_size = int(average_page_length // 3)
            print(f"Average page length: {average_page_length}")
            print(f"Chunk size: {chunk_size}")
            print("Creating new embeddings...")
            texts = create_searchable_chunks(_doc, chunk_size)
            print(f"length of document chunks generated for get_response_source:{len(texts)}")

        # Initialize image files and try to append image context to texts with error handling
        try:
            markdown_dir = os.path.join(embedding_folder, "markdown")
            image_context_path, _ = initialize_image_files(markdown_dir)

            with open(image_context_path, "r") as f:
                image_context = json.load(f)

            # Only process image context if there are actual images
            if image_context:
                print(f"Found {len(image_context)} images with context")

                # Create a temporary FAISS index for similarity search
                temp_db = FAISS.from_documents(texts, embeddings)

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
                print("No image context found to process")
        except Exception as e:
            print(f"Error processing image context: {e}")
            print("Continuing without image context...")

        # Create the vector store to use as the index
        db = FAISS.from_documents(texts, embeddings)
        # Save the embeddings to the specified folder
        db.save_local(embedding_folder)

        try:
            # Generate and save document summary using the texts we created
            print("Generating document summary...")
            generate_document_summary(texts, embedding_folder, doc_processor.get_md_document())
            print("Document summary generated and saved successfully.")
        except Exception as e:
            print(f"Error generating document summary: {e}")
            print("Continuing without document summary...")

    return


def generate_document_summary(_document, embedding_folder, md_document=None):
    """
    Generate a comprehensive markdown-formatted summary of the document using multiple LLM calls.
    Documents can come from either processed PDFs or markdown files.
    """
    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)  # Using advanced model for better quality

    # First try to get content from markdown document
    combined_content = ""
    if md_document:
        print("Using provided markdown content...")
        combined_content = md_document

    # If no content in markdown document, fall back to document content
    if not combined_content and _document:
        print("Using document content as source...")
        combined_content = "\n\n".join(doc.page_content for doc in _document)

    if not combined_content:
        raise ValueError("No content available from either markdown document or document")

    # First generate the take-home message
    takehome_prompt = """
    Provide a single, impactful sentence that captures the most important takeaway from this document.

    Guidelines:
    - Be extremely concise and specific
    - Focus on the main contribution or finding
    - Use bold for key terms
    - Keep it to one sentence
    - Add a relevant emoji at the start of bullet points or the first sentence
    - For inline formulas use single $ marks: $E = mc^2$
    - For block formulas use double $$ marks:
      $$
      F = ma (just an example, may not be a real formula in the doc)
      $$

    Document: {document}
    """
    takehome_prompt = ChatPromptTemplate.from_template(takehome_prompt)
    str_parser = StrOutputParser()
    takehome_chain = takehome_prompt | llm | str_parser
    takehome = takehome_chain.invoke({"document": truncate_document(combined_content)})

    # Topics extraction
    topics_prompt = """
    Identify only the most essential topics/sections from this document.
    Be extremely selective and concise - only include major components.

    Return format:
    {{"topics": ["topic1", "topic2", ...]}}

    Guidelines:
    - Include maximum 4-5 topics
    - Focus only on critical sections
    - Use short, descriptive names

    Document: {document}
    """
    topics_prompt = ChatPromptTemplate.from_template(topics_prompt)
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    topics_chain = topics_prompt | llm | error_parser
    topics_result = topics_chain.invoke({"document": truncate_document(combined_content)})

    try:
        topics = topics_result.get("topics", [])
    except AttributeError:
        print("Warning: Failed to get topics. Using default topics.")
        topics = ["Overview", "Methods", "Results", "Discussion"]

    # Generate overview
    overview_prompt = """
    Provide a clear and engaging overview using bullet points.

    Guidelines:
    - Use 3-4 concise bullet points
    - **Bold** for key terms
    - Each bullet point should be one short sentence
    - For inline formulas use single $ marks: $E = mc^2$
    - For block formulas use double $$ marks:
      $$
      F = ma (just an example, may not be a real formula in the doc)
      $$

    Document: {document}
    """
    overview_prompt = ChatPromptTemplate.from_template(overview_prompt)
    overview_chain = overview_prompt | llm | str_parser
    overview = overview_chain.invoke({"document": truncate_document(combined_content)})

    # Generate summaries for each topic
    summaries = []
    for topic in topics:
        topic_prompt = """
        Provide an engaging summary for the topic "{topic}" using bullet points.

        Guidelines:
        - Use 2-3 bullet points
        - Each bullet point should be one short sentence
        - **Bold** for key terms
        - Use simple, clear language
        - Include specific metrics only if crucial
        - For inline formulas use single $ marks: $E = mc^2$
        - For block formulas use double $$ marks:
          $$
          F = ma (just an example, may not be a real formula in the doc)
          $$

        Document: {document}
        """
        topic_prompt = ChatPromptTemplate.from_template(topic_prompt)
        topic_chain = topic_prompt | llm | str_parser
        topic_summary = topic_chain.invoke({
            "topic": topic,
            "document": truncate_document(combined_content)
        })
        summaries.append((topic, topic_summary))

    # Combine everything into markdown format with welcome message and take-home message
    markdown_summary = f"""### 👋 Welcome to DeepTutor!

I'm your AI tutor 🤖 ready to help you understand this document.

### 💡 Key Takeaway
{takehome}

### 📚 Document Overview
{overview}

"""

    # Add emojis for common topic titles
    topic_emojis = {
        "introduction": "📖",
        "overview": "🔎",
        "background": "📚",
        "methods": "🔬",
        "methodology": "🔬", 
        "results": "📊",
        "discussion": "💭",
        "conclusion": "🎯",
        "future work": "🔮",
        "implementation": "⚙️",
        "evaluation": "📈",
        "analysis": "🔍",
        "design": "✏️",
        "architecture": "🏗️",
        "experiments": "🧪",
        "related work": "🔗",
        "motivation": "💪",
        "approach": "🎯",
        "system": "🖥️",
        "framework": "🔧",
        "model": "🤖",
        "data": "📊",
        "algorithm": "⚡",
        "performance": "⚡",
        "limitations": "⚠️",
        "applications": "💡",
        "default": "📌" # Default emoji for topics not in the mapping
    }

    for topic, summary in summaries:
        # Get emoji based on topic, defaulting to 📌 if not found
        topic_lower = topic.lower()
        emoji = next((v for k, v in topic_emojis.items() if k in topic_lower), topic_emojis["default"])

        markdown_summary += f"""### {emoji} {topic}
{summary}

"""

    markdown_summary += """
---
### 💬 Ask Me Anything!
Feel free to ask me any questions about the document! I'm here to help! ✨
"""

    document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
    with open(document_summary_path, "w", encoding='utf-8') as f:
        f.write(markdown_summary)

    return markdown_summary
