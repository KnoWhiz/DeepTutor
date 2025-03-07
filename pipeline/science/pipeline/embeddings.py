import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler

import logging
logger = logging.getLogger("tutorpipeline.science.embeddings")

load_dotenv()

# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


def get_embedding_models(embedding_type, para):
    para = para
    api = ApiHandler(para)
    embedding_model_default = api.embedding_models['default']['instance']
    embedding_model_lite = api.embedding_models['lite']['instance']
    embedding_model_small = api.embedding_models['small']['instance']
    if embedding_type == 'default':
        return embedding_model_default
    elif embedding_type == 'lite':
        return embedding_model_lite
    elif embedding_type == 'small':
        return embedding_model_small
    else:
        return embedding_model_default


# Create markdown embeddings
def create_markdown_embeddings(md_document: str, output_dir: str | Path, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Create markdown embeddings from a markdown document and save them to the specified directory.

    Args:
        md_document: Markdown document
        output_dir: Directory where embeddings will be saved

    Returns:
        None
    """
    # Load the markdown file
    # Create and save markdown embeddings
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)

    logger.info("Creating markdown embeddings...")
    if md_document:
        # Create markdown directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Split markdown content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )
        markdown_texts = [
            Document(page_content=chunk, metadata={"source": "markdown"})
            for chunk in text_splitter.split_text(md_document)
        ]

        # Create and save markdown embeddings
        db_markdown = FAISS.from_documents(markdown_texts, embeddings)
        db_markdown.save_local(output_dir)
        logger.info(f"Saved {len(markdown_texts)} markdown chunks to {output_dir}")
    else:
        logger.info("No markdown content available to create markdown embeddings")


async def generate_LiteRAG_embedding(_doc, file_path, embedding_folder):
    """
    Generate LiteRAG embeddings for the document
    """
    config = load_config()
    para = config['llm']
    # file_id = generate_file_id(file_path)
    lite_embedding_folder = os.path.join(embedding_folder, 'lite_embedding')
    # Check if all necessary files exist to load the embeddings
    faiss_path = os.path.join(lite_embedding_folder, "index.faiss")
    pkl_path = os.path.join(lite_embedding_folder, "index.pkl")
    embeddings = get_embedding_models('lite', para)
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        # Try to load existing txt file in graphrag_embedding folder
        logger.info("LiteRAG embedding already exists. We can load existing embeddings...")
    else:
        # If embeddings don't exist, create them from raw text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        raw_text = "\n\n".join([page.get_text() for page in _doc])
        chunks = text_splitter.create_documents([raw_text])
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(lite_embedding_folder)


def load_embeddings(embedding_folder: str | Path, embedding_type: str = 'default'):
    """
    Load embeddings from the specified folder
    """
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models(embedding_type, para)
    db = FAISS.load_local(embedding_folder, embeddings, allow_dangerous_deserialization=True)
    return db