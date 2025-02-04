import os
import pprint
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from streamlit_float import *

from pipeline.config import load_config
from pipeline.utils import (
    tiktoken,
    truncate_chat_history,
    get_llm,
    get_embedding_models,
    robust_search_for
)


def get_response_source(_doc, _documents, user_input, answer, chat_history, embedding_folder):
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)

    # Load image context
    image_context_path = os.path.join(embedding_folder, "markdown/image_context.json")
    with open(image_context_path, 'r') as f:
        image_context = json.loads(f.read())
    
    # Create reverse mapping from description to image name. But note that multiple descriptions can map to the same image name.
    image_mapping = {}
    for image_name, descriptions in image_context.items():
        for desc in descriptions:
            image_mapping[desc] = image_name

    # Define the default filenames used by FAISS when saving
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        # Load existing embeddings
        print("Loading existing embeddings...")
        db = FAISS.load_local(
            embedding_folder, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # Split the documents into chunks, respecting page boundaries
        print("Creating new embeddings...")
        text_splitter = PageAwareTextSplitter(
            chunk_size=config['embedding']['chunk_size'],
            chunk_overlap=0
        )
        texts = text_splitter.split_documents(_documents)
        print(f"length of document chunks generated for get_response_source:{len(texts)}")

        # Create the vector store to use as the index
        db = FAISS.from_documents(texts, embeddings)
        # Save the embeddings to the specified folder
        db.save_local(embedding_folder)

    # Configure retriever with search parameters from config
    retriever = db.as_retriever(search_kwargs={"k": config['sources_retriever']['k']})

    # Get relevant chunks for both question and answer
    # question_chunks = retriever.get_relevant_documents(user_input)
    # answer_chunks = retriever.get_relevant_documents(answer)
    question_chunks = retriever.invoke(user_input)
    answer_chunks = retriever.invoke(answer)

    # Extract page content from chunks
    sources_question = [chunk.page_content for chunk in question_chunks]
    sources_answer = [chunk.page_content for chunk in answer_chunks]

    # Combine sources from question and answer and remove duplicates
    sources = list(set(sources_question + sources_answer))

    # Replace matching sources with image names
    sources = [image_mapping.get(source, source) for source in sources]

    # TEST
    # Display the list of strings in a beautiful way
    print("sources before refine:")
    pprint.pprint(sources)
    print(f"length of sources before refine: {len(sources)}")

    # Refine and limit sources
    markdown_dir = os.path.join(embedding_folder, "markdown")
    sources = refine_sources(_doc, _documents, sources, markdown_dir, user_input)

    # TEST
    # Display the list of strings in a beautiful way
    print("sources after refine:")
    pprint.pprint(sources)
    print(f"length of sources after refine: {len(sources)}")
    return sources


def refine_sources(_doc, _documents, sources, markdown_dir, user_input):
    """
    Refine sources by checking if they can be found in the document
    Only get first 20 sources
    Show them in the order they are found in the document
    Preserve image filenames without filtering them
    """
    refined_sources = []
    image_sources = []
    
    # First separate image sources from text sources
    text_sources = []
    for source in sources:
        # Check if source looks like an image filename (has image extension)
        if any(source.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
            image_sources.append(source)
        else:
            text_sources.append(source)
    
    # Process text sources as before
    for page in _doc:
        for source in text_sources:
            text_instances = robust_search_for(page, source)
            if text_instances:
                refined_sources.append(source)
    
    # Combine image sources with refined text sources
    final_sources = image_sources + refined_sources
    return final_sources[:20]


class PageAwareTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter that respects page boundaries"""
    
    def split_documents(self, documents):
        """Split documents while respecting page boundaries"""
        final_chunks = []
        
        for doc in documents:
            # Get the page number from the metadata
            page_num = doc.metadata.get("page", 0)
            text = doc.page_content
            
            # Use parent class's splitting logic first
            chunks = super().split_text(text)
            
            # Create new documents for each chunk with original metadata
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                # Update metadata to indicate chunk position
                metadata["chunk_index"] = i
                final_chunks.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
                
        # Sort chunks by page number and then by chunk index
        final_chunks.sort(key=lambda x: (
            x.metadata.get("page", 0),
            x.metadata.get("chunk_index", 0)
        ))
        
        return final_chunks