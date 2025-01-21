import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from streamlit_float import *

from pipeline.config import load_config
from pipeline.utils import (
    tiktoken,
    truncate_chat_history,
    get_llm,
    get_embedding_models
)


def get_response_source(_doc, _documents, user_input, answer, chat_history, embedding_folder):
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)

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
        # Split the documents into chunks
        print("Creating new embeddings...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['embedding']['chunk_size'],
            chunk_overlap=config['embedding']['chunk_overlap']
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

    # TEST
    print(f"sources before refine: {sources}")
    print(f"length of sources before refine: {len(sources)}")

    # Refine and limit sources
    sources = refine_sources(_doc, _documents, sources)

    # TEST
    print(f"sources after refine: {sources}")
    print(f"length of sources after refine: {len(sources)}")
    return sources


def refine_sources(_doc, _documents, sources):
    """
    Refine sources by checking if they can be found in the document
    Only get first 10 sources
    Show then in the order they are found in the document
    """
    refined_sources = []
    for page in _doc:
        for source in sources:
            text_instances = page.search_for(source)
            if text_instances:
                refined_sources.append(source)
    return refined_sources[:20]