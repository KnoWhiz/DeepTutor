import os

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    get_llm,
    get_embedding_models,
)
from pipeline.science.pipeline.doc_processor import (
    generate_embedding,
)
from pipeline.science.pipeline.session_manager import ChatSession
import logging
logger = logging.getLogger("tutorpipeline.science.get_rag_response")


async def get_standard_rag_response(
    prompt_string: str,
    user_input: str,
    chat_history: str,
    embedding_folder: str,
    embedding_type: str = 'default',
    chat_session: ChatSession = None,
    doc: dict = None,
    document: dict = None,
    file_path: str = None
):
    """
    Standard function for RAG-based response generation.
    
    Args:
        prompt_string: The system prompt to use
        user_input: The user's query
        chat_history: The conversation history (can be empty string)
        embedding_folder: Path to the folder containing embeddings
        embedding_type: Type of embedding model to use ('default' or 'lite')
        chat_session: Optional ChatSession object for generating embeddings if needed
        doc: Optional document dict for generating embeddings if needed
        document: Optional document dict for generating embeddings if needed
        file_path: Optional file path for generating embeddings if needed
        
    Returns:
        str: The generated response
    """
    config = load_config()
    para = config['llm']
    llm = get_llm('basic', para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    embeddings = get_embedding_models(embedding_type, para)

    # Handle different embedding folders based on type
    actual_embedding_folder = os.path.join(embedding_folder, 'lite_embedding') if embedding_type == 'lite' else embedding_folder

    try:
        db = FAISS.load_local(actual_embedding_folder, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        if embedding_type == 'lite':
            # For lite mode, try to generate embeddings if loading fails
            if chat_session and doc and document and file_path:
                await generate_embedding(chat_session.mode, doc, document, file_path, embedding_folder=embedding_folder)
                db = FAISS.load_local(actual_embedding_folder, embeddings, allow_dangerous_deserialization=True)
            else:
                raise Exception(f"Failed to load lite embeddings and missing parameters to generate them: {str(e)}")
        else:
            # For default mode, try markdown embeddings as fallback
            try:
                db = FAISS.load_local(
                    os.path.join(embedding_folder, "markdown"), 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                raise Exception(f"Failed to load embeddings: {str(e)}")

    retriever = db.as_retriever(search_kwargs={"k": config['retriever']['k']})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_string),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(x["context"]),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | error_parser
    )

    retrieve_docs = (lambda x: x["input"]) | retriever
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain
    )

    parsed_result = chain.invoke({
        "input": user_input,
        "chat_history": truncate_chat_history(chat_history) if chat_history else ""
    })
    
    return parsed_result['answer']