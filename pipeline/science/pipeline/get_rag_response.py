import os

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    get_llm,
)
from pipeline.science.pipeline.embeddings import (
    load_embeddings,
)
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
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
    file_path: str = None,
    stream: bool = False
):
    """
    Basic function for RAG-based response generation.
    
    Args:
        prompt_string: The system prompt to use
        user_input: The user's query
        chat_history: The conversation history (can be empty string)
        embedding_folder: Path to the folder containing embeddings
        embedding_type: Type of embedding model to use (default, lite, small)
        chat_session: Optional ChatSession object for generating embeddings if needed
        doc: Optional document dict for generating embeddings if needed
        document: Optional document dict for generating embeddings if needed
        file_path: Optional file path for generating embeddings if needed
        stream: Whether to stream the response
    Returns:
        str: The generated response
    """
    config = load_config()
    para = config['llm']
    llm = get_llm('basic', para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    if chat_session is None:
        chat_session = ChatSession()

    try:
        # Handle different embedding folders based on type
        if chat_session.mode == ChatMode.LITE:
            actual_embedding_folder = os.path.join(embedding_folder, 'lite_embedding')
        elif chat_session.mode == ChatMode.BASIC or chat_session.mode == ChatMode.ADVANCED:
            actual_embedding_folder = os.path.join(embedding_folder, 'markdown')
        else:
            actual_embedding_folder = embedding_folder
    except Exception as e:
        logger.exception(f"Failed to load session mode: {str(e)}")
        actual_embedding_folder = os.path.join(embedding_folder, 'markdown')

    try:
        db = load_embeddings(actual_embedding_folder, embedding_type)
    except Exception as e:
        logger.exception(f"Failed to load embeddings: {str(e)}")

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

    if not stream:
        parsed_result = chain.invoke({
            "input": user_input,
            "chat_history": truncate_chat_history(chat_history) if chat_history else ""
        })
    else:
        parsed_result = chain.invoke({
            "input": user_input,
            "chat_history": truncate_chat_history(chat_history) if chat_history else ""
        })

    # Memory cleanup
    db = None
    
    return parsed_result['answer']