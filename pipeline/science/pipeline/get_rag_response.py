import os
from typing import Any
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


async def get_rag_response(
    prompt_string: str,
    user_input: str,
    chat_history: str,
    chat_session: ChatSession = None,
    db: Any = None,
    stream: bool = False
):
    """
    Basic function for RAG-based response generation. For single file response only.

    Args:
        prompt_string: The system prompt to use
        user_input: The user's query
        chat_history: The conversation history (can be empty string)
        chat_session: The chat session to use
        db: The database to use
    Returns:
        str: The generated response
    """
    config = load_config()
    para = config["llm"]
    llm = get_llm("basic", para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # Increase k for better context retrieval if in LITE mode
    k_value = config["retriever"]["k"]
    if chat_session is None:
        chat_session = ChatSession()
    if chat_session.mode == ChatMode.LITE:
        k_value = min(k_value + 2, 8)  # Add more context chunks for LITE mode, but cap at reasonable limit
        
    retriever = db.as_retriever(search_kwargs={"k": k_value})

    # Process chat history to ensure proper formatting
    processed_chat_history = truncate_chat_history(chat_history) if chat_history else ""
    
    # Create prompt template with better messaging sequence
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_string),
        ("human", "{input}")
    ])

    def format_docs(docs):
        # Enhanced document formatting that emphasizes document structure
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_docs.append(f"Document {i+1}:\n{doc.page_content}")
        return "\n\n".join(formatted_docs)

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

    try:
        parsed_result = chain.invoke({
            "input": user_input,
            "chat_history": processed_chat_history
        })
    except Exception as e:
        logger.exception(f"Error generating response: {str(e)}")
        return "I encountered an error while generating your response. Please try again with a different question."

    # Memory cleanup
    db = None

    return answer