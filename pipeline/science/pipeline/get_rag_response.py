import os
import sys
import logging
from typing import Any

# Add the project root to the Python path to make imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# After adding root to path, import the modules
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser

# Import from within the package structure
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    get_llm,
)
from pipeline.science.pipeline.embeddings import (
    load_embeddings,
)
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode

logger = logging.getLogger("tutorpipeline.science.get_rag_response")


async def get_db_rag_response(
    prompt_string: str,
    user_input: str,
    chat_history: str,
    chat_session: Any = None,
    db: Any = None,
    stream: bool = False
):
    """
    Basic function for RAG-based response generation. For single file response only.

    The most basic version of RAG response generation, also used in get_embedding_folder_rag_response

    Args:
        prompt_string: The system prompt to use
        user_input: The user's query
        chat_history: The conversation history (can be empty string)
        chat_session: The chat session to use
        db: The database to use
    Returns:
        str: The generated response
    """
    if chat_session is None:
        logger.info("Session not specified, creating new chat session")
        chat_session = ChatSession()

    config = load_config()
    para = config["llm"]

    if chat_session.mode == ChatMode.LITE:
        logger.info("RAG response in LITE mode")
        k_value = config["retriever"]["k"]  # Increase k for better context retrieval if in LITE mode
        k_value = min(k_value + 2, 8)  # Add more context chunks for LITE mode, but cap at reasonable limit
        # logger.info(f"Stream: {stream}")
        llm = get_llm("backup", para, stream)
        parser = StrOutputParser()
        # error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    else:
        logger.info("RAG response in other modes")
        k_value = config["retriever"]["k"]
        llm = get_llm("advanced", para)
        parser = StrOutputParser()
        # error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

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
        | parser
    )

    retrieve_docs = (lambda x: x["input"]) | retriever
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain
    )

    try:
        if stream:
            parsed_result = chain.stream({
                "input": user_input,
                "chat_history": processed_chat_history
            })
            logger.info(f"Response type: {type(parsed_result)}")
        else:
            parsed_result = chain.invoke({
                "input": user_input,
                "chat_history": processed_chat_history
            })
            parsed_result = parsed_result["answer"]
            logger.info(f"Response type: {type(parsed_result)}")
    except Exception as e:
        logger.exception(f"Error generating response: {str(e)}")
        return "I encountered an error while generating your response. Please try again with a different question."

    return parsed_result


async def get_embedding_folder_rag_response(
    prompt_string: str,
    user_input: str,
    chat_history: str,
    embedding_folder: str,
    embedding_type: str = "default",
    chat_session: Any = None,
    file_path: str = None,
    stream: bool = False
):
    """
    Basic function for RAG-based response generation. For single file response only.

    Mainly used in doc_summary generation

    Args:
        prompt_string: The system prompt to use
        user_input: The user's query
        chat_history: The conversation history (can be empty string)
        embedding_folder: Path to the folder containing embeddings
        embedding_type: Type of embedding model to use (default, lite, small)
        chat_session: Optional ChatSession object for generating embeddings if needed
        file_path: Optional file path for generating embeddings if needed
        stream: Whether to stream the response
    Returns:
        str: The generated response
    """
    config = load_config()
    
    if chat_session is None:
        chat_session = ChatSession()

    try:
        # Check if the path is a direct reference to a markdown folder
        if "markdown" in embedding_folder and os.path.exists(embedding_folder):
            logger.info(f"Using direct markdown folder: {embedding_folder}")
            actual_embedding_folder = embedding_folder
        # Handle different embedding folders based on mode
        elif chat_session.mode == ChatMode.LITE:
            actual_embedding_folder = os.path.join(embedding_folder, "lite_embedding")
        elif chat_session.mode == ChatMode.BASIC or chat_session.mode == ChatMode.ADVANCED:
            actual_embedding_folder = os.path.join(embedding_folder, "markdown")
        else:
            actual_embedding_folder = embedding_folder
    except Exception as e:
        logger.exception(f"Failed to load session mode: {str(e)}")
        actual_embedding_folder = os.path.join(embedding_folder, "markdown")

    logger.info(f"Loading embeddings from: {actual_embedding_folder}")
    try:
        db = load_embeddings([actual_embedding_folder], embedding_type)
    except Exception as e:
        logger.exception(f"Failed to load embeddings: {str(e)}")
        return "I'm sorry, I couldn't access the document information. Please try again later."

    # Increase k for better context retrieval if in LITE mode
    k_value = config["retriever"]["k"]
    if chat_session.mode == ChatMode.LITE:
        k_value = min(k_value + 2, 8)  # Add more context chunks for LITE mode, but cap at reasonable limit
        
    answer = await get_db_rag_response(
        prompt_string=prompt_string,
        user_input=user_input,
        chat_history=chat_history,
        chat_session=chat_session,
        db=db
    )

    # Memory cleanup
    db = None

    return answer

# Testing function - only used when file is run directly
def _run_tests():
    """Run tests for the RAG pipeline when the file is executed directly."""
    import asyncio
    import time
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting direct test execution of get_embedding_folder_rag_response.py")
    
    # Class for mock components (only used if real ones fail)
    class MockDB:
        def __init__(self):
            self.name = "MockDB"
            
        def as_retriever(self, search_kwargs=None):
            return MockRetriever()
    
    class MockRetriever:
        def __init__(self):
            pass
            
        def get_relevant_documents(self, query):
            return [
                MockDocument("This is a sample document for testing. It contains information about RAG (Retrieval Augmented Generation)."),
                MockDocument("RAG systems combine retrieval mechanisms with generative models to produce more accurate and contextual responses.")
            ]
    
    class MockDocument:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {"source": "mock_source"}
    
    class MockLLM:
        async def ainvoke(self, prompt):
            return "This is a mock response from the LLM for testing purposes. RAG (Retrieval Augmented Generation) is a technique that enhances generative AI by retrieving relevant information from external sources."
    
    # Test configuration
    test_prompt = "You are a helpful AI assistant that answers questions based on the provided context."
    test_user_input = "What is the main purpose of the RAG pipeline?"
    test_chat_history = ""
    
    # Use the specific embedded content path for testing
    def find_embeddings_folder():
        # Use the specified embedded content path as first priority
        specified_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/f7a7da83acea518683b13f634079fbfc/markdown"
        if os.path.exists(specified_path):
            logger.info(f"Using specified embedded content path: {specified_path}")
            return os.path.dirname(specified_path)  # Return parent folder so markdown gets appended properly
        
        # Fallback options if the specified path doesn't exist
        possibilities = [
            # Project structure embedding locations
            os.path.join(project_root, "data", "embeddings"),
            os.path.join(project_root, "pipeline", "data", "embeddings"),
            os.path.join(project_root, "pipeline", "science", "data", "embeddings"),
            # Current directory for testing
            os.path.join(os.path.dirname(__file__), "test_embeddings"),
        ]
        
        for folder in possibilities:
            if os.path.exists(folder):
                logger.info(f"Found embeddings folder: {folder}")
                return folder
        
        # If none found, create a test folder
        test_folder = os.path.join(os.path.dirname(__file__), "test_embeddings")
        os.makedirs(test_folder, exist_ok=True)
        logger.info(f"Created test embeddings folder: {test_folder}")
        return test_folder
    
    # Simple test runner
    async def run_test():
        print("\n🔍 RUNNING RAG PIPELINE TESTS 🔍\n")
        
        embedding_folder = find_embeddings_folder()
        
        # Try the real implementation first
        try:
            print("\nTest: Using get_embedding_folder_rag_response with embedded content path")
            response = await get_embedding_folder_rag_response(
                prompt_string=test_prompt,
                user_input=test_user_input,
                chat_history=test_chat_history,
                embedding_folder=embedding_folder,
                embedding_type="default"
            )
            print(f"Response: {response}")
            
            # If the first test succeeds, try with a more complex query
            complex_query = "Please explain how retrieval augmented generation works in detail."
            print("\nTest: Using a more complex query")
            response = await get_embedding_folder_rag_response(
                prompt_string=test_prompt,
                user_input=complex_query,
                chat_history=test_chat_history,
                embedding_folder=embedding_folder,
                embedding_type="default"
            )
            print(f"Complex query response: {response}")
                
        except Exception as e:
            print(f"Test with real implementation failed: {str(e)}")
            logger.exception("Error with real implementation")
            
            # Try with mock implementation
            print("\nFalling back to mock implementation for testing")
            
            # Create mocks
            original_load_embeddings = globals().get("load_embeddings")
            original_get_llm = globals().get("get_llm")
            
            # Replace with mocks temporarily
            globals()["load_embeddings"] = lambda folders, embedding_type="default": MockDB()
            globals()["get_llm"] = lambda mode, para: MockLLM()
            
            try:
                response = await get_embedding_folder_rag_response(
                    prompt_string=test_prompt,
                    user_input=test_user_input,
                    chat_history=test_chat_history,
                    embedding_folder=embedding_folder
                )
                print(f"Response with mocks: {response}")
            except Exception as e:
                print(f"Test with mock implementation failed: {str(e)}")
                logger.exception("Error with mock implementation")
            finally:
                # Restore original functions
                if original_load_embeddings:
                    globals()["load_embeddings"] = original_load_embeddings
                if original_get_llm:
                    globals()["get_llm"] = original_get_llm
        
        print("\n🔍 TESTS COMPLETED 🔍")
    
    # Run the tests
    asyncio.run(run_test())

# This block only runs when the file is executed directly
if __name__ == "__main__":
    _run_tests()