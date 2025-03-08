import os
import sys
from typing import Dict, List, Optional, Any, Union, Callable
from dotenv import load_dotenv
import logging
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.api_handler import ApiHandler

# Configure logging - set to WARNING to suppress HTTP request logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("streaming_test.py")
logger.setLevel(logging.INFO)

# Disable httpx logs that interfere with streaming output
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

class CustomStreamingHandler(BaseCallbackHandler):
    """
    Custom streaming handler that prints tokens as they are generated.
    """
    def __init__(self) -> None:
        """Initialize the streaming handler."""
        self.full_response = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Process a new token from the LLM.
        
        Args:
            token: The token received from the LLM
            **kwargs: Additional keyword arguments
        """
        # Print the token without a newline and flush immediately
        print(token, end="", flush=True)
        self.full_response += token
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Process the end of LLM generation.
        
        Args:
            response: The final LLM result
            **kwargs: Additional keyword arguments
        """
        # Add a newline at the end of the response
        print()
        
    def get_response(self) -> str:
        """
        Get the accumulated response.
        
        Returns:
            str: The full response text
        """
        return self.full_response

def create_default_params() -> Dict[str, Any]:
    """
    Create default parameters for the API handler.
    
    Returns:
        Dict: Dictionary containing default parameters.
    """
    return {
        "openai_key_dir": ".env",
        "temperature": 0.0,
        "creative_temperature": 0.7,
        "llm_source": "azure",
        "stream": True
    }

def create_chat_chain(params: Optional[Dict[str, Any]] = None, model_type: str = "advanced") -> RunnableWithMessageHistory:
    """
    Create a streaming chat chain with the modern LangChain components.
    
    Args:
        params: Parameters for the API handler
        model_type: Type of model to use (basic, advanced, creative, backup)
        
    Returns:
        RunnableWithMessageHistory: A LangChain chain with message history handling
    """
    if params is None:
        params = create_default_params()
    
    # Initialize API handler with streaming explicitly enabled
    params["stream"] = True
    api_handler = ApiHandler(params, stream=True)
    
    # Get the requested model
    if model_type not in api_handler.models:
        logger.warning(f"Model type {model_type} not found. Using basic model.")
        model_type = "basic"
    
    llm = api_handler.models[model_type]["instance"]
    logger.info(f"Using {model_type} LLM: {llm}")
    
    # Create a conversation prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful, friendly AI assistant. Be concise and clear in your responses."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Build the chain with modern components
    chain = prompt | llm | StrOutputParser()
    
    # Add message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="history"
    )
    
    return chain_with_history

def chat_with_bot(chat_chain: RunnableWithMessageHistory, input_text: str, session_id: str = "default") -> str:
    """
    Send a message to the chatbot and display the streaming response.
    
    Args:
        chat_chain: The chat chain to use
        input_text: The input text from the user
        session_id: Identifier for the chat session
        
    Returns:
        str: The complete response after streaming
    """
    # Create a custom streaming handler
    streaming_handler = CustomStreamingHandler()
    
    # Configure the run with streaming callbacks
    config = RunnableConfig(
        callbacks=[streaming_handler],
        configurable={"session_id": session_id}
    )
    
    try:
        # Execute the chain with the streaming handler
        response = chat_chain.invoke(
            {"input": input_text},
            config=config
        )
        return response
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        return f"Error: {str(e)}"

def interactive_chat_session(chat_chain: RunnableWithMessageHistory) -> None:
    """
    Run an interactive chat session with the streaming chatbot.
    
    Args:
        chat_chain: The chat chain to use
    """
    print("Starting interactive chat session. Type 'exit' to quit.")
    print("-" * 50)
    
    session_id = "user-session-" + str(os.getpid())
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Ending chat session. Goodbye!")
            break
        
        print("\nAI: ", end="", flush=True)
        chat_with_bot(chat_chain, user_input, session_id)

if __name__ == "__main__":
    # Create a chatbot with streaming enabled
    params = create_default_params()
    chat_chain = create_chat_chain(params)
    
    # Run interactive session
    interactive_chat_session(chat_chain)
