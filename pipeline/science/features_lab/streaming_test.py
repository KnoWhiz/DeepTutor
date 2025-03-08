import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dotenv import load_dotenv
import logging
import uuid
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
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

# Create history directory if it doesn't exist
HISTORY_DIR = Path("./chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)

class ColoredStreamingHandler(BaseCallbackHandler):
    """
    Custom streaming handler that prints tokens as they are generated with color formatting.
    """
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m"
    }
    
    def __init__(self) -> None:
        """Initialize the streaming handler."""
        self.full_response = ""
        self.start_time = time.time()
        self.token_count = 0
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Process a new token from the LLM.
        
        Args:
            token: The token received from the LLM
            **kwargs: Additional keyword arguments
        """
        # Print the token without a newline and flush immediately
        print(f"{self.COLORS['green']}{token}{self.COLORS['reset']}", end="", flush=True)
        self.full_response += token
        self.token_count += 1
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Process the end of LLM generation.
        
        Args:
            response: The final LLM result
            **kwargs: Additional keyword arguments
        """
        # Calculate and display generation stats
        elapsed_time = time.time() - self.start_time
        tokens_per_second = self.token_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add a newline and generation stats at the end of the response
        print()
        if self.token_count > 10:  # Only show stats for non-trivial responses
            print(f"{self.COLORS['cyan']}[Generated {self.token_count} tokens in {elapsed_time:.2f}s • {tokens_per_second:.1f} tokens/sec]{self.COLORS['reset']}")
        
    def get_response(self) -> str:
        """
        Get the accumulated response.
        
        Returns:
            str: The full response text
        """
        return self.full_response

def get_enhanced_system_prompt() -> str:
    """
    Creates a comprehensive system prompt for the chatbot.
    
    Returns:
        str: The system prompt
    """
    return """You are a helpful, friendly AI assistant designed to provide accurate and detailed information.

Guidelines:
1. Be concise yet thorough in your responses
2. When appropriate, structure information with bullet points or numbered lists
3. If you don't know something, admit it rather than making up information
4. Maintain memory of our conversation context and reference previous exchanges when relevant
5. Address the user's questions directly and provide practical examples where helpful

Your goal is to provide a helpful, educational experience while maintaining a conversational tone."""

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

def get_persistent_chat_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get a persistent chat history that saves messages to disk.
    
    Args:
        session_id: The session identifier
        
    Returns:
        BaseChatMessageHistory: A chat history that persists to disk
    """
    file_path = HISTORY_DIR / f"{session_id}.json"
    return FileChatMessageHistory(str(file_path))

def display_message_history(history: BaseChatMessageHistory) -> None:
    """
    Display the current message history in a formatted way.
    
    Args:
        history: The chat message history to display
    """
    if not history.messages:
        print("No previous messages.")
        return
    
    handler = ColoredStreamingHandler()
    print(f"\n{handler.COLORS['bold']}===== Conversation History ====={handler.COLORS['reset']}")
    
    for i, message in enumerate(history.messages):
        if isinstance(message, HumanMessage):
            print(f"{handler.COLORS['yellow']}User ({i+1}): {message.content}{handler.COLORS['reset']}")
        elif isinstance(message, AIMessage):
            print(f"{handler.COLORS['green']}AI ({i+1}): {message.content[:100]}{'...' if len(message.content) > 100 else ''}{handler.COLORS['reset']}")
    
    print(f"{handler.COLORS['bold']}============================={handler.COLORS['reset']}\n")

def create_chat_chain(params: Optional[Dict[str, Any]] = None, 
                      model_type: str = "advanced",
                      system_prompt: Optional[str] = None) -> RunnableWithMessageHistory:
    """
    Create a streaming chat chain with the modern LangChain components.
    
    Args:
        params: Parameters for the API handler
        model_type: Type of model to use (basic, advanced, creative, backup)
        system_prompt: Optional custom system prompt
        
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
    
    # Create a conversation prompt template with the enhanced system prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            system_prompt or get_enhanced_system_prompt()
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Build the chain with modern components
    chain = prompt | llm | StrOutputParser()
    
    # Add message history with persistent storage
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_persistent_chat_history,
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
    # Create a custom streaming handler with color support
    streaming_handler = ColoredStreamingHandler()
    
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
        print(f"\n{streaming_handler.COLORS['red']}Error: {str(e)}{streaming_handler.COLORS['reset']}")
        return f"Error: {str(e)}"

def generate_session_id() -> str:
    """
    Generate a unique session ID with timestamp for better identification.
    
    Returns:
        str: A unique session ID
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"session-{timestamp}-{str(uuid.uuid4())[:8]}"

def interactive_chat_session(chat_chain: RunnableWithMessageHistory) -> None:
    """
    Run an interactive chat session with the streaming chatbot.
    
    Args:
        chat_chain: The chat chain to use
    """
    handler = ColoredStreamingHandler()
    session_id = generate_session_id()
    
    print(f"{handler.COLORS['bold']}Starting interactive chat session ({session_id}){handler.COLORS['reset']}")
    print(f"{handler.COLORS['cyan']}Type 'exit' to quit, 'history' to show message history, 'clear' to clear history{handler.COLORS['reset']}")
    print("-" * 50)
    
    # Get history for the session
    history = get_persistent_chat_history(session_id)
    
    while True:
        user_input = input(f"\n{handler.COLORS['yellow']}You: {handler.COLORS['reset']}")
        
        # Handle special commands
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"{handler.COLORS['bold']}Ending chat session. Goodbye!{handler.COLORS['reset']}")
            break
        elif user_input.lower() == "history":
            display_message_history(history)
            continue
        elif user_input.lower() == "clear":
            history.clear()
            print(f"{handler.COLORS['cyan']}Chat history cleared.{handler.COLORS['reset']}")
            continue
        elif not user_input.strip():
            continue
        
        print(f"\n{handler.COLORS['green']}AI: {handler.COLORS['reset']}", end="", flush=True)
        chat_with_bot(chat_chain, user_input, session_id)

def main() -> None:
    """
    Main function to run the streaming chatbot.
    """
    handler = ColoredStreamingHandler()
    
    # Display banner
    print(f"\n{handler.COLORS['bold']}{handler.COLORS['cyan']}=" * 60)
    print(f"  Streaming LLM Chatbot  ")
    print("=" * 60 + f"{handler.COLORS['reset']}\n")
    
    # Create a chatbot with streaming enabled
    params = create_default_params()
    
    # Allow model selection
    print(f"{handler.COLORS['bold']}Select model type:{handler.COLORS['reset']}")
    print(f"1. {handler.COLORS['green']}Basic{handler.COLORS['reset']} (Default, Faster)")
    print(f"2. {handler.COLORS['cyan']}Advanced{handler.COLORS['reset']} (More Capable)")
    print(f"3. {handler.COLORS['yellow']}Creative{handler.COLORS['reset']} (Higher Temperature)")
    choice = input(f"\nEnter choice [1-3] or press Enter for default: ")
    
    model_type = "basic"
    if choice == "2":
        model_type = "advanced"
    elif choice == "3":
        model_type = "creative"
    
    print(f"\n{handler.COLORS['cyan']}Initializing {model_type} model...{handler.COLORS['reset']}")
    
    try:
        chat_chain = create_chat_chain(params, model_type)
        interactive_chat_session(chat_chain)
    except KeyboardInterrupt:
        print(f"\n\n{handler.COLORS['bold']}Session interrupted. Goodbye!{handler.COLORS['reset']}")
    except Exception as e:
        print(f"\n{handler.COLORS['red']}Error: {str(e)}{handler.COLORS['reset']}")
        logger.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
