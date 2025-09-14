"""
Claude Code SDK Chatbot Implementation

This module implements a chatbot using the Claude Code SDK for code analysis and generation.
It provides a streaming response generator similar to the get_response function format.
"""

import os
from typing import Generator, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
from dotenv import load_dotenv
import json
import uuid
import asyncio

try:
    from bson import ObjectId
    from anthropic import Anthropic
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install: pip install anthropic pymongo")
    ObjectId = None
    Anthropic = None

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage, TextBlock, ResultMessage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_chat_history_path(session_id: str) -> str:
    """Get the full path for a chat history file.

    Args:
        session_id: Unique identifier for the chat session

    Returns:
        str: Full path to the chat history JSON file
    """
    base_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    return os.path.join(base_dir, f"{session_id}.json")


def create_session_id() -> str:
    """Create a unique session ID using timestamp and UUID.

    Returns:
        str: Unique session identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"chat_session_{timestamp}_{unique_id}"


def save_chat_history(session_id: str, chat_history: List[Dict]) -> None:
    """Save chat history to a JSON file.

    Args:
        session_id: Unique identifier for the chat session
        chat_history: List of chat messages with their metadata
    """
    if not chat_history:
        return

    file_path = get_chat_history_path(session_id)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "last_updated": datetime.now().isoformat(),
                "messages": chat_history
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.info(f"Error saving chat history: {e}")


def load_chat_history(session_id: str) -> Optional[List[Dict]]:
    """Load chat history from a JSON file.

    Args:
        session_id: Unique identifier for the chat session

    Returns:
        Optional[List[Dict]]: List of chat messages if file exists, None otherwise
    """
    file_path = get_chat_history_path(session_id)

    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", [])
    except Exception as e:
        logger.info(f"Error loading chat history: {e}")
        return None


def delete_chat_history(session_id: str) -> bool:
    """Delete a chat history file.

    Args:
        session_id: Unique identifier for the chat session

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    file_path = get_chat_history_path(session_id)

    if not os.path.exists(file_path):
        return False

    try:
        os.remove(file_path)
        return True
    except Exception as e:
        logger.info(f"Error deleting chat history: {e}")
        return False


def cleanup_old_sessions() -> None:
    """Clean up chat history files older than 24 hours."""
    base_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    if not os.path.exists(base_dir):
        return

    current_time = datetime.now()

    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(base_dir, filename)
        try:
            # Check file modification time
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            # Delete if older than 24 hours
            if (current_time - file_time).total_seconds() > 86400:  # 24 hours in seconds
                os.remove(file_path)
        except Exception as e:
            logger.info(f"Error cleaning up old session {filename}: {e}")

# Copy Question class from utils.py
class Question:
    """
    Represents a question with its text, language, and type information.

    Attributes:
        text (str): The text content of the question
        language (str): The detected language of the question (e.g., "English")
        question_type (str): The type of question (e.g., "local" or "global" or "image")
        special_context (str): Special context for the question
        answer_planning (dict): Planning information for constructing the answer
    """

    def __init__(self, text="", language="English", question_type="global", special_context="", answer_planning=None, image_url=None):
        """
        Initialize a Question object.

        Args:
            text (str): The text content of the question
            language (str): The language of the question
            question_type (str): The type of the question (local or global or image)
            special_context (str): Special context for the question
            answer_planning (dict): Planning information for constructing the answer
            image_url (str): The image url for the image question
        """
        self.text = text
        self.language = language
        if question_type not in ["local", "global", "image"]:
            self.question_type = "global"
        else:
            self.question_type = question_type

        self.special_context = special_context
        self.answer_planning = answer_planning or {}
        self.image_url = image_url   # This is the image url for the image question

    def __str__(self):
        """Return string representation of the Question."""
        return f"Question(text='{self.text}', language='{self.language}', type='{self.question_type}', image_url='{self.image_url}')"

    def to_dict(self):
        """Convert Question object to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "question_type": self.question_type,
            "special_context": self.special_context,
            "answer_planning": self.answer_planning,
            "image_url": str(self.image_url)
        }

# Copy ChatMode enum and ChatSession class from session_manager.py
class ChatMode:
    LITE = "lite"
    BASIC = "basic"
    ADVANCED = "advanced"

def create_session_id_from_objectid() -> str:
    """Create a unique session ID using ObjectId."""
    return str(ObjectId())

# Copy ChatSession class from session_manager.py
@dataclass
class ChatSession:
    """Class to manage all chat session related information.

    This class replaces the need for ST's session state by maintaining all
    chat-related information in a single place.

    Attributes:
        session_id: Unique identifier for the session
        mode: Current chat mode (Lite, Basic or Advanced)
        chat_history: List of chat messages
        uploaded_files: Set of uploaded file paths
        current_language: Current programming language context
        is_initialized: Whether the session has been initialized
        accumulated_cost: Total accumulated cost for the current session
    """

    session_id: str = field(default_factory=create_session_id_from_objectid)
    mode: ChatMode = ChatMode.BASIC # ChatMode.LITE, ChatMode.BASIC or ChatMode.ADVANCED
    chat_history: List[Dict] = field(default_factory=list)
    uploaded_files: Set[str] = field(default_factory=set)
    current_language: Optional[str] = None
    is_initialized: bool = False
    current_message: Optional[str] = "" # Latest current response message from streaming tutor agent
    new_message_id: str = str(ObjectId()) # new message from user
    question: Optional[Question] = None # Question object
    formatted_context: Optional[Dict] = None # Formatted context for the question
    page_formatted_context: Optional[Dict] = None # Page-level formatted context for broader coverage
    accumulated_cost: float = 0.0 # Total accumulated cost for the current session

    def initialize(self) -> None:
        """Initialize the chat session if not already initialized."""
        if self.is_initialized:
            return

        # Load existing chat history if available
        loaded_history = load_chat_history(self.session_id)
        if loaded_history:
            self.chat_history = loaded_history

        self.is_initialized = True

    def add_message(self, message: Dict) -> None:
        """Add a new message to the chat history.

        Args:
            message: Dictionary containing message data
        """
        self.chat_history.append(message)
        save_chat_history(self.session_id, self.chat_history)

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        delete_chat_history(self.session_id)
        self.accumulated_cost = 0.0  # Reset accumulated cost when history is cleared

    def set_mode(self, mode: ChatMode) -> None:
        """Set the chat mode.

        Args:
            mode: New chat mode to set
        """
        self.mode = mode

    def add_file(self, file_path: str) -> None:
        """Add a file to the session's uploaded files.

        Args:
            file_path: Path to the uploaded file
        """
        self.uploaded_files.add(file_path)

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the session's uploaded files.

        Args:
            file_path: Path to the file to remove
        """
        self.uploaded_files.discard(file_path)

    def set_language(self, language: str) -> None:
        """Set the current programming language context.

        Args:
            language: Programming language to set
        """
        self.current_language = language
        
    def update_cost(self, cost: float) -> None:
        """Update the accumulated cost of the session.
        
        Args:
            cost: Cost to add to the total accumulated cost
        """
        self.accumulated_cost += cost
        logger.info(f"Updated accumulated cost for session {self.session_id}: ${self.accumulated_cost:.6f}")
        
    def get_accumulated_cost(self) -> float:
        """Get the total accumulated cost of the session.
        
        Returns:
            Total accumulated cost as a float
        """
        return self.accumulated_cost

    def to_dict(self) -> Dict:
        """Convert session state to dictionary format.

        Returns:
            Dict containing all session state information
        """
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "chat_history": self.chat_history,
            "uploaded_files": list(self.uploaded_files),
            "current_language": self.current_language,
            "last_updated": datetime.now().isoformat(),
            "current_message": self.current_message,
            "new_message_id": self.new_message_id,
            "question": self.question,
            "formatted_context": self.formatted_context,
            "page_formatted_context": self.page_formatted_context,
            "accumulated_cost": self.accumulated_cost
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatSession":
        """Create a ChatSession instance from dictionary data.

        Args:
            data: Dictionary containing session state information

        Returns:
            ChatSession instance initialized with the provided data
        """
        if "current_message" not in data:
            data["current_message"] = ""
        # Set default accumulated_cost if not present in data
        accumulated_cost = data.get("accumulated_cost", 0.0)
        session = cls(
            session_id=data["session_id"],
            mode=ChatMode(data["mode"]),
            chat_history=data["chat_history"],
            uploaded_files=set(data["uploaded_files"]),
            current_language=data["current_language"],
            current_message=data["current_message"],
            new_message_id=data["new_message_id"],
            question=data["question"],
            formatted_context=data["formatted_context"],
            page_formatted_context=data.get("page_formatted_context", None),
            accumulated_cost=accumulated_cost
        )
        session.is_initialized = True
        return session


async def get_claude_code_response_async(
    chat_session: ChatSession,
    file_path_list: List[str], 
    question: Question,
    chat_history: str, 
    deep_thinking: bool = True,
    stream: bool = True
) -> Generator[str, None, None]:
    """
    Async implementation of Claude Code SDK chatbot that analyzes codebases and performs web searches.
    
    Args:
        chat_session: Current chat session
        file_path_list: List containing the codebase folder path
        question: Question object with text and metadata
        chat_history: Previous chat history as string
        deep_thinking: Whether to use deep thinking mode
        stream: Whether to stream responses
        
    Yields:
        str: Response chunks in streaming format
    """
    import subprocess
    
    # Install Claude Code SDK globally
    try:
        yield "Installing Claude Code SDK globally...\n"
        subprocess.run([
            "npm", "install", "-g", "@anthropic-ai/claude-code"
        ], check=True, capture_output=True, text=True)
        yield "Claude Code SDK installed successfully.\n"
    except subprocess.CalledProcessError as e:
        yield f"Warning: Failed to install Claude Code SDK: {e}\n"
    
    codebase_folder_dir = file_path_list[0]
    
    # Load environment variables from project root
    project_root = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor"
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path, override=True)  # Override existing env vars
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        yield "Error: ANTHROPIC_API_KEY not found in .env file\n"
        return
    
    # Debug: Show that we have the API key (first few characters only)
    yield f"✅ API key loaded: {api_key[:15]}... (length: {len(api_key)})\n"
    
    yield "<response>\n"
    
    try:
        # Set environment variable for the current process
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
        # Configure Claude Code options
        options = ClaudeCodeOptions(
            cwd=codebase_folder_dir,
            allowed_tools=[
                "Read", "Glob", "Grep", "WebSearch"
            ],
            env={"ANTHROPIC_API_KEY": api_key},
            permission_mode="plan",  # Read-only mode, no file edits
            system_prompt=f"""You are a helpful coding assistant analyzing a codebase. 
            
            Context from chat history: {chat_history}
            
            Your task: {question.text}
            
            Instructions:
            1. First, explore and understand the codebase structure
            2. Analyze the relevant code files
            3. If you need additional information to provide a complete answer, use web search
            4. Provide a comprehensive response based on the codebase analysis and any web search results
            5. Focus on being helpful and informative
            
            Remember: You can only read files and search the web. Do not attempt to edit any files."""
        )
        
        # Create the prompt for Claude Code
        prompt = f"""Please analyze this codebase and answer the following question:

Question: {question.text}

Please:
1. First explore the codebase structure to understand what files are available
2. Read and analyze relevant code files
3. If you need additional context or information to provide a complete answer, perform web searches
4. Provide a comprehensive analysis and answer

The codebase is located at: {codebase_folder_dir}
"""
        
        # Use Claude Code SDK to analyze the codebase
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # Stream the text content immediately
                        yield block.text
                    elif hasattr(block, 'text'):
                        yield block.text
            elif isinstance(message, ResultMessage):
                # Handle result messages - show the final result
                if hasattr(message, 'result') and message.result:
                    yield message.result
            else:
                # Show progress for other message types (like UserMessage with tool results)
                message_str = str(message)
                if "ToolResultBlock" in message_str:
                    # Extract meaningful progress information
                    if "sample_code.py" in message_str:
                        yield "📄 Analyzing Python code file...\n"
                    elif "sample.js" in message_str:
                        yield "📄 Analyzing JavaScript code file...\n"
                    elif "README.md" in message_str:
                        yield "📄 Reading documentation file...\n"
                    elif "paper2.txt" in message_str:
                        yield "📄 Processing large text document...\n"
                    elif "total" in message_str:
                        yield "📁 Exploring directory structure...\n"
                elif hasattr(message, 'subtype') and message.subtype == 'init':
                    yield "🔧 Initializing Claude Code session...\n"
        
        yield "\n</response>\n"
        
    except Exception as e:
        yield f"Error during Claude Code analysis: {str(e)}\n"
        yield "</response>\n"


def get_claude_code_response(
    chat_session: ChatSession,
    file_path_list: List[str], 
    question: Question,
    chat_history: str, 
    deep_thinking: bool = True,
    stream: bool = True
) -> Generator[str, None, None]:
    """
    Synchronous wrapper for the async Claude Code SDK chatbot.
    
    Args:
        chat_session: Current chat session
        file_path_list: List containing the codebase folder path
        question: Question object with text and metadata
        chat_history: Previous chat history as string
        deep_thinking: Whether to use deep thinking mode
        stream: Whether to stream responses
        
    Yields:
        str: Response chunks in streaming format
    """
    async def async_generator():
        async for chunk in get_claude_code_response_async(
            chat_session, file_path_list, question, chat_history, deep_thinking, stream
        ):
            yield chunk
    
    # Run the async generator in a new event loop
    async def run_async_generator():
        chunks = []
        async for chunk in async_generator():
            chunks.append(chunk)
        return chunks
    
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use a different approach
            import concurrent.futures
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(run_async_generator())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                chunks = future.result()
        else:
            # If no loop is running, we can use it directly
            chunks = loop.run_until_complete(run_async_generator())
    except RuntimeError:
        # No event loop exists, create a new one
        chunks = asyncio.run(run_async_generator())
    
    # Yield all chunks
    for chunk in chunks:
        yield chunk


# Example usage and testing function
def test_claude_code_sdk():
    """
    Test function to demonstrate how to use the Claude Code SDK chatbot.
    """
    
    print("Testing Claude Code SDK chatbot...")
    print("=" * 70)
    
    # Activate conda environment
    print("Activating conda environment 'deeptutor'...")
    try:
        # Note: In practice, the conda environment should be activated before running this script
        # This is just for demonstration
        print("Please ensure you have activated the conda environment with: conda activate deeptutor")
    except Exception as e:
        print(f"Note: Please activate conda environment manually: {e}")
    
    # Create a test session
    session = ChatSession()
    session.initialize()
    
    # Create test questions to demonstrate different capabilities
    test_questions = [
        Question(
            text="Analyze the codebase structure and explain what each file does",
            language="English",
            question_type="global",
            special_context="Focus on understanding the overall architecture"
        ),
        # Question(
        #     text="Compare the Python and JavaScript implementations and identify similarities",
        #     language="English", 
        #     question_type="global",
        #     special_context="Cross-language analysis"
        # ),
        # Question(
        #     text="What design patterns are used in this codebase?",
        #     language="English",
        #     question_type="global", 
        #     special_context="Focus on design patterns and best practices"
        # )
    ]
    
    # Codebase folder directory
    codebase_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/claude_code_integration_test/test_files"
    
    # Sample chat history for testing
    chat_history = """Previous conversation:
User: Hello, I'd like to analyze some code.
Assistant: I'd be happy to help you analyze your code! Please share the codebase or specific files you'd like me to review.
"""
    
    # Test each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {question.text}")
        print('='*70)
        
        try:
            response_generator = get_claude_code_response(
                chat_session=session,
                file_path_list=[codebase_dir],
                question=question,
                chat_history=chat_history,
                deep_thinking=True,
                stream=True
            )
            
            print("Streaming response:")
            print("-" * 50)
            response_content = ""
            for chunk in response_generator:
                print(chunk, end="", flush=True)
                response_content += chunk
            
            # Add this response to chat history for next test
            chat_history += f"\nUser: {question.text}\nAssistant: {response_content}"
            
            print("\n" + "-" * 50)
            print(f"Test {i} completed successfully!")
                
        except Exception as e:
            print(f"Error during test {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Run the test
    test_claude_code_sdk()
