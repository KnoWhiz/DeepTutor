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

try:
    from bson import ObjectId
    from anthropic import Anthropic
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install: pip install anthropic pymongo")
    ObjectId = None
    Anthropic = None

import os, asyncio
from claude_code_sdk import (
    query,
    ClaudeSDKClient, ClaudeCodeOptions,
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


def get_claude_code_response(
    chat_session: ChatSession,
    file_path_list: List[str], 
    question: Question,
    chat_history: str, 
    deep_thinking: bool = True,
    stream: bool = True
) -> Generator[str, None, None]:
    codebase_folder_dir = file_path_list[0]
    # pass the API key to the Claude Code process
    load_dotenv()
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    print(f"API_KEY: {API_KEY}")
    opts = ClaudeCodeOptions(
        env={"ANTHROPIC_API_KEY": API_KEY},
        permission_mode="plan"  # safe: read/plan only, no file edits/exec
    )
    def generate_response():
        yield "<response>\n"
        yield "Here is the place holder response chunk 1 from claude code sdk\n"
        yield "Here is the place holder response chunk 2 from claude code sdk\n"
        yield "Here is the place holder response chunk 3 from claude code sdk\n"
        yield "Here is the place holder response chunk 4 from claude code sdk\n"
        yield "Here is the place holder response chunk 5 from claude code sdk\n"
        yield "Here is the place holder response chunk 6 from claude code sdk\n"
        yield "Here is the place holder response chunk 7 from claude code sdk\n"
        yield "Here is the place holder response chunk 8 from claude code sdk\n"
        yield "Here is the place holder response chunk 9 from claude code sdk\n"
        yield "Here is the place holder response chunk 10 from claude code sdk\n"
        yield "</response>\n"
    return generate_response()


# Example usage and testing function
def test_claude_code_sdk():
    """
    Test function to demonstrate how to use the Claude Code SDK chatbot.
    """
    # Create a test session
    session = ChatSession()
    session.initialize()
    
    # Create a test question
    question = Question(
        # text="Can you analyze the code structure and suggest improvements?",
        text="Review the codebase and draw the flow of the code",
        language="English",
        question_type="global",
        special_context="Focus on code quality and best practices"
    )
    
    # Codebase folder directory
    codebase_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/claude_code_integration_test/test_files"
    
    # Empty chat history for testing
    chat_history = """chat history"""
    
    # Test the function
    print("Testing Claude Code SDK chatbot...")
    print("=" * 50)
    
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
        for chunk in response_generator:
            print(chunk, end="", flush=True)
            
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    # Run the test
    test_claude_code_sdk()
