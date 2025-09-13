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

try:
    from bson import ObjectId
    from anthropic import Anthropic
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install: pip install anthropic pymongo")
    ObjectId = None
    Anthropic = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def create_session_id() -> str:
    """Create a unique session ID."""
    return str(ObjectId())

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

    session_id: str = field(default_factory=create_session_id)
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
        # For this implementation, we'll skip loading from file
        self.is_initialized = True

    def add_message(self, message: Dict) -> None:
        """Add a new message to the chat history.
        
        Args:
            message: Dictionary containing message data
        """
        self.chat_history.append(message)

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
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
    chat_history: List[Dict], 
    codebase_folder_dir: str,
    deep_thinking: bool = True,
    stream: bool = True
) -> Generator[str, None, None]:
    """
    Generate a response using Claude Code SDK for code analysis and generation.
    
    This function provides the same interface as get_response but uses Claude Code SDK
    for enhanced code understanding and generation capabilities.
    
    Args:
        chat_session: ChatSession object containing session information
        file_path_list: List of file paths to analyze
        question: Question object containing the user's question
        chat_history: List of previous chat messages
        codebase_folder_dir: Path to the codebase folder for context
        deep_thinking: Whether to use deep thinking mode (not used in this implementation)
        stream: Whether to return a streaming generator (always True for this implementation)
    
    Yields:
        str: Streaming response chunks
    """
    try:
        # Check if dependencies are available
        if Anthropic is None:
            raise ImportError("Anthropic library not available. Please install: pip install anthropic")
        

        load_dotenv()
        # Initialize Claude client
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Prepare the user input
        user_input = question.text
        user_input_string = str(user_input + "\n\n" + question.special_context)
        
        # Build context from uploaded files
        context_files = []
        for file_path in file_path_list:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    context_files.append({
                        "path": file_path,
                        "content": content
                    })
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
        
        # Add codebase folder context if provided
        if codebase_folder_dir and os.path.exists(codebase_folder_dir):
            try:
                # Get all Python files in the codebase folder
                for root, dirs, files in os.walk(codebase_folder_dir):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                context_files.append({
                                    "path": file_path,
                                    "content": content
                                })
                            except Exception as file_error:
                                logger.warning(f"Could not read codebase file {file_path}: {file_error}")
            except Exception as folder_error:
                logger.warning(f"Could not process codebase folder {codebase_folder_dir}: {folder_error}")
        
        # Build the system prompt
        system_prompt = """You are an expert code analysis and generation assistant. You have access to the user's codebase and uploaded files. 

Your capabilities include:
- Analyzing code structure and patterns
- Generating code based on requirements
- Explaining complex code logic
- Suggesting improvements and optimizations
- Debugging and fixing issues
- Creating documentation and comments

Guidelines:
1. Provide clear, accurate, and helpful responses
2. Use proper code formatting with syntax highlighting
3. Explain your reasoning when generating or modifying code
4. Consider best practices and coding standards
5. Be specific about file locations and line numbers when relevant
6. Use markdown formatting for better readability

Always provide practical, actionable advice that helps the user understand and improve their code."""

        # Build the user message with context
        user_message = f"""User Question: {user_input_string}

Context from uploaded files and codebase:
"""
        
        for file_info in context_files[:10]:  # Limit to first 10 files to avoid token limits
            user_message += f"\n--- File: {file_info['path']} ---\n"
            user_message += file_info['content'][:2000] + "\n"  # Limit content length
        
        if len(context_files) > 10:
            user_message += f"\n... and {len(context_files) - 10} more files"
        
        # Add chat history context
        if chat_history:
            user_message += "\n\nPrevious conversation:\n"
            for msg in chat_history[-5:]:  # Last 5 messages
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    user_message += f"{role}: {content}\n"
        
        # Create the streaming response
        def process_stream():
            try:
                yield "<response>\n\n"
                
                # Use Claude Code SDK for streaming response
                with client.messages.stream(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}]
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, 'text'):
                                yield event.delta.text
                        elif event.type == "message_stop":
                            break
                
                yield "\n\n</response>"
                
            except Exception as e:
                logger.error(f"Error in Claude Code SDK streaming: {e}")
                yield f"Error: {str(e)}\n\n</response>"
        
        return process_stream()
        
    except Exception as e:
        logger.error(f"Error in get_claude_code_response: {e}")
        error_message = str(e)
        
        def error_stream():
            yield "<response>\n\n"
            yield f"Error: {error_message}\n\n</response>"
        
        return error_stream()


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
        text="Can you analyze the code structure and suggest improvements?",
        language="English",
        question_type="global",
        special_context="Focus on code quality and best practices"
    )
    
    # Test file paths (adjust these to your actual files)
    file_paths = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/pipeline/get_response.py"
    ]
    
    # Codebase folder directory
    codebase_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/pipeline"
    
    # Empty chat history for testing
    chat_history = []
    
    # Test the function
    print("Testing Claude Code SDK chatbot...")
    print("=" * 50)
    
    try:
        response_generator = get_claude_code_response(
        chat_session=session,
        file_path_list=file_paths,
        question=question,
            chat_history=chat_history,
            codebase_folder_dir=codebase_dir,
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
