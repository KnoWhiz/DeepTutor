"""Module for managing chat session state information."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime
from bson.objectid import ObjectId

from .chat_history_manager import (
    create_session_id,
    save_chat_history,
    load_chat_history,
    delete_chat_history
)
from pipeline.science.pipeline.utils import Question

import logging
logger = logging.getLogger("tutorpipeline.science.session_manager")


class ChatMode(Enum):
    """Enum for different chat modes."""
    LITE = "Lite"
    BASIC = "Basic"
    ADVANCED = "Advanced"


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
        if not "current_message" in data:
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