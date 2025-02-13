"""Module for managing chat session state information."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import os
from datetime import datetime

from .chat_history_manager import (
    create_session_id,
    save_chat_history,
    load_chat_history,
    delete_chat_history
)


class ChatMode(Enum):
    """Enum for different chat modes."""
    BASIC = "basic"
    ADVANCED = "advanced"


@dataclass
class ChatSession:
    """Class to manage all chat session related information.
    
    This class replaces the need for ST's session state by maintaining all
    chat-related information in a single place.
    
    Attributes:
        session_id: Unique identifier for the session
        mode: Current chat mode (basic or advanced)
        chat_history: List of chat messages
        uploaded_files: Set of uploaded file paths
        current_language: Current programming language context
        is_initialized: Whether the session has been initialized
    """
    
    session_id: str = field(default_factory=create_session_id)
    mode: ChatMode = ChatMode.BASIC #   ChatMode.BASIC or ChatMode.ADVANCED
    chat_history: List[Dict] = field(default_factory=list)
    uploaded_files: Set[str] = field(default_factory=set)
    current_language: Optional[str] = None
    is_initialized: bool = False
    
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
            "last_updated": datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ChatSession":
        """Create a ChatSession instance from dictionary data.
        
        Args:
            data: Dictionary containing session state information
            
        Returns:
            ChatSession instance initialized with the provided data
        """
        session = cls(
            session_id=data["session_id"],
            mode=ChatMode(data["mode"]),
            chat_history=data["chat_history"],
            uploaded_files=set(data["uploaded_files"]),
            current_language=data["current_language"]
        )
        session.is_initialized = True
        return session 