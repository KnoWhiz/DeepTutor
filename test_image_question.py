import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add the project root to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.tutor_agent import tutor_agent_basic_streaming
from pipeline.science.pipeline.get_response import Question

async def test_image_question():
    """Test handling of an image question"""
    chat_session = ChatSession('test', ChatMode.BASIC)
    print("Starting image question test...")
    
    # Create a temporary directory for embeddings
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_embedding_dir = Path(temp_dir) / "test_embedding"
        temp_embedding_dir.mkdir(exist_ok=True)
        
        # Create a dummy documents_summary.txt file
        with open(temp_embedding_dir / "documents_summary.txt", "w") as f:
            f.write("This is a test document summary.")
        
        # Mock the handling of an image question by setting the Question object manually
        question = Question(
            text="Can you provide a summary or explanation of the main content and significance of Figure 1?",
            language="English",
            question_type="image",
            image_url="https://example.com/test_image.jpg"
        )
        
        # Set up the chat session with our question
        chat_session.current_message = "What is in this image?"
        
        count = 0
        async for chunk in tutor_agent_basic_streaming(
            chat_session=chat_session,
            file_path_list=[],
            user_input="Can you explain Figure 1?",
            time_tracking={},
            deep_thinking=False,
            stream=True
        ):
            count += 1
            print(f"CHUNK {count}: {chunk}")
        
        print(f"\nReceived {count} chunks total")

if __name__ == "__main__":
    asyncio.run(test_image_question()) 