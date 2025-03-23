import asyncio
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.science.pipeline.get_response import get_query_helper, Question
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode

class MockRAGChain:
    """Simple mock for RAG chain that returns a fixed string"""
    def __init__(self, fixed_response="This is a mock response"):
        self.fixed_response = fixed_response
    
    async def __call__(self, **kwargs):
        # Sleep a bit to simulate processing time
        await asyncio.sleep(0.1)
        return self.fixed_response

async def test_get_query_helper_integration():
    """Test the get_query_helper integration with mocked dependencies"""
    chat_session = ChatSession('test', ChatMode.BASIC)
    
    print("Starting get_query_helper integration test...")
    
    # Create a temporary directory for test embeddings
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_embedding_dir = Path(temp_dir) / "test_embedding"
        temp_embedding_dir.mkdir(exist_ok=True)
        
        # Create a dummy documents_summary.txt file
        with open(temp_embedding_dir / "documents_summary.txt", "w") as f:
            f.write("This is a test document summary.")
        
        count = 0
        question_obj = None
        
        # Mock the LLM chain that would normally require embeddings
        with patch('pipeline.science.pipeline.get_response.get_llm') as mock_get_llm:
            # Configure mock to return a predictable output
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = {"question": "What is the meaning of life according to philosophy?", "question_type": "global"}
            mock_get_llm.return_value = mock_llm
            
            # Also patch the planning chain
            with patch('pipeline.science.pipeline.get_response.OutputFixingParser.from_llm') as mock_parser:
                mock_parser.return_value = MagicMock()
                
                # Process each chunk from the generator
                async for chunk in get_query_helper(
                    chat_session, 
                    'What is the meaning of life?', 
                    [], 
                    [str(temp_embedding_dir)]
                ):
                    count += 1
                    if isinstance(chunk, Question):
                        question_obj = chunk
                        print(f"CHUNK {count}: Question - {question_obj}")
                    else:
                        print(f"CHUNK {count}: {type(chunk).__name__} - {chunk}")
        
        print(f"\nReceived {count} chunks total")
        print(f"Final Question object: {question_obj}")

if __name__ == "__main__":
    asyncio.run(test_get_query_helper_integration()) 