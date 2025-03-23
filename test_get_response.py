import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add the project root to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.science.pipeline.get_response import get_response
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode

async def test_get_response_streaming():
    """Test the get_response function with streaming enabled"""
    chat_session = ChatSession('test', ChatMode.BASIC)
    
    print("Starting get_response streaming test...")
    count = 0
    
    # Create a temporary directory for embeddings to avoid the empty list error
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_embedding_dir = Path(temp_dir) / "test_embedding"
        temp_embedding_dir.mkdir(exist_ok=True)
        
        # Create a dummy documents_summary.txt file
        with open(temp_embedding_dir / "documents_summary.txt", "w") as f:
            f.write("This is a test document summary.")
        
        # Process each chunk from the generator - properly awaiting
        try:
            generator = await get_response(
                chat_session=chat_session,
                file_path_list=[],
                user_input='What is the meaning of life?', 
                chat_history=[],
                embedding_folder_list=[str(temp_embedding_dir)],
                deep_thinking=False,
                stream=True
            )
            
            if isinstance(generator, tuple):
                # If it's a tuple, first element is async generator
                async_generator = generator[0]
                async for chunk in async_generator:
                    count += 1
                    print(f"CHUNK {count}: {type(chunk).__name__} - {chunk}")
            else:
                # If it's not a tuple, it should be an async generator
                async for chunk in generator:
                    count += 1
                    print(f"CHUNK {count}: {type(chunk).__name__} - {chunk}")
        except Exception as e:
            print(f"Error during streaming: {e}")
        
        print(f"\nReceived {count} chunks total")

async def test_get_response_non_streaming():
    """Test the get_response function with streaming disabled (should return a string)"""
    chat_session = ChatSession('test', ChatMode.BASIC)
    
    print("\nStarting get_response non-streaming test...")
    
    # Create a temporary directory for embeddings to avoid the empty list error
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_embedding_dir = Path(temp_dir) / "test_embedding"
        temp_embedding_dir.mkdir(exist_ok=True)
        
        # Create a dummy documents_summary.txt file
        with open(temp_embedding_dir / "documents_summary.txt", "w") as f:
            f.write("This is a test document summary.")
        
        try:
            # Get the complete response as a string
            response = await get_response(
                chat_session=chat_session,
                file_path_list=[],
                user_input='What is the meaning of life?', 
                chat_history=[],
                embedding_folder_list=[str(temp_embedding_dir)],
                deep_thinking=False,
                stream=False
            )
            
            print(f"Response type: {type(response).__name__}")
            if response:
                print(f"Response length: {len(response)}")
                print(f"Response preview: {response[:100]}...")
            else:
                print("Empty response")
        except Exception as e:
            print(f"Error during non-streaming test: {e}")

async def main():
    """Run all tests"""
    try:
        await test_get_response_streaming()
    except Exception as e:
        print(f"Streaming test failed: {e}")
    
    try:
        await test_get_response_non_streaming()
    except Exception as e:
        print(f"Non-streaming test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 