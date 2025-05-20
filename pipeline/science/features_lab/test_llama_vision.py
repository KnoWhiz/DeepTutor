import os
import sys
from dotenv import load_dotenv

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
science_dir = os.path.dirname(current_dir)
pipeline_dir = os.path.dirname(science_dir)
project_root = os.path.dirname(pipeline_dir)
sys.path.append(project_root)

# Import the function to test
from pipeline.science.features_lab.images_understanding_test import process_image_with_llama

# Load environment variables
load_dotenv()

def test_llama_vision():
    """Test the process_image_with_llama function with a sample image."""
    print("Testing Llama-4-Maverick vision model...")
    
    # Sample image URL (using the same one from the main script)
    test_image_url = "https://knowhiztutorrag.blob.core.windows.net/knowhiztutorrag/file_appendix/3671da1e844b53ffbdccac7bc8c57341/images/_page_1_Figure_1.jpeg"
    
    # Test prompt
    test_prompt = "Describe this scientific figure in detail. What do you see?"
    
    # Non-streaming test
    print("\nRunning non-streaming test...")
    result = process_image_with_llama(test_image_url, test_prompt, stream=False)
    print("\nNon-streaming result:")
    print(result)
    
    # Streaming test
    print("\nRunning streaming test...")
    stream_result = process_image_with_llama(test_image_url, test_prompt, stream=True)
    print("\nStreaming result:")
    
    # Handle streaming response - improved to work with different formats
    try:
        streaming_content = ""
        for chunk in stream_result:
            # Check for different possible response structures
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content_text = delta.content
                    streaming_content += content_text
                    print(content_text, end="", flush=True)
            elif isinstance(chunk, dict) and "choices" in chunk:
                # Alternative format
                delta = chunk["choices"][0].get("delta", {})
                content_text = delta.get("content", "")
                if content_text:
                    streaming_content += content_text
                    print(content_text, end="", flush=True)
        
        print("\n\nComplete streaming response:")
        print(streaming_content)
    except Exception as e:
        print(f"Error processing stream: {e}")
        print(f"Stream result type: {type(stream_result)}")
        
    print("\nLlama-4-Maverick vision model test completed.")

if __name__ == "__main__":
    test_llama_vision() 