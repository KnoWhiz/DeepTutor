#!/usr/bin/env python3
"""
Example usage of the streamlined Azure OpenAI PDF Chatbot API
"""

import os
from openai_streaming_api import process_files_and_stream_response

def main():
    """
    Simple example of using the streaming API function.
    """
    
    # List of PDF files to analyze
    pdf_files = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf",
        "/Users/bingran_you/Downloads/2504.07923v1.pdf"
    ]
    
    # Previous chat history (can be empty string)
    chat_history = """
    User: What is the main topic of these papers?
    Assistant: The papers discuss quantum photonics and multiplexed single photon sources for quantum computing applications.
    User: Can you explain the key technical challenges?
    Assistant: The main challenges include maintaining coherence, scaling efficiency, and reducing noise in quantum systems.
    """
    
    # Current user question
    user_query = "What are the specific experimental results and how do they validate the theoretical models?"
    
    # Filter to only existing files
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No PDF files found!")
        return
    
    print(f"📚 Processing {len(existing_files)} PDF file(s):")
    for file_path in existing_files:
        print(f"  - {os.path.basename(file_path)}")
    
    print(f"\n❓ Question: {user_query}")
    print("\n🤖 Assistant Response:")
    print("-" * 50)
    
    try:
        # Stream the response - each chunk is yielded as it arrives
        response_text = ""
        for chunk in process_files_and_stream_response(
            file_paths=existing_files,
            chat_history=chat_history,
            user_query=user_query
        ):
            print(chunk, end="", flush=True)
            response_text += chunk
        
        print(f"\n{'-' * 50}")
        print(f"✅ Response completed! Total length: {len(response_text)} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_without_chat_history():
    """
    Example usage without chat history.
    """
    pdf_files = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf"
    ]
    
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No PDF files found!")
        return
    
    print("\n" + "="*60)
    print("🔬 EXAMPLE: Without Chat History")
    print("="*60)
    
    try:
        for chunk in process_files_and_stream_response(
            file_paths=existing_files,
            chat_history="",  # Empty chat history
            user_query="Summarize the main contributions of this paper in 3 key points."
        ):
            print(chunk, end="", flush=True)
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_with_custom_credentials():
    """
    Example usage with custom API credentials.
    """
    # You can pass custom credentials instead of using environment variables
    custom_api_key = "your_custom_api_key_here"
    custom_endpoint = "https://your-custom-endpoint.openai.azure.com/"
    
    pdf_files = ["/path/to/your/paper.pdf"]
    
    print("\n" + "="*60)
    print("🔑 EXAMPLE: With Custom Credentials")
    print("="*60)
    
    try:
        for chunk in process_files_and_stream_response(
            file_paths=pdf_files,
            chat_history="",
            user_query="What is this paper about?",
            api_key=custom_api_key,
            endpoint=custom_endpoint,
            model="gpt-4o"  # You can specify different models
        ):
            print(chunk, end="", flush=True)
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Run example without chat history
    example_without_chat_history()
    
    # Uncomment to run example with custom credentials
    # example_with_custom_credentials() 