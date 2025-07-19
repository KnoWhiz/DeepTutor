#!/usr/bin/env python3
"""
Streamlined Azure OpenAI PDF Chatbot API

This module provides a single function to upload PDF files, process chat history,
and return a streaming generator for responses using Azure OpenAI's Assistants API.
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path

try:
    from openai import AzureOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError("Required packages not installed. Please run: pip install openai python-dotenv") from e

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_files_and_stream_response(
    file_paths: List[str],
    chat_history: str,
    user_query: str,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: str = "gpt-4o"
) -> Generator[str, None, None]:
    """
    Upload PDF files, process chat history, and return a streaming generator for the response.
    
    Args:
        file_paths: List of paths to PDF files to upload and analyze
        chat_history: Previous chat history as a string (can be empty)
        user_query: The current user question/query
        api_key: Azure OpenAI API key (if not provided, loads from env)
        endpoint: Azure OpenAI endpoint (if not provided, loads from env)
        model: Model to use (default: gpt-4o)
        
    Yields:
        str: Streaming response chunks from the assistant
        
    Raises:
        Exception: If any part of the process fails
    """
    # Load environment variables if not provided
    if not api_key or not endpoint:
        load_dotenv()
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY_BACKUP")
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP")
    
    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI API key and endpoint are required")
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-05-01-preview",
        azure_endpoint=endpoint
    )
    
    uploaded_files = []
    vector_store_id = None
    assistant_id = None
    thread_id = None
    
    try:
        # Step 1: Upload PDF files
        logger.info(f"Uploading {len(file_paths)} PDF files...")
        uploaded_file_ids = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            if not file_path.lower().endswith('.pdf'):
                logger.warning(f"Skipping non-PDF file: {file_path}")
                continue
            
            # Check file size (max 512MB)
            file_size = os.path.getsize(file_path)
            if file_size > 512 * 1024 * 1024:
                logger.warning(f"File too large (max 512MB): {file_path}")
                continue
            
            with open(file_path, "rb") as file:
                uploaded_file = client.files.create(
                    file=file,
                    purpose="assistants"
                )
            
            uploaded_file_ids.append(uploaded_file.id)
            uploaded_files.append(uploaded_file.id)
            logger.info(f"Uploaded: {Path(file_path).name} (ID: {uploaded_file.id})")
        
        if not uploaded_file_ids:
            raise Exception("No files were successfully uploaded")
        
        # Step 2: Create vector store
        logger.info("Creating vector store...")
        vector_store = client.beta.vector_stores.create(
            name="PDF Document Store",
            file_ids=uploaded_file_ids
        )
        vector_store_id = vector_store.id
        
        # Wait for file processing
        logger.info("Waiting for file processing...")
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            vector_store_status = client.beta.vector_stores.retrieve(vector_store.id)
            if vector_store_status.file_counts.completed == len(uploaded_file_ids):
                logger.info("All files processed successfully")
                break
            elif vector_store_status.file_counts.failed > 0:
                logger.warning(f"Some files failed to process: {vector_store_status.file_counts.failed}")
            time.sleep(5)
        
        # Step 3: Create assistant
        logger.info("Creating assistant...")
        assistant = client.beta.assistants.create(
            name="PDF Papers Researcher and Professor",
            instructions="""
            You are a professional deep thinking researcher reading papers. Analyze the papers context content and provide detailed answers.
            
            If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or \n$$...\n$$.

            When answering questions:
            1. Base your responses on the content from the uploaded PDF files
            2. Provide specific citations for each claim or fact you mention
            3. If you cannot find relevant information in the documents, clearly state this
            4. Be thorough and accurate in your analysis
            5. Break down complex topics into clear, understandable explanations

            Always prioritize accuracy and provide evidence-based responses.
            """,
            tools=[{"type": "file_search"}],
            model=model,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        assistant_id = assistant.id
        
        # Step 4: Create thread
        logger.info("Creating conversation thread...")
        thread = client.beta.threads.create()
        thread_id = thread.id
        
        # Step 5: Add chat history to thread (if provided)
        if chat_history and chat_history.strip():
            logger.info("Adding chat history to thread...")
            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=f"Here is our previous conversation context:\n\n{chat_history}"
            )
        
        # Step 6: Add user query to thread
        logger.info("Adding user query to thread...")
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_query
        )
        
        # Step 7: Create streaming run and yield response chunks
        logger.info("Starting streaming response...")
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id
        ) as stream:
            for event in stream:
                if event.event == "thread.message.delta":
                    # Get the delta content
                    if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                        for content_item in event.data.delta.content:
                            if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                                # Yield the text chunk
                                yield content_item.text.value
                
                elif event.event == "thread.run.completed":
                    logger.info("Streaming run completed")
                    break
                
                elif event.event == "thread.run.failed":
                    logger.error(f"Streaming run failed: {event.data.last_error}")
                    yield f"Error: {event.data.last_error}"
                    break
    
    except Exception as e:
        logger.error(f"Error in process_files_and_stream_response: {e}")
        yield f"Error: {str(e)}"
    
    finally:
        # Cleanup resources
        logger.info("Cleaning up resources...")
        
        # Clean up assistant
        if assistant_id:
            try:
                client.beta.assistants.delete(assistant_id)
                logger.info(f"Deleted assistant: {assistant_id}")
            except Exception as e:
                logger.warning(f"Failed to delete assistant: {e}")
        
        # Clean up vector store
        if vector_store_id:
            try:
                client.beta.vector_stores.delete(vector_store_id)
                logger.info(f"Deleted vector store: {vector_store_id}")
            except Exception as e:
                logger.warning(f"Failed to delete vector store: {e}")
        
        # Clean up uploaded files
        for file_id in uploaded_files:
            try:
                client.files.delete(file_id)
                logger.info(f"Deleted file: {file_id}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_id}: {e}")


def main():
    """
    Example usage of the streaming API function.
    """
    # Example PDF file paths
    pdf_files = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf",
        "/Users/bingran_you/Downloads/2504.07923v1.pdf"
    ]
    
    # Example chat history
    chat_history = """
    User: What is the main topic of these papers?
    Assistant: The papers discuss quantum photonics and multiplexed single photon sources.
    User: Can you explain the key innovations?
    Assistant: The key innovations include novel approaches to multiplexing single photon sources for improved efficiency.
    """
    
    # Example user query
    user_query = "What are the experimental results and how do they compare to theoretical predictions?"
    
    # Filter existing files
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ Error: No PDF files found!")
        return
    
    print(f"📚 Processing {len(existing_files)} PDF file(s)...")
    print("🔄 Streaming response:\n")
    
    try:
        # Stream the response
        for chunk in process_files_and_stream_response(
            file_paths=existing_files,
            chat_history=chat_history,
            user_query=user_query
        ):
            print(chunk, end="", flush=True)
        
        print("\n\n✅ Response completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 