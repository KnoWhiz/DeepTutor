#!/usr/bin/env python3
"""
Azure OpenAI PDF Chatbot with Source Citations

This script implements a terminal-based chatbot that can analyze PDF files using Azure OpenAI's 
Assistants API with file search capabilities. It provides responses with source citations for 
each sentence based on the uploaded PDF content.

Features:
- Upload multiple PDF files to Azure OpenAI
- Create an assistant with file search capabilities
- Interactive terminal chat interface
- Source citations for each response
- Streaming response support
- Proper error handling and logging

Requirements:
- Azure OpenAI API key and endpoint in .env file
- Python packages: openai, python-dotenv
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from openai import AzureOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Required packages not installed. Please run:")
    print("pip install openai python-dotenv")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AzureOpenAIPDFChatbot:
    """
    A chatbot that uses Azure OpenAI Assistants API to answer questions based on PDF content.
    
    This class handles:
    - PDF file uploads to Azure OpenAI
    - Assistant creation with file search capabilities
    - Interactive chat sessions with source citations
    - Streaming and non-streaming response modes
    - Proper cleanup of resources
    """
    
    def __init__(self, streaming: bool = False):
        """
        Initialize the chatbot with Azure OpenAI configuration.
        
        Args:
            streaming: Whether to use streaming responses (default: False)
        """
        # Load environment variables
        load_dotenv()
        
        # Validate required environment variables
        # self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        # self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY_BACKUP")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP")
        
        if not self.api_key or not self.endpoint:
            logger.error("Missing required environment variables:")
            logger.error("- AZURE_OPENAI_API_KEY")
            logger.error("- AZURE_OPENAI_ENDPOINT")
            logger.error("Please set these in your .env file")
            sys.exit(1)
        
        # Initialize Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2024-05-01-preview",  # Required for Assistants API
                azure_endpoint=self.endpoint
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            sys.exit(1)
        
        # Initialize instance variables
        self.streaming = streaming
        self.assistant_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.uploaded_files: List[str] = []
        self.vector_store_id: Optional[str] = None
        
        # Log streaming mode
        logger.info(f"Streaming mode: {'enabled' if streaming else 'disabled'}")
    
    def upload_pdf_files(self, file_paths: List[str]) -> List[str]:
        """
        Upload PDF files to Azure OpenAI for processing.
        
        Args:
            file_paths: List of paths to PDF files to upload
            
        Returns:
            List of file IDs that were successfully uploaded
            
        Raises:
            Exception: If file upload fails
        """
        uploaded_file_ids = []
        
        for file_path in file_paths:
            try:
                # Validate file exists and is a PDF
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    continue
                
                if not file_path.lower().endswith('.pdf'):
                    logger.error(f"File is not a PDF: {file_path}")
                    continue
                
                # Check file size (max 512MB for Azure OpenAI)
                file_size = os.path.getsize(file_path)
                if file_size > 512 * 1024 * 1024:  # 512MB
                    logger.error(f"File too large (max 512MB): {file_path}")
                    continue
                
                logger.info(f"Uploading file: {file_path} ({file_size / (1024*1024):.2f} MB)")
                
                # Upload file to Azure OpenAI
                with open(file_path, "rb") as file:
                    uploaded_file = self.client.files.create(
                        file=file,
                        purpose="assistants"  # Required for Assistants API
                    )
                
                uploaded_file_ids.append(uploaded_file.id)
                self.uploaded_files.append(uploaded_file.id)
                logger.info(f"Successfully uploaded: {Path(file_path).name} (ID: {uploaded_file.id})")
                
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")
                continue
        
        if not uploaded_file_ids:
            raise Exception("No files were successfully uploaded")
        
        logger.info(f"Successfully uploaded {len(uploaded_file_ids)} files")
        return uploaded_file_ids
    
    def create_vector_store(self, file_ids: List[str]) -> str:
        """
        Create a vector store with the uploaded files for file search.
        
        Args:
            file_ids: List of file IDs to include in the vector store
            
        Returns:
            Vector store ID
            
        Raises:
            Exception: If vector store creation fails
        """
        try:
            logger.info("Creating vector store with uploaded files...")
            
            # Create vector store with files
            vector_store = self.client.beta.vector_stores.create(
                name="PDF Document Store",
                file_ids=file_ids
            )
            
            self.vector_store_id = vector_store.id
            logger.info(f"Vector store created: {vector_store.id}")
            
            # Poll for completion of file processing
            logger.info("Waiting for file processing to complete...")
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                vector_store_status = self.client.beta.vector_stores.retrieve(vector_store.id)
                
                if vector_store_status.file_counts.completed == len(file_ids):
                    logger.info("All files processed successfully")
                    break
                elif vector_store_status.file_counts.failed > 0:
                    logger.warning(f"Some files failed to process: {vector_store_status.file_counts.failed}")
                
                logger.info(f"Processing files... ({vector_store_status.file_counts.completed}/{len(file_ids)} completed)")
                time.sleep(5)
            else:
                logger.warning("File processing timeout - proceeding anyway")
            
            return vector_store.id
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def create_assistant(self) -> str:
        """
        Create an Azure OpenAI assistant with file search capabilities.
        
        Returns:
            Assistant ID
            
        Raises:
            Exception: If assistant creation fails
        """
        try:
            logger.info("Creating Azure OpenAI assistant...")
            
            # Create assistant with file search tool
            assistant = self.client.beta.assistants.create(
                name="PDF Papers Researcher and Professor",
                instructions="""
                You are a professional deep thinking researcher reading papers. Analyze the papers context content and provide detailed answers with source citations.
                
                If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or \n$$...\n$$.

When answering questions:
1. Base your responses on the content from the uploaded PDF files
2. Provide specific citations for each claim or fact you mention
3. Use the format [Source: filename, page X] for citations when possible
4. If you cannot find relevant information in the documents, clearly state this
5. Be thorough and accurate in your analysis
6. Break down complex topics into clear, understandable explanations

Always prioritize accuracy and provide evidence-based responses.""",
                tools=[{"type": "file_search"}],
                model="gpt-4.1",  # Use GPT-4.1 model to match api_handler.py configuration
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [self.vector_store_id]
                    }
                }
            )
            
            self.assistant_id = assistant.id
            logger.info(f"Assistant created successfully: {assistant.id}")
            return assistant.id
            
        except Exception as e:
            logger.error(f"Failed to create assistant: {e}")
            raise
    
    def create_thread(self) -> str:
        """
        Create a new conversation thread.
        
        Returns:
            Thread ID
            
        Raises:
            Exception: If thread creation fails
        """
        try:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
            logger.info(f"Thread created: {thread.id}")
            return thread.id
            
        except Exception as e:
            logger.error(f"Failed to create thread: {e}")
            raise
    
    def process_message_annotations(self, message_content: str, annotations: List[Any]) -> str:
        """
        Process message annotations to replace citation placeholders with readable citations.
        
        Args:
            message_content: The original message content with annotation placeholders
            annotations: List of annotation objects from the API response
            
        Returns:
            Processed message content with readable citations
        """
        processed_content = message_content
        
        for annotation in annotations:
            # Replace file citation annotations
            if hasattr(annotation, 'file_citation'):
                citation_text = f"[Source: {annotation.file_citation.file_id}]"
                processed_content = processed_content.replace(annotation.text, citation_text)
            
            # Replace file path annotations  
            elif hasattr(annotation, 'file_path'):
                file_path_text = f"[File: {annotation.file_path.file_id}]"
                processed_content = processed_content.replace(annotation.text, file_path_text)
        
        return processed_content
    
    def send_message_and_get_response(self, user_message: str) -> str:
        """
        Send a message to the assistant and get the response with citations.
        Supports both streaming and non-streaming modes.
        
        Args:
            user_message: The user's question or message
            
        Returns:
            Assistant's response with processed citations
            
        Raises:
            Exception: If message processing fails
        """
        try:
            # Add user message to thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=user_message
            )
            
            if self.streaming:
                return self._handle_streaming_response()
            else:
                return self._handle_non_streaming_response()
                
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _handle_streaming_response(self) -> str:
        """
        Handle streaming response from the assistant.
        
        Returns:
            Complete response text
        """
        try:
            logger.info("Creating streaming run...")
            
            # Create streaming run
            with self.client.beta.threads.runs.stream(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            ) as stream:
                response_text = ""
                current_message = ""
                
                for event in stream:
                    # Handle different event types
                    if event.event == "thread.message.delta":
                        # Get the delta content
                        if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                            for content_item in event.data.delta.content:
                                if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                                    # Stream the text to console
                                    chunk = content_item.text.value
                                    print(chunk, end="", flush=True)
                                    current_message += chunk
                    
                    elif event.event == "thread.message.completed":
                        # Message is complete, process annotations
                        if hasattr(event.data, 'content'):
                            full_content = ""
                            all_annotations = []
                            
                            for content_item in event.data.content:
                                if hasattr(content_item, 'text'):
                                    full_content += content_item.text.value
                                    if hasattr(content_item.text, 'annotations'):
                                        all_annotations.extend(content_item.text.annotations)
                            
                            # Process annotations
                            response_text = self.process_message_annotations(full_content, all_annotations)
                            
                            # If the processed text is different from what we streamed, show the corrected version
                            if response_text != current_message:
                                print(f"\n\n[Processed with citations: {response_text}]")
                    
                    elif event.event == "thread.run.completed":
                        logger.info("Streaming run completed")
                        break
                    
                    elif event.event == "thread.run.failed":
                        logger.error(f"Streaming run failed: {event.data.last_error}")
                        return "Sorry, I encountered an error processing your question."
                
                return response_text if response_text else current_message
                
        except Exception as e:
            logger.error(f"Failed to handle streaming response: {e}")
            return f"Sorry, I encountered an error with streaming: {str(e)}"
    
    def _handle_non_streaming_response(self) -> str:
        """
        Handle non-streaming response from the assistant (original behavior).
        
        Returns:
            Complete response text with citations
        """
        try:
            # Create and execute run
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            )
            
            # Wait for run completion
            logger.info("Processing your question...")
            max_wait_time = 120  # 2 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run.id
                )
                
                if run_status.status == "completed":
                    break
                elif run_status.status == "failed":
                    logger.error(f"Run failed: {run_status.last_error}")
                    return "Sorry, I encountered an error processing your question."
                elif run_status.status in ["cancelled", "expired"]:
                    logger.error(f"Run {run_status.status}")
                    return f"Sorry, the request was {run_status.status}."
                
                time.sleep(2)
            else:
                logger.error("Run timeout")
                return "Sorry, the request timed out. Please try again."
            
            # Get the assistant's response
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread_id,
                order="desc",
                limit=1
            )
            
            if not messages.data:
                return "Sorry, I couldn't generate a response."
            
            # Process the response and annotations
            message = messages.data[0]
            response_content = ""
            
            for content_item in message.content:
                if hasattr(content_item, 'text'):
                    text_content = content_item.text.value
                    annotations = content_item.text.annotations
                    
                    # Process annotations to create readable citations
                    processed_text = self.process_message_annotations(text_content, annotations)
                    response_content += processed_text
            
            return response_content
            
        except Exception as e:
            logger.error(f"Failed to handle non-streaming response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def toggle_streaming(self):
        """Toggle streaming mode on/off."""
        self.streaming = not self.streaming
        logger.info(f"Streaming mode: {'enabled' if self.streaming else 'disabled'}")
        return self.streaming
    
    def start_chat_session(self):
        """
        Start an interactive chat session in the terminal.
        """
        streaming_indicator = "📡 STREAMING" if self.streaming else "💬 STANDARD"
        print("\n" + "="*80)
        print(f"🤖 Azure OpenAI PDF Chatbot ({streaming_indicator})")
        print("="*80)
        print("Ask me anything about the uploaded PDF documents!")
        print("Type 'quit', 'exit', or 'bye' to end the session.")
        print("Type 'help' for available commands.")
        print("="*80 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thanks for using the PDF Chatbot.")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("  - Ask any question about the PDF content")
                    print("  - 'streaming' - Toggle streaming mode on/off")
                    print("  - 'status' - Show current streaming mode")
                    print("  - 'quit', 'exit', 'bye' - End the session")
                    print("  - 'help' - Show this help message")
                    continue
                
                elif user_input.lower() == 'streaming':
                    is_streaming = self.toggle_streaming()
                    mode = "enabled" if is_streaming else "disabled"
                    print(f"\n🔄 Streaming mode {mode}")
                    continue
                
                elif user_input.lower() == 'status':
                    mode = "enabled" if self.streaming else "disabled"
                    print(f"\n📊 Current mode: Streaming {mode}")
                    continue
                
                elif not user_input:
                    print("Please enter a question or command.")
                    continue
                
                # Get response from assistant
                if self.streaming:
                    print("\n🤖 Assistant: ", end="", flush=True)
                    response = self.send_message_and_get_response(user_input)
                    # Response is already printed via streaming, just add newline
                    print()
                else:
                    print("\n🤖 Assistant: ", end="", flush=True)
                    response = self.send_message_and_get_response(user_input)
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat session: {e}")
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")
    
    def cleanup(self):
        """
        Clean up resources (files, assistant, etc.).
        """
        try:
            # Note: In a production environment, you might want to keep the assistant
            # and vector store for reuse. For this demo, we'll clean up everything.
            
            if self.assistant_id:
                logger.info(f"Cleaning up assistant: {self.assistant_id}")
                try:
                    self.client.beta.assistants.delete(self.assistant_id)
                except Exception as e:
                    logger.warning(f"Failed to delete assistant: {e}")
            
            if self.vector_store_id:
                logger.info(f"Cleaning up vector store: {self.vector_store_id}")
                try:
                    self.client.beta.vector_stores.delete(self.vector_store_id)
                except Exception as e:
                    logger.warning(f"Failed to delete vector store: {e}")
            
            # Clean up uploaded files
            for file_id in self.uploaded_files:
                try:
                    self.client.files.delete(file_id)
                    logger.info(f"Deleted file: {file_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_id}: {e}")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def run(self, pdf_paths: List[str]):
        """
        Main method to run the chatbot with the specified PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files to analyze
        """
        try:
            # Upload PDF files
            print("📤 Uploading PDF files...")
            file_ids = self.upload_pdf_files(pdf_paths)
            
            # Create vector store
            print("🔍 Creating vector store for file search...")
            self.create_vector_store(file_ids)
            
            # Create assistant
            print("🤖 Creating AI assistant...")
            self.create_assistant()
            
            # Create thread
            print("💬 Setting up chat session...")
            self.create_thread()
            
            print("✅ Setup complete! Ready to chat.")
            
            # Start chat session
            self.start_chat_session()
            
        except Exception as e:
            logger.error(f"Error running chatbot: {e}")
            print(f"\n❌ Error: {e}")
            
        finally:
            # Always cleanup resources
            print("\n🧹 Cleaning up resources...")
            self.cleanup()


def main():
    """
    Main function to run the PDF chatbot.
    """
    # Example PDF file path (you can modify this or make it configurable)
    default_pdf_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf"

    default_pdf_path_2 = "/Users/bingran_you/Downloads/2504.07923v1.pdf"

    # You can add more PDF files here
    pdf_files = [default_pdf_path, default_pdf_path_2]
    
    # Validate that at least one PDF file exists
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ Error: No PDF files found!")
        print(f"Please ensure the following files exist:")
        for f in pdf_files:
            print(f"  - {f}")
        print("\nYou can modify the pdf_files list in the main() function to add your own PDF files.")
        return
    
    print(f"📚 Found {len(existing_files)} PDF file(s) to analyze:")
    for f in existing_files:
        print(f"  - {Path(f).name}")
    
    # Ask user for streaming preference
    while True:
        streaming_choice = input("\n🔄 Enable streaming responses? (y/n) [default: n]: ").strip().lower()
        if streaming_choice in ['', 'n', 'no']:
            streaming = False
            break
        elif streaming_choice in ['y', 'yes']:
            streaming = True
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    # Create and run the chatbot
    chatbot = AzureOpenAIPDFChatbot(streaming=streaming)
    chatbot.run(existing_files)


if __name__ == "__main__":
    main()
