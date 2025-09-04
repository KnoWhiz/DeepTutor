"""
Gemini CLI Handler for streaming chatbot responses.

This module provides functionality to interact with the Gemini CLI
and generate streaming responses for a chatbot interface.
"""

import os
import subprocess
import json
import shutil
from typing import Generator, Optional
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiCLIHandler:
    """Handler for Gemini CLI interactions with streaming capabilities."""
    
    def __init__(self, working_dir: str = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files"):
        """
        Initialize the Gemini CLI handler.
        
        Args:
            working_dir: Directory where files will be processed
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Setup environment variables for Gemini CLI."""
        # Copy GEMINI_API_KEY from the integration test directory .env file
        env_file = Path("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/Gemini_CLI_integration_test/.env")
        
        if env_file.exists():
            # Read the .env file and set the environment variable
            with open(env_file, "r") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        if key == "GEMINI_API_KEY":
                            os.environ["GEMINI_API_KEY"] = value.strip('"\'')
                            logger.info("GEMINI_API_KEY loaded from .env file")
                            break
        else:
            logger.warning("No .env file found, assuming GEMINI_API_KEY is already set")
    
    def _construct_gemini_prompt(self, file_path: str, query: str) -> str:
        """
        Construct the prompt for Gemini CLI based on the provided example.
        
        Args:
            file_path: Path to the file to be processed
            query: User's query
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""read_file(path="{file_path}")
Act as a domain expert research assistant.
User question: {query}

Using the loaded document as a primary source, do a deep research workflow:
1) Extract the document's main topics, key points, methods, results, limitations, and important information.
2) From these, craft 3–5 precise queries and call google_web_search to find ≥8 recent, high-quality sources (peer-reviewed or official) since 2024.
3) Select the best 5 diverse sources and call web_fetch to pull details; compare against the document: methods, results, limitations, reproducibility; flag contradictions.
4) Synthesize a markdown brief that answers the question, clearly labeling what comes from the document vs. external sources, with inline numeric citations and a short bibliography; include a comparison table."""
        
        return prompt
    
    def process_file_with_query(self, file_dir: str, query: str) -> Generator[str, None, None]:
        """
        Process a file with a query using Gemini CLI and yield streaming chunks.
        
        Args:
            file_dir: Directory containing the file or path to the specific file
            query: User's query string
            
        Yields:
            String chunks from the Gemini CLI response
        """
        try:
            # Determine the file path
            file_path = Path(file_dir)
            
            if file_path.is_dir():
                # Find the first text or PDF file in the directory
                for ext in ["*.txt", "*.pdf"]:
                    files = list(file_path.glob(ext))
                    if files:
                        file_path = files[0]
                        break
                else:
                    yield f"Error: No .txt or .pdf files found in directory {file_dir}"
                    return
            
            if not file_path.exists():
                yield f"Error: File {file_path} does not exist"
                return
            
            # Copy file to working directory if not already there
            if file_path.parent != self.working_dir:
                working_file_path = self.working_dir / file_path.name
                shutil.copy2(file_path, working_file_path)
                file_path = working_file_path
            
            # Construct the prompt
            prompt = self._construct_gemini_prompt(str(file_path), query)
            
            # Prepare the Gemini CLI command
            cmd = ["gemini", "--yolo", "-p", prompt]
            
            logger.info(f"Executing Gemini CLI command for file: {file_path}")
            logger.info(f"Query: {query}")
            
            # Execute the command with streaming output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.working_dir)
            )
            
            # Stream the output
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    yield output.rstrip("\n")
            
            # Check for any errors
            stderr_output = process.stderr.read()
            if stderr_output and process.returncode != 0:
                yield f"\nError from Gemini CLI: {stderr_output}"
                
        except Exception as e:
            logger.error(f"Error processing file with query: {str(e)}")
            yield f"Error: {str(e)}"


def main_streaming_function(file_dir: str, query: str) -> Generator[str, None, None]:
    """
    Main function that takes file directory and query, returns streaming generator.
    
    Args:
        file_dir: Directory containing the file or path to specific file
        query: User's query string
        
    Yields:
        String chunks from the Gemini response
    """
    handler = GeminiCLIHandler()
    yield from handler.process_file_with_query(file_dir, query)


if __name__ == "__main__":
    # Test the functionality
    test_file_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files"
    test_query = "What are the main topics discussed in this document?"
    
    print("Testing Gemini CLI Handler...")
    for chunk in main_streaming_function(test_file_dir, test_query):
        print(chunk) 