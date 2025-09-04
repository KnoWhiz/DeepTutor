"""
Gemini CLI Agent - Python Integration

This module provides a Python interface to the Gemini CLI, allowing users to send
commands and receive streaming responses. Based on the Gemini-CLI-UI-main implementation.
"""

import os
import subprocess
import json
import time
import tempfile
import shutil
from typing import Generator, Dict, Any, Optional, List
from pathlib import Path
import threading
import queue
import re
from dataclasses import dataclass


@dataclass
class GeminiResponse:
    """Data class for Gemini CLI responses"""
    content: str
    is_partial: bool = False
    error: Optional[str] = None
    exit_code: Optional[int] = None
    session_id: Optional[str] = None


class GeminiResponseBuffer:
    """
    Intelligent response buffering similar to the JavaScript implementation.
    Handles streaming output and provides proper message chunking.
    """
    
    def __init__(self, min_buffer_size: int = 30, partial_delay: float = 0.3, max_wait_time: float = 1.5):
        self.buffer = ""
        self.last_sent_time = time.time()
        self.min_buffer_size = min_buffer_size
        self.partial_delay = partial_delay
        self.max_wait_time = max_wait_time
        self.in_code_block = False
        
        # Patterns to detect message completion
        self.completion_patterns = [
            re.compile(r'\.\s*$'),      # Ends with period
            re.compile(r'\?\s*$'),      # Ends with question mark
            re.compile(r'!\s*$'),       # Ends with exclamation
            re.compile(r'```\s*$'),     # Ends with code block
            re.compile(r':\s*$'),       # Ends with colon
            re.compile(r'\n\n$'),       # Double line break
        ]
    
    def add_data(self, data: str) -> Optional[str]:
        """Add data to buffer and return content if ready to send"""
        self.buffer += data
        self._update_code_block_state()
        
        if self._should_send_immediately():
            return self._flush()
        return None
    
    def _update_code_block_state(self):
        """Track if we're inside a code block"""
        code_block_count = self.buffer.count('```')
        self.in_code_block = (code_block_count % 2) != 0
    
    def _should_send_immediately(self) -> bool:
        """Determine if buffer should be sent immediately"""
        # Don't send tiny fragments
        if len(self.buffer) < self.min_buffer_size:
            return False
        
        # Never split in the middle of a code block
        if self.in_code_block:
            return False
        
        # Check for completion patterns
        trimmed_buffer = self.buffer.strip()
        for pattern in self.completion_patterns:
            if pattern.search(trimmed_buffer):
                return True
        
        # Check if enough time has passed
        time_since_last_send = time.time() - self.last_sent_time
        if time_since_last_send > self.max_wait_time and len(self.buffer) > 0:
            return True
        
        return False
    
    def _flush(self) -> str:
        """Flush buffer and return content"""
        if not self.buffer:
            return ""
        
        content = self._fix_formatting(self.buffer.strip())
        self.buffer = ""
        self.last_sent_time = time.time()
        return content
    
    def _fix_formatting(self, content: str) -> str:
        """Fix common formatting issues"""
        # Remove excessive line breaks
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        
        # Fix list formatting
        content = re.sub(r'(\d+\.\s+[^\n]+)\n\n+(\d+\.)', r'\1\n\2', content)
        content = re.sub(r'([-*]\s+[^\n]+)\n\n+([-*])', r'\1\n\2', content)
        
        # Ensure proper spacing around headers
        content = re.sub(r'\n{3,}(#{1,6}\s)', r'\n\n\1', content)
        content = re.sub(r'(#{1,6}\s[^\n]+)\n{3,}', r'\1\n\n', content)
        
        return content.strip()
    
    def force_flush(self) -> str:
        """Force flush any remaining content"""
        return self._flush()


def gemini_cli_agent(
    input_folder_dir: str,
    query: str,
    session_id: Optional[str] = None,
    model: str = "gemini-2.5-pro",
    gemini_path: Optional[str] = None,
    skip_permissions: bool = False,
    debug: bool = False,
    images: Optional[List[str]] = None,
    timeout: float = 30.0
) -> Generator[GeminiResponse, None, None]:
    """
    Send a command to Gemini CLI and receive streaming responses.
    
    Args:
        input_folder_dir: Directory path where Gemini CLI should operate
        query: Command/query to send to Gemini CLI
        session_id: Optional session ID for continuing conversations
        model: Gemini model to use (default: "gemini-2.5-flash")
        gemini_path: Path to gemini CLI executable (uses PATH if None)
        skip_permissions: Skip permission prompts (--yolo flag)
        debug: Enable debug output
        images: List of image file paths to include
        timeout: Timeout in seconds for CLI response
    
    Yields:
        GeminiResponse: Streaming responses from Gemini CLI
        
    Raises:
        FileNotFoundError: If gemini CLI is not found
        ValueError: If input parameters are invalid
        RuntimeError: If CLI execution fails
    """
    
    # Validate inputs
    if not input_folder_dir or not os.path.exists(input_folder_dir):
        raise ValueError(f"Input folder directory does not exist: {input_folder_dir}")
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Get gemini CLI path
    cli_path = gemini_path or os.environ.get('GEMINI_PATH', 'gemini')
    
    # Check if gemini CLI is available
    try:
        subprocess.run([cli_path, '--version'], capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        raise FileNotFoundError(
            f"Gemini CLI not found at '{cli_path}'. "
            "Please install it with: npm install -g @google/generative-ai-cli"
        )
    
    # Build command arguments
    args = [cli_path]
    
    # Add prompt with query
    args.extend(['--prompt', query])
    
    # Add model
    args.extend(['--model', model])
    
    # Add flags
    if skip_permissions:
        args.append('--yolo')
    
    if debug:
        args.append('--debug')
    
    # Handle images if provided
    temp_image_paths = []
    temp_dir = None
    
    if images:
        try:
            # Create temporary directory for images
            temp_dir = tempfile.mkdtemp(prefix='gemini_cli_images_')
            
            for i, image_path in enumerate(images):
                if not os.path.exists(image_path):
                    continue
                
                # Copy image to temp directory
                image_name = f"image_{i}_{os.path.basename(image_path)}"
                temp_image_path = os.path.join(temp_dir, image_name)
                shutil.copy2(image_path, temp_image_path)
                temp_image_paths.append(temp_image_path)
            
            # Add image note to query if we have images
            if temp_image_paths:
                image_note = f"\n\n[Images attached: {len(temp_image_paths)} images at the following paths:]\n"
                image_note += "\n".join(f"{i+1}. {path}" for i, path in enumerate(temp_image_paths))
                
                # Update the prompt argument
                prompt_index = args.index('--prompt')
                if prompt_index != -1:
                    args[prompt_index + 1] = args[prompt_index + 1] + image_note
                    
        except Exception as e:
            # Clean up on error
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Error processing images: {e}")
    
    # Create response buffer
    response_buffer = GeminiResponseBuffer()
    full_response = ""
    generated_session_id = session_id or f"gemini_{int(time.time())}"
    
    try:
        # Start Gemini CLI process
        process = subprocess.Popen(
            args,
            cwd=input_folder_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Close stdin to signal we're done sending input
        if process.stdin:
            process.stdin.close()
        
        # Create output queue for thread communication
        output_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def read_stdout():
            """Read stdout in a separate thread"""
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        output_queue.put(('stdout', line))
                output_queue.put(('stdout', None))  # Signal end
            except Exception as e:
                error_queue.put(f"Error reading stdout: {e}")
        
        def read_stderr():
            """Read stderr in a separate thread"""
            try:
                for line in iter(process.stderr.readline, ''):
                    if line:
                        output_queue.put(('stderr', line))
                output_queue.put(('stderr', None))  # Signal end
            except Exception as e:
                error_queue.put(f"Error reading stderr: {e}")
        
        # Start reader threads
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Track if we've received any output
        has_received_output = False
        start_time = time.time()
        stdout_ended = False
        stderr_ended = False
        
        # Process output
        while not (stdout_ended and stderr_ended):
            try:
                # Check for timeout
                if time.time() - start_time > timeout and not has_received_output:
                    process.terminate()
                    yield GeminiResponse(
                        content="",
                        error="Gemini CLI timeout - no response received",
                        exit_code=-1
                    )
                    return
                
                # Check for errors from reader threads
                try:
                    error_msg = error_queue.get_nowait()
                    yield GeminiResponse(content="", error=error_msg, exit_code=-1)
                    return
                except queue.Empty:
                    pass
                
                # Get output from queue
                try:
                    stream_type, data = output_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if data is None:  # End signal
                    if stream_type == 'stdout':
                        stdout_ended = True
                    elif stream_type == 'stderr':
                        stderr_ended = True
                    continue
                
                has_received_output = True
                
                if stream_type == 'stdout':
                    # Filter out debug messages and system messages
                    if any(pattern in data for pattern in [
                        '[DEBUG]', 'Flushing log events', 'Clearcut response',
                        '[MemoryDiscovery]', '[BfsFileSearch]', 'Loaded cached credentials'
                    ]):
                        continue
                    
                    filtered_data = data.strip()
                    if filtered_data:
                        full_response += (('\n' if full_response else '') + filtered_data)
                        
                        # Add to buffer and check if we should send
                        buffered_content = response_buffer.add_data(filtered_data)
                        if buffered_content:
                            yield GeminiResponse(
                                content=buffered_content,
                                is_partial=True,
                                session_id=generated_session_id
                            )
                
                elif stream_type == 'stderr':
                    # Filter out deprecation warnings
                    if any(pattern in data for pattern in [
                        '[DEP0040]', 'DeprecationWarning', '--trace-deprecation',
                        'Loaded cached credentials'
                    ]):
                        continue
                    
                    error_msg = data.strip()
                    if error_msg:
                        yield GeminiResponse(
                            content="",
                            error=error_msg,
                            session_id=generated_session_id
                        )
            
            except KeyboardInterrupt:
                process.terminate()
                break
        
        # Wait for process to complete
        exit_code = process.wait()
        
        # Flush any remaining buffered content
        remaining_content = response_buffer.force_flush()
        if remaining_content:
            yield GeminiResponse(
                content=remaining_content,
                is_partial=False,
                session_id=generated_session_id
            )
        
        # Send completion response
        yield GeminiResponse(
            content="",
            is_partial=False,
            exit_code=exit_code,
            session_id=generated_session_id
        )
        
    except Exception as e:
        yield GeminiResponse(
            content="",
            error=f"Process execution error: {str(e)}",
            exit_code=-1,
            session_id=generated_session_id
        )
    
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# Example usage and testing function
def test_gemini_cli_agent():
    """Test function to demonstrate usage"""
    
    # Example usage
    folder_dir = "/path/to/your/project"
    query = "Explain the main function in this codebase"
    
    try:
        for response in gemini_cli_agent(folder_dir, query):
            if response.error:
                print(f"ERROR: {response.error}")
            elif response.content:
                print(f"CONTENT: {response.content}")
            elif response.exit_code is not None:
                print(f"COMPLETED with exit code: {response.exit_code}")
                
    except Exception as e:
        print(f"EXCEPTION: {e}")


if __name__ == "__main__":
    # Run test if executed directly
    test_gemini_cli_agent() 