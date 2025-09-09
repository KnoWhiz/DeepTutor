"""
Gemini CLI Streaming Wrapper

A Python module that wraps the locally installed Gemini CLI to stream output incrementally.
Provides cross-platform support for streaming both stdout and stderr from the Gemini CLI.

## Installation & Setup

1. Ensure you have the Gemini CLI installed and accessible in your PATH
2. Verify with: `gemini --help` or `gemini --version`
3. Import and use this module in your Python code

## Usage Example

```python
from gemini_stream import stream_gemini

# Basic usage
for chunk in stream_gemini("What is quantum computing?"):
    print(chunk, end="", flush=True)

# Advanced usage with custom model and args
for chunk in stream_gemini(
    "Explain machine learning", 
    model="gemini-2.0-flash",
    extra_args=["--temperature", "0.7"],
    timeout=30.0
):
    print(chunk, end="", flush=True)
```

## Error Handling

```python
from gemini_stream import stream_gemini, GeminiCLIError

try:
    for chunk in stream_gemini("Your prompt here"):
        print(chunk, end="", flush=True)
except GeminiCLIError as e:
    print(f"CLI Error (exit code {e.exit_code}): {e.stderr_tail}")
except TimeoutError:
    print("Request timed out")
```
"""

import subprocess
import threading
import time
import os
from typing import Iterator, Optional, Dict, List
from queue import Queue, Empty
import sys


class GeminiCLIError(Exception):
    """
    Exception raised when the Gemini CLI exits with a non-zero code.
    
    Attributes:
        exit_code: The CLI exit code
        stderr_tail: Last N lines of stderr for debugging
    """
    
    def __init__(self, exit_code: int, stderr_tail: str) -> None:
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        super().__init__(f"Gemini CLI failed with exit code {exit_code}")


def stream_gemini(
    prompt: str,
    model: str = "gemini-2.5-pro",
    extra_args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None
) -> Iterator[str]:
    """
    Stream output from the Gemini CLI incrementally.
    
    Args:
        prompt: The prompt to send to Gemini
        model: The Gemini model to use (default: "gemini-2.5-pro")
        extra_args: Additional CLI arguments to append
        cwd: Working directory for the CLI process
        env: Environment variables for the CLI process
        timeout: Maximum time to wait for completion (seconds)
        
    Yields:
        str: Text chunks from stdout, or "[stderr] ..." prefixed stderr chunks
        
    Raises:
        GeminiCLIError: If CLI exits with non-zero code
        TimeoutError: If timeout is exceeded
        FileNotFoundError: If gemini CLI is not found in PATH
    """
    # Build the command
    cmd = ["gemini", "--prompt", prompt, "--model", model]
    if extra_args:
        cmd.extend(extra_args)
    
    # Prepare environment
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    
    # Track stderr for error reporting
    stderr_lines: List[str] = []
    
    # Queues for thread-safe communication
    stdout_queue: Queue[Optional[str]] = Queue()
    stderr_queue: Queue[Optional[str]] = Queue()
    
    def read_stream(stream, queue: Queue[Optional[str]], prefix: str = "") -> None:
        """Read from a stream and put chunks into a queue."""
        try:
            while True:
                # Read in small chunks to enable streaming
                chunk = stream.read(1024)
                if not chunk:
                    break
                
                # Decode with error replacement to handle partial UTF-8 bytes
                try:
                    text = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    text = chunk.decode("utf-8", errors="replace")
                
                if prefix:
                    # For stderr, prefix each line
                    lines = text.splitlines(keepends=True)
                    for line in lines:
                        if line.strip():  # Only prefix non-empty lines
                            queue.put(f"{prefix}{line}")
                        else:
                            queue.put(line)
                else:
                    queue.put(text)
        except Exception:
            # Stream closed or other error
            pass
        finally:
            queue.put(None)  # Signal end of stream
    
    def collect_stderr(stream) -> None:
        """Collect stderr lines for error reporting."""
        nonlocal stderr_lines
        try:
            while True:
                chunk = stream.read(1024)
                if not chunk:
                    break
                
                try:
                    text = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    text = chunk.decode("utf-8", errors="replace")
                
                # Store stderr lines for error reporting
                lines = text.splitlines()
                stderr_lines.extend(lines)
                
                # Also put in queue for streaming
                if text.strip():
                    stderr_queue.put(f"[stderr] {text}")
        except Exception:
            pass
        finally:
            stderr_queue.put(None)
    
    # Start the process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=process_env,
            bufsize=0,  # Unbuffered for real-time streaming
            text=False  # We'll handle encoding ourselves
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Gemini CLI not found in PATH. Please ensure it's installed and accessible."
        )
    
    # Start reader threads
    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_queue),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=collect_stderr,
        args=(process.stderr,),
        daemon=True
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Track timing for timeout
    start_time = time.time()
    stdout_done = False
    stderr_done = False
    
    try:
        while not (stdout_done and stderr_done):
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError(f"Gemini CLI timed out after {timeout} seconds")
            
            # Check for generator cancellation
            try:
                # This allows the generator to be cancelled via GeneratorExit
                yield ""  # Empty yield to check for cancellation
            except GeneratorExit:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
            
            # Process stdout
            if not stdout_done:
                try:
                    chunk = stdout_queue.get(timeout=0.1)
                    if chunk is None:
                        stdout_done = True
                    elif chunk:  # Skip empty chunks from cancellation check
                        yield chunk
                except Empty:
                    pass
            
            # Process stderr
            if not stderr_done:
                try:
                    chunk = stderr_queue.get(timeout=0.1)
                    if chunk is None:
                        stderr_done = True
                    elif chunk:
                        yield chunk
                except Empty:
                    pass
        
        # Wait for process completion
        exit_code = process.wait(timeout=5.0 if not timeout else min(5.0, timeout))
        
        # Check exit code
        if exit_code != 0:
            # Get last 10 lines of stderr for error context
            stderr_tail = "\n".join(stderr_lines[-10:]) if stderr_lines else "No stderr output"
            raise GeminiCLIError(exit_code, stderr_tail)
            
    except subprocess.TimeoutExpired:
        process.kill()
        raise TimeoutError("Process cleanup timed out")
    
    finally:
        # Ensure process is cleaned up
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()


def _demo_with_echo() -> None:
    """
    Demo function using echo command for testing the streaming mechanism.
    This is useful for testing when Gemini CLI is not available.
    """
    print("=== Demo with echo command ===")
    
    # Test basic streaming with echo
    test_text = "This is a test of the streaming functionality.\nIt should work line by line.\nAnd handle multiple lines correctly."
    
    if sys.platform == "win32":
        cmd_args = ["-n", test_text]  # Windows echo
    else:
        cmd_args = ["-e", test_text]  # Unix echo
    
    print("Streaming echo output:")
    try:
        # We'll modify the function temporarily for demo purposes
        import subprocess
        import threading
        from queue import Queue, Empty
        
        process = subprocess.Popen(
            ["echo"] + cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            text=False
        )
        
        stdout_queue: Queue[Optional[str]] = Queue()
        
        def read_stdout():
            try:
                while True:
                    chunk = process.stdout.read(10)  # Small chunks for demo
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    stdout_queue.put(text)
                    time.sleep(0.1)  # Simulate slow streaming
            except Exception:
                pass
            finally:
                stdout_queue.put(None)
        
        thread = threading.Thread(target=read_stdout, daemon=True)
        thread.start()
        
        while True:
            try:
                chunk = stdout_queue.get(timeout=0.5)
                if chunk is None:
                    break
                print(chunk, end="", flush=True)
            except Empty:
                if process.poll() is not None:
                    break
        
        exit_code = process.wait()
        print(f"\nProcess completed with exit code: {exit_code}")
        
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    """
    Example usage and testing of the gemini_stream module.
    """
    print("Gemini CLI Streaming Wrapper")
    print("=" * 40)
    
    # Check if we should run the echo demo
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        _demo_with_echo()
        sys.exit(0)
    
    # Example with actual Gemini CLI
    test_prompt = "What is the capital of France? Please provide a brief answer."
    
    print(f"Testing with prompt: '{test_prompt}'")
    print("-" * 40)
    
    try:
        # Test basic streaming
        print("Streaming Gemini response:")
        for chunk in stream_gemini(test_prompt, timeout=30.0):
            if chunk.strip():  # Skip empty chunks from cancellation checks
                print(chunk, end="", flush=True)
        
        print("\n" + "=" * 40)
        print("Test completed successfully!")
        
    except GeminiCLIError as e:
        print(f"\nGemini CLI Error (exit code {e.exit_code}):")
        print(e.stderr_tail)
        
    except TimeoutError:
        print("\nRequest timed out!")
        
    except FileNotFoundError:
        print("\nGemini CLI not found in PATH.")
        print("Please install the Gemini CLI and ensure it's accessible.")
        print("\nRunning echo demo instead...")
        print("-" * 40)
        _demo_with_echo()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
