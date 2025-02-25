import os
import json
import sys
from typing import Dict, List, Set

def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string using a simple word-based approach.

    Parameters:
        text (str): The input text to count tokens for

    Returns:
        int: Estimated number of tokens
    """
    # This is a simple estimation - in practice you might want to use 
    # a proper tokenizer from transformers or tiktoken
    return len(text.split())

def get_context_window(lines: List[str], target_line_idx: int, max_tokens: int) -> List[str]:
    """
    Get surrounding context around a target line within token limit.

    Parameters:
        lines (List[str]): All lines from the document
        target_line_idx (int): Index of the target line
        max_tokens (int): Maximum number of tokens for the context window

    Returns:
        List[str]: Context lines within token budget
    """
    context: List[str] = []
    current_tokens = 0

    # Add the target line first
    target_line = lines[target_line_idx]
    context.append(target_line)
    current_tokens += count_tokens(target_line)

    # Expand context in both directions
    left_idx = target_line_idx - 1
    right_idx = target_line_idx + 1

    while current_tokens < max_tokens and (left_idx >= 0 or right_idx < len(lines)):
        # Try to add line from left
        if left_idx >= 0:
            left_line = lines[left_idx]
            left_tokens = count_tokens(left_line)
            if current_tokens + left_tokens <= max_tokens:
                context.insert(0, left_line)
                current_tokens += left_tokens
            left_idx -= 1

        # Try to add line from right
        if right_idx < len(lines):
            right_line = lines[right_idx]
            right_tokens = count_tokens(right_line)
            if current_tokens + right_tokens <= max_tokens:
                context.append(right_line)
                current_tokens += right_tokens
            right_idx += 1

    return context

def extract_image_context(folder_dir: str, context_tokens: int = 1000) -> None:
    """
    Scan the given folder for image files and a markdown file.
    For each image file, search the markdown file for lines that mention the image filename,
    including surrounding context within a token limit.

    Parameters:
        folder_dir (str): The path to the folder containing image files and one markdown file
        context_tokens (int): Maximum number of tokens to include in context window around each mention

    Output:
        A JSON file named 'image_context.json' is written to the folder.
        The JSON contains a dictionary mapping image filenames to a list of context strings.
    """
    # Define the image file extensions we care about
    image_extensions: Set[str] = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg'}

    # List all image files in the folder (case-insensitive match)
    image_files = [f for f in os.listdir(folder_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    if not image_files:
        print("No image files found in the folder.")
        return

    # Find the markdown file (assuming there's only one .md file)
    md_files = [f for f in os.listdir(folder_dir) if f.lower().endswith('.md')]
    if not md_files:
        print("No markdown file found in the folder.")
        return

    md_file = md_files[0]
    md_path = os.path.join(folder_dir, md_file)

    # Read the content of the markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        md_lines = f.read().splitlines()

    # Create a dictionary to store image filename vs. list of context windows
    image_context: Dict[str, List[str]] = {}

    for image in image_files:
        # Find all lines in the markdown file that mention the image filename
        contexts = []
        for idx, line in enumerate(md_lines):
            if image in line:
                # Get context window around this mention
                context_window = get_context_window(md_lines, idx, context_tokens)
                contexts.append("\n".join(context_window))

        if contexts:
            image_context[image] = contexts

    # Write the dictionary to a JSON file in the same folder
    output_path = os.path.join(folder_dir, 'image_context.json')
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_context, outfile, indent=2, ensure_ascii=False)

    print(f"Image context data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage with custom context window size
    folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/markdown/"
    file_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/16005aaa19145334b5605c6bf61661a0.pdf"
    extract_image_context(folder_dir, file_path, context_tokens=1000)