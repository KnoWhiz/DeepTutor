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

def get_context_window(lines: List[str], target_line_idx: int) -> List[str]:
    """
    Get only the second line following a target line.
    
    Parameters:
        lines (List[str]): All lines from the document
        target_line_idx (int): Index of the target line
        
    Returns:
        List[str]: Context line (only the second line after target, if it exists)
    """
    context: List[str] = []
    
    # Look for the second line after target
    next_idx = target_line_idx + 2
    if next_idx < len(lines):
        next_line = lines[next_idx].strip()
        if next_line:  # Only add non-empty line
            context.append(next_line)
    
    return context

def extract_image_context(folder_dir: str, context_tokens: int = 500) -> None:
    """
    Scan the given folder for image files and a markdown file.
    For each image file, search the markdown file for lines that mention the image filename,
    including only the second line after each mention.
    
    Parameters:
        folder_dir (str): The path to the folder containing image files and one markdown file
    
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
                # Get only the second line after this mention
                context_window = get_context_window(md_lines, idx)
                if context_window:  # Only add if we found a valid context line
                    contexts.append(context_window[0])
        
        if contexts:
            image_context[image] = contexts
    
    # Write the dictionary to a JSON file in the same folder
    output_path = os.path.join(folder_dir, 'image_context.json')
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_context, outfile, indent=2, ensure_ascii=False)
    
    print(f"Image context data saved to: {output_path}")

if __name__ == "__main__":
    # folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/markdown/"
    folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/c8773c4a9a62ca3bafd2010d3d0093f5/markdown"
    extract_image_context(folder_dir) 