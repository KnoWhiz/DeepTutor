import os
import json
import sys
from typing import Dict, List, Set
from pathlib import Path

# Add the project root to Python path for direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Try importing as a module first
    from pipeline.helper.azure_blob import AzureBlobHelper
except ImportError:
    # If that fails, try importing directly
    from helper.azure_blob import AzureBlobHelper

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

def initialize_image_files(folder_dir: str | Path) -> tuple[str, str]:
    """
    Initialize both image context and image URLs JSON files if they don't exist.
    
    Parameters:
        folder_dir (str | Path): Directory where the image files should be created
        
    Returns:
        tuple[str, str]: Paths to the image context and image URLs files
    """
    folder_dir = Path(folder_dir)
    image_context_path = folder_dir / "image_context.json"
    image_urls_path = folder_dir / "image_urls.json"
    
    # Create directory if it doesn't exist
    os.makedirs(folder_dir, exist_ok=True)
    
    # Create empty image context file if it doesn't exist
    if not image_context_path.exists():
        print("Initializing empty image context file...")
        with open(image_context_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    
    # Create empty image URLs file if it doesn't exist
    if not image_urls_path.exists():
        print("Initializing empty image URLs file...")
        with open(image_urls_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    
    return str(image_context_path), str(image_urls_path)

def upload_images_to_azure(folder_dir: str | Path) -> None:
    """
    Upload images from the given folder to Azure Blob storage and create a mapping file.
    The images will be stored in the format '/course_id/images/...'
    If an image already exists in Azure storage, it will be skipped.
    
    Parameters:
        folder_dir (str | Path): The path to the folder containing image files
    """
    # Convert to Path object if it's a string
    folder_path = Path(folder_dir) if isinstance(folder_dir, str) else folder_dir
    
    # Initialize both JSON files
    _, output_path = initialize_image_files(folder_path)
    
    # Define the image file extensions we care about
    image_extensions: Set[str] = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    
    # Extract course_id from the folder path
    course_id = folder_path.parts[-2]
    
    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in image_extensions]
    if not image_files:
        print("No image files found in the folder.")
        return
        
    # Initialize Azure Blob helper
    azure_blob = AzureBlobHelper()
    container_name = "knowhiztutorrag"
    
    # Load existing image URLs
    with open(output_path, 'r', encoding='utf-8') as infile:
        image_urls = json.load(infile)
        print("Loaded existing image_urls.json")
    
    # Upload each image and store its URL if not already uploaded
    for image_file in image_files:
        # Skip if image is already in our mapping
        if image_file in image_urls:
            print(f"Skipping {image_file} - already uploaded")
            continue
            
        local_path = folder_path / image_file
        blob_name = f"file_appendix/{course_id}/images/{image_file}"
        
        try:
            # Check if blob already exists
            blob_client = azure_blob.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            if blob_client.exists():
                url = f"https://{azure_blob.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
                image_urls[image_file] = url
                print(f"Skipping {image_file} - already exists in Azure storage")
                continue
                
            # Upload only if doesn't exist
            azure_blob.upload(str(local_path), blob_name, container_name)
            url = f"https://{azure_blob.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
            image_urls[image_file] = url
            print(f"Uploaded {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    # Write the updated URL mapping to JSON file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_urls, outfile, indent=2, ensure_ascii=False)
    
    print(f"Image URLs mapping saved to: {output_path}")

def extract_image_context(folder_dir: str | Path, context_tokens: int = 500) -> None:
    """
    Extract context for each image in a folder and save to JSON.
    
    Parameters:
        folder_dir (str | Path): Directory containing the images
        context_tokens (int): Maximum number of tokens for context per image
    """
    folder_dir = Path(folder_dir)
    
    # Initialize both JSON files
    image_context_path, _ = initialize_image_files(folder_dir)
    
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
    md_path = folder_dir / md_file
    
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
    with open(image_context_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_context, outfile, indent=2, ensure_ascii=False)

    # Save images to Azure Blob Storage
    upload_images_to_azure(folder_dir)
    
    print(f"Image context data saved to: {image_context_path}")

if __name__ == "__main__":
    # folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/markdown/"
    folder_dir = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/c8773c4a9a62ca3bafd2010d3d0093f5/markdown"
    extract_image_context(folder_dir)
    # upload_images_to_azure(folder_dir)