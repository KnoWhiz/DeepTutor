import os
import json
import sys
import openai
import requests
import base64
from typing import Dict, List, Set
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from pipeline.science.pipeline.utils import generate_file_id
from pipeline.science.pipeline.config import load_config
load_dotenv()

import logging
logger = logging.getLogger("tutorpipeline.science.images_understanding")


# Add the project root to Python path for direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Try importing as a module first
    from pipeline.science.pipeline.helper.azure_blob import AzureBlobHelper
except ImportError:
    # If that fails, try importing directly
    from pipeline.helper.azure_blob import AzureBlobHelper


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
        logger.info("Initializing empty image context file...")
        with open(image_context_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    # Create empty image URLs file if it doesn't exist
    if not image_urls_path.exists():
        logger.info("Initializing empty image URLs file...")
        with open(image_urls_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    return str(image_context_path), str(image_urls_path)


def upload_images_to_azure(folder_dir: str | Path, file_path) -> None:
    """
    Upload images from the given folder to Azure Blob storage and create a mapping file.
    The images will be stored in the format '/file_id/images/...'
    If an image already exists in Azure storage, it will be skipped.

    Parameters:
        folder_dir (str | Path): The path to the folder containing image files
    """
    # Convert to Path object if it's a string
    folder_path = Path(folder_dir) if isinstance(folder_dir, str) else folder_dir

    # Initialize both JSON files
    _, output_path = initialize_image_files(folder_path)

    # Define the image file extensions we care about
    config = load_config()
    image_extensions: Set[str] = set(config["image_extensions"])

    # Extract file_id from the folder path
    file_id = generate_file_id(file_path)

    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in image_extensions]
    if not image_files:
        logger.info("No image files found in the folder.")
        return

    # Initialize Azure Blob helper
    azure_blob = AzureBlobHelper()
    container_name = "knowhiztutorrag"

    # Load existing image URLs
    with open(output_path, 'r', encoding='utf-8') as infile:
        image_urls = json.load(infile)
        logger.info("Loaded existing image_urls.json")

    # Upload each image and store its URL if not already uploaded
    for image_file in image_files:
        # Skip if image is already in our mapping
        if image_file in image_urls:
            logger.info(f"Skipping {image_file} - already uploaded")
            continue

        local_path = folder_path / image_file
        blob_name = f"file_appendix/{file_id}/images/{image_file}"

        try:
            # Check if blob already exists
            blob_client = azure_blob.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            if blob_client.exists():
                url = f"https://{azure_blob.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
                image_urls[image_file] = url
                logger.info(f"Skipping {image_file} - already exists in Azure storage")
                continue

            # Upload only if doesn't exist
            azure_blob.upload(str(local_path), blob_name, container_name)
            url = f"https://{azure_blob.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
            image_urls[image_file] = url
            logger.info(f"Uploaded {image_file}")
        except Exception as e:
            logger.info(f"Error processing {image_file}: {e}")
            continue

    # Write the updated URL mapping to JSON file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_urls, outfile, indent=2, ensure_ascii=False)

    logger.info(f"Image URLs mapping saved to: {output_path}")


def upload_markdown_to_azure(folder_dir: str | Path, file_path: str) -> None:
    """
    Upload markdown file to Azure Blob storage.
    """
    folder_dir = Path(folder_dir)
    logger.info(f"Uploading folder_dir markdown to Azure Blob storage: {folder_dir}")
    file_id = generate_file_id(file_path)
    azure_blob = AzureBlobHelper()
    container_name = "knowhiztutorrag"
    # Upload all the md files in the folder
    md_files = [os.path.join(folder_dir, f"{file_id}.md")]
    logger.info(f"Uploading dir {md_files} markdown files to Azure Blob storage")
    for md_file in md_files:
        blob_name = f"file_appendix/{file_id}/images/{md_file}"
        local_path = md_file
        azure_blob.upload(str(local_path), blob_name, container_name)

    # TEST
    logger.info(f"Uploaded {(md_files)} markdown files to Azure Blob storage")


def extract_image_context(folder_dir: str | Path, file_path: str = "", context_tokens: int = 1000) -> None:
    """
    Extract context for each image in a folder and save to JSON.

    Parameters:
        folder_dir (str | Path): Directory containing the images
        context_tokens (int): Maximum number of tokens for context per image
    """
    folder_dir = Path(folder_dir)
    logger.info(f"Current markdown folder: {folder_dir}")

    # Initialize both JSON files
    image_context_path, _ = initialize_image_files(folder_dir)

    # Define the image file extensions we care about
    config = load_config()
    image_extensions: Set[str] = set(config["image_extensions"])

    # List all image files in the folder (case-insensitive match)
    image_files = [f for f in os.listdir(folder_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    if not image_files:
        logger.info("No image files found in the folder.")
        return

    # Find the markdown file (with file_id.md file name)
    file_id = generate_file_id(file_path)
    md_files = [os.path.join(folder_dir, f"{file_id}.md")]
    if not md_files:
        logger.info("No markdown file found in the folder.")
        return

    md_file = md_files[0]
    md_path = md_file

    # Read the content of the markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        md_lines = f.read().splitlines()

    # If there are images_files and md_files, re-order the list image_files to match the order that images show up in md_files
    image_order = []
    for line in md_lines:
        for image in image_files:
            if image in line and image not in image_order:
                image_order.append(image)
    # Add any images that weren't found in the markdown file to the end of the order list
    for image in image_files:
        if image not in image_order:
            image_order.append(image)
    # Replace image_files with the ordered list
    image_files = image_order

    # Create a dictionary to store image filename vs. list of context windows
    image_context: Dict[str, List[str]] = {}

    for i, image in enumerate(image_files):
        # Find all lines in the markdown file that mention the image filename
        contexts = []
        for idx, line in enumerate(md_lines):
            if image in line:
                # Get only the second line after this mention
                context_window = get_context_window(md_lines, idx)
                if context_window:  # Only add if we found a valid context line
                    contexts.append(f"This is Image #{i+1} / Fig.{i+1} / Figure {i+1}: \n" + context_window[0] + " <markdown>")

        if contexts:
            image_context[image] = contexts

    # Write the dictionary to a JSON file in the same folder
    with open(image_context_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_context, outfile, indent=2, ensure_ascii=False)

    # Process folder images
    process_folder_images(folder_dir)

    logger.info(f"Image context data saved to: {image_context_path}")


def analyze_image(image_url=None):
    """
    Analyze an image using Azure OpenAI's vision model.

    Args:
        image_url (str, optional): URL of the image to analyze. Defaults to a test image if none provided.

    Returns:
        str: The analysis result from the vision model
    """
    # Initialize Azure OpenAI client
    api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_KEY')
    deployment_name = 'gpt-4o'
    api_version = '2024-06-01'

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}openai/deployments/{deployment_name}"
    )

    try:
        # Create messages for the vision model
        messages = [
            {
                "role": "system",
                "content": "You are an professor helping students to understand images in a research paper. Please describe what you see in detail, and explain the research concept in an easy-to-understand manner."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this picture:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        # Generate response
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=2000
        )
        result = response.choices[0].message.content
        return result

    except Exception as e:
        logger.exception(f"Error occurred in analyze_image with Azure OpenAI: {str(e)}")
        
        # Try another model for image understanding
        result = process_image_with_llama(image_url, "You are an professor helping students to understand images in a research paper. Please describe what you see in detail, and explain the research concept in an easy-to-understand manner.")
        return result


def process_folder_images(folder_path):
    """
    Process all images in a folder that have contexts ending with <markdown>.
    Updates the image_context.json file with analysis results.

    Args:
        folder_path (str): Path to the folder containing image_context.json and image_urls.json

    Returns:
        dict: Updated context dictionary with analysis results
    """
    try:
        # Read image contexts
        context_file = os.path.join(folder_path, 'image_context.json')
        urls_file = os.path.join(folder_path, 'image_urls.json')

        with open(context_file, 'r') as f:
            contexts = json.load(f)

        with open(urls_file, 'r') as f:
            urls = json.load(f)

        # Process each image that has context ending with <markdown>
        for image_name, context_list in contexts.items():
            if image_name in urls:
                for i, context in enumerate(context_list):
                    if context.strip().endswith('<markdown>'):
                        # Get image analysis
                        image_url = urls[image_name]
                        analysis = analyze_image(image_url)

                        # Update context with analysis
                        contexts[image_name][i] = f"{context}\nImage Analysis: {analysis}"

        # Save updated contexts back to file
        with open(context_file, 'w') as f:
            json.dump(contexts, f, indent=2)

        return contexts

    except Exception as e:
        logger.info(f"Error processing folder images: {str(e)}")
        raise


def process_image_with_llama(image_url, prompt_text, stream=False):
    """
    Process an image with Llama-3.2-90B-Vision-Instruct model.
    
    Args:
        image_url (str): URL of the image to process
        prompt_text (str): Text prompt to send along with the image
        stream (bool): Whether to stream the response incrementally (default: False)
        
    Returns:
        str or generator: The model's response as text, or a generator of response chunks if stream = True
    """
    # Initialize the client
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    
    # Convert image URL to base64
    base64_image = get_image_base64(image_url)
    
    try:
        response = client.chat.completions.create(
            model="Llama-3.2-90B-Vision-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ],
            temperature=0.1,
            top_p=0.1,
            stream=stream
        )
        
        # Handle streaming response
        if stream:
            return response  # Return the streaming response generator
        
        # Return the model's response for non-streaming case
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        else:
            return f"API response without expected structure: {response}"
    except Exception as e:
        return f"Error: {e}"

# Function to convert image URL to base64
def get_image_base64(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        # Get the content type from response headers or infer from URL
        content_type = response.headers.get('Content-Type', '')
        if not content_type or 'image' not in content_type:
            # Try to infer from URL extension
            if image_url.lower().endswith(('jpg', 'jpeg')):
                content_type = 'image/jpeg'
            elif image_url.lower().endswith('png'):
                content_type = 'image/png'
            elif image_url.lower().endswith('webp'):
                content_type = 'image/webp'
            else:
                content_type = 'image/jpeg'  # Default to JPEG
        
        # Encode the image in base64
        base64_image = base64.b64encode(response.content).decode('utf-8')
        # Return in required format
        return f"data:{content_type};base64,{base64_image}"
    else:
        raise Exception(f"Failed to download image: {response.status_code}")


if __name__ == "__main__":
    # folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/markdown/"
    folder_dir = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/c8773c4a9a62ca3bafd2010d3d0093f5/markdown"
    file_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/16005aaa19145334b5605c6bf61661a0.pdf"
    extract_image_context(folder_dir, file_path)
    # upload_images_to_azure(folder_dir)