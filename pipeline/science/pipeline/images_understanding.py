import os
import json
import sys
import openai
import requests
import base64
from typing import Dict, List, Set, Union
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Add the project root to Python path for direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Use try/except pattern for imports to handle both direct execution and module import
try:
    # Try importing as a module first
    from pipeline.science.pipeline.utils import generate_file_id, create_truncated_db
    from pipeline.science.pipeline.config import load_config
    from pipeline.science.pipeline.embeddings import get_embedding_models
except ImportError:
    # If that fails, try relative imports
    try:
        from .utils import generate_file_id, create_truncated_db
        from .config import load_config
        from .embeddings import get_embedding_models
    except ImportError:
        # Last resort, try direct import from current directory or parent
        sys.path.append(os.path.dirname(current_dir))
        from utils import generate_file_id, create_truncated_db
        from config import load_config
        from embeddings import get_embedding_models

load_dotenv()

import logging
logger = logging.getLogger("tutorpipeline.science.images_understanding")

try:
    # Try importing as a module first
    from pipeline.science.pipeline.helper.azure_blob import AzureBlobHelper
except ImportError:
    # If that fails, try importing directly
    try:
        from pipeline.helper.azure_blob import AzureBlobHelper
    except ImportError:
        # Last resort, try relative import
        try:
            from .helper.azure_blob import AzureBlobHelper
        except ImportError:
            # Direct import from a nearby location
            sys.path.append(os.path.join(current_dir, "helper"))
            from azure_blob import AzureBlobHelper


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
    # yield f"Current markdown folder: {folder_dir}"

    # Initialize both JSON files
    image_context_path, _ = initialize_image_files(folder_dir)

    # Define the image file extensions we care about
    config = load_config()
    image_extensions: Set[str] = set(config["image_extensions"])

    # List all image files in the folder (case-insensitive match)
    image_files = [f for f in os.listdir(folder_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    if not image_files:
        logger.info("No image files found in the folder.")
        yield "\n\n**❌ No image files found in the folder.**"
        return

    # Find the markdown file (with file_id.md file name)
    file_id = generate_file_id(file_path)
    md_files = [os.path.join(folder_dir, f"{file_id}.md")]
    if not md_files:
        logger.info("No markdown file found in the folder.")
        yield "\n\n**❌ No markdown file found in the folder.**"
        return

    md_file = md_files[0]
    md_path = md_file

    # Read the content of the markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        md_lines = f.read().splitlines()

    # If there are images_files and md_files, re-order the list image_files to match the order that images show up in md_files
    # yield "\n\n**Re-ordering image files to match the order that images show up in md_files...**"
    logger.info("Re-ordering image files to match the order that images show up in md_files...")
    image_order = []
    for line in md_lines:
        for image in image_files:
            if image in line and image not in image_order:
                image_order.append(image)
    # Add any images that weren't found in the markdown file to the end of the order list
    # yield "\n\n**Adding any images that weren't found in the markdown file to the end of the order list...**"
    logger.info("Adding any images that weren't found in the markdown file to the end of the order list...")
    for image in image_files:
        if image not in image_order:
            image_order.append(image)
    # Replace image_files with the ordered list
    image_files = image_order

    # Create a dictionary to store image filename vs. list of context windows
    # yield "\n\n**Creating a dictionary to store image filename vs. list of context windows...**"
    logger.info("Creating a dictionary to store image filename vs. list of context windows...")
    image_context: Dict[str, List[str]] = {}

    for i, image in enumerate(image_files):
        # yield f"\n\n**Processing image {i+1} of {len(image_files)}: {image}**"
        # yield f"\n\n**Processing image {i+1}/{len(image_files)}**"
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
    # yield "\n\n**Writing the dictionary to a JSON file in the same folder...**"
    logger.info(f"Writing the dictionary to a JSON file in the same folder: {image_context_path}")
    with open(image_context_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_context, outfile, indent=2, ensure_ascii=False)

    # Process folder images
    # yield "\n\n**Processing folder images...**"
    logger.info("Processing folder images...")
    for chunk in process_folder_images(folder_dir):
        yield chunk

    logger.info(f"Image context data saved to: {image_context_path}")


def analyze_image(image_url=None, system_prompt=None, user_prompt=None, context=None):
    """
    Analyze an image using Azure OpenAI's vision model.

    Args:
        image_url (str, optional): URL of the image to analyze. Defaults to a test image if none provided.
        system_prompt (str, optional): System prompt for the vision model. Defaults to a test image if none provided.
        user_prompt (str, optional): User prompt for the vision model. Defaults to a test image if none provided.
        context (str, optional): Context for the image. Defaults to None.

    Returns:
        str: The analysis result from the vision model
    """
    if system_prompt is None:
        system_prompt = ""
    if user_prompt is None:
        user_prompt = ""
    
    context_info = ""
    if context is not None and context:
        context_info = f"REFERENCE CONTEXT FROM DOCUMENT: {context}\n\n"
    
    user_prompt = user_prompt + f"""You are an expert scientist analyzing scientific figures. Provide a factual, objective analysis of what is ACTUALLY VISIBLE in the image. Adapt your analysis to the type of figure shown (data plot, experimental illustration, conceptual diagram, etc.).

{context_info}Based on the image and any provided context, analyze ONLY what is clearly observable:

1. FIGURE TYPE AND PURPOSE:
   - Identify if this is a data visualization, experimental setup, conceptual diagram, microscopy image, etc.
   - Note the apparent scientific discipline or subject area

2. VISIBLE ELEMENTS:
   - For data plots: chart type, axes, scales, legends, data series, error indicators
   - For diagrams: labeled components, pathways, relationships, structures
   - For experimental illustrations: equipment, materials, procedures, conditions
   - For microscopy/imagery: visible structures, scale markers, coloration, highlighting

3. QUANTITATIVE INFORMATION:
   - Any explicitly visible measurements, values, statistics
   - Trends, patterns, or relationships evident in the data
   - Statistical indicators (p-values, error bars, confidence intervals)

CRITICAL: Do NOT speculate beyond what is explicitly shown. If information is unclear or not provided, state this plainly rather than making assumptions. Deliver your analysis in a scientific, precise tone."""

    system_prompt = system_prompt + f"""You are analyzing a scientific figure. Your task is to provide a comprehensive, accurate, and factual description of EXACTLY what appears in the image. 

{context_info}Guidelines for your analysis:

1. ADAPTABILITY: 
   - Adjust your analysis based on the figure type (graph, diagram, microscopy image, etc.)
   - Focus on the most relevant aspects for each type of visualization

2. PRECISION AND FACTUALITY:
   - Describe only what is visibly present - no speculation or assumptions
   - Use precise scientific terminology appropriate to the apparent field
   - When exact values are visible, report them accurately with units
   - When relationships or trends are shown, describe them objectively

3. COMPLETENESS:
   - Identify all labeled elements, axes, legends, and annotations
   - Describe the visualization structure and organization
   - Note any visible statistical information or metrics
   - Mention any apparent limitations or qualifications shown

4. SCIENTIFIC TONE:
   - Use formal, technical language appropriate for a scientific publication
   - Maintain objectivity throughout your description
   - Be concise yet thorough

If certain details are unclear or not visible, simply state "This information is not visible in the image" rather than making educated guesses."""

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
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
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
        logger.info(f"The full messages are: {messages}")

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
        result = process_image_with_llama(image_url, system_prompt)
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
        img_count = len(contexts.items())
        for image_name, context_list in contexts.items():
            if image_name in urls:
                for i, context in enumerate(context_list):
                    if context.strip().endswith('<markdown>'):
                        # Get image analysis
                        yield "\n\n**📊 Getting image analysis for saved image ...**"
                        image_url = urls[image_name]
                        yield f"\n\n![{image_name}]({image_url})"
                        analysis = analyze_image(image_url, context = context)
                        if type(analysis) == str:
                            yield "\n\n"
                            # yield f"Context: {context}"
                            yield analysis
                            yield "\n\n"
                        # yield f"\n\n**Image analysis for {image_name} completed.**"
                        # Update context with analysis
                        contexts[image_name][i] = f"{context}\nImage Analysis: {analysis}"

        # Save updated contexts back to file
        with open(context_file, 'w') as f:
            json.dump(contexts, f, indent=2)

        # return contexts
        return

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


def aggregate_image_contexts_to_urls(folder_list: List[Union[str, Path]]) -> Dict[str, str]:
    """
    Create a mapping from image context strings to their corresponding URLs in Azure Blob.

    Parameters:
        folder_list (List[Union[str, Path]]): List of folder paths containing image_urls.json and image_context.json files

    Returns:
        Dict[str, str]: Dictionary mapping from context string to image URL

    Note:
        If multiple context strings are identical across different images,
        the last processed image URL will be associated with that context.
    """
    context_to_url_mapping: Dict[str, str] = {}

    for folder in folder_list:
        folder_path = str(folder) if isinstance(folder, Path) else folder

        # Get paths to the JSON files using os.path.join
        image_context_path = os.path.join(folder_path, "image_context.json")
        image_urls_path = os.path.join(folder_path, "image_urls.json")

        # Check if both files exist
        if not os.path.exists(image_context_path) or not os.path.exists(image_urls_path):
            logger.warning(f"Missing required JSON files in folder: {folder_path}")
            continue

        # Load the JSON files
        try:
            with open(image_context_path, "r", encoding="utf-8") as f:
                image_contexts = json.load(f)

            with open(image_urls_path, "r", encoding="utf-8") as f:
                image_urls = json.load(f)

            # Create mapping from context to URL
            for image_name, contexts in image_contexts.items():
                if image_name in image_urls:
                    image_url = image_urls[image_name]

                    # Map each context to the image URL
                    for context in contexts:
                        context_to_url_mapping[context] = image_url

        except Exception as e:
            logger.warning(f"Error processing files in folder {folder_path}: {str(e)}")
            continue

    return context_to_url_mapping


def create_image_context_embeddings_text(folder_list: List[Union[str, Path]]) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    """
    Process image contexts from multiple folders and format them as embedding chunks for database storage.

    Parameters:
        folder_list (List[Union[str, Path]]): List of folder paths containing image_urls.json and image_context.json files

    Returns:
        List[Dict[str, Union[str, Dict[str, str]]]]: List of embedding chunks with image contexts and metadata
    """
    embedding_chunks: List[Dict[str, Union[str, Dict[str, str]]]] = []

    for folder in folder_list:
        folder_path = str(folder) if isinstance(folder, Path) else folder

        # Extract file_id from the folder path
        # Assuming folder structure like: "/path/to/embedded_content/{file_id}/markdown"
        try:
            file_id = os.path.basename(os.path.dirname(folder_path))
        except Exception:
            file_id = "unknown"
            logger.warning(f"Could not extract file_id from path: {folder_path}, using 'unknown'")

        # Get paths to the JSON files
        image_context_path = os.path.join(folder_path, "image_context.json")
        image_urls_path = os.path.join(folder_path, "image_urls.json")

        # Check if both files exist
        if not os.path.exists(image_context_path) or not os.path.exists(image_urls_path):
            logger.warning(f"Missing required JSON files in folder: {folder_path}")
            continue

        # Load the JSON files
        try:
            with open(image_context_path, "r", encoding="utf-8") as f:
                image_contexts = json.load(f)

            with open(image_urls_path, "r", encoding="utf-8") as f:
                image_urls = json.load(f)

            # Process each image and its contexts
            for image_name, contexts in image_contexts.items():
                if image_name in image_urls:
                    image_url = image_urls[image_name]

                    # Create an embedding chunk for each context
                    for i, context in enumerate(contexts):
                        # Skip empty or None contexts
                        if not context or not isinstance(context, str):
                            logger.warning(f"Skipping invalid context for {image_name} at index {i}")
                            continue

                        # Create chunk with context and metadata
                        chunk = {
                            "text": context,
                            "metadata": {
                                "source_type": "image",
                                "file_id": file_id,
                                "image_name": image_name,
                                "image_url": image_url,
                                "chunk_id": f"{file_id}_{image_name}_{i}",
                                "context_index": i
                            }
                        }
                        embedding_chunks.append(chunk)

        except Exception as e:
            logger.warning(f"Error processing files in folder {folder_path}: {str(e)}")
            continue

    logger.info(f"Created {len(embedding_chunks)} embedding chunks from image contexts")
    return embedding_chunks


def create_image_context_embeddings_db(folder_list: List[Union[str, Path]], embedding_type: str = "default") -> FAISS:
    """
    Process image contexts from multiple folders and create a FAISS database with embeddings.
    This function is compatible with load_embeddings() return type.

    Parameters:
        folder_list (List[Union[str, Path]]): List of folder paths containing image_urls.json and image_context.json files
        embedding_type (str): Type of embedding model to use ("default", "lite", or "small")

    Returns:
        FAISS: A FAISS vector database containing the image context embeddings
    """
    # Initialize required components
    config = load_config()
    para = config["llm"]
    embeddings = get_embedding_models(embedding_type, para)

    # Convert image context data to Document objects for FAISS
    documents = []

    for folder in folder_list:
        folder_path = str(folder) if isinstance(folder, Path) else folder

        # Extract file_id from the folder path
        try:
            file_id = os.path.basename(os.path.dirname(folder_path))
        except Exception:
            file_id = "unknown"
            logger.warning(f"Could not extract file_id from path: {folder_path}, using 'unknown'")

        # Get paths to the JSON files
        image_context_path = os.path.join(folder_path, "image_context.json")
        image_urls_path = os.path.join(folder_path, "image_urls.json")

        # Check if both files exist
        if not os.path.exists(image_context_path) or not os.path.exists(image_urls_path):
            logger.warning(f"Missing required JSON files in folder: {folder_path}")
            continue

        # Load the JSON files
        try:
            with open(image_context_path, "r", encoding="utf-8") as f:
                image_contexts = json.load(f)

            with open(image_urls_path, "r", encoding="utf-8") as f:
                image_urls = json.load(f)

            # Process each image and its contexts
            for image_name, contexts in image_contexts.items():
                if image_name in image_urls:
                    image_url = image_urls[image_name]

                    # Create a Document object for each context
                    for i, context in enumerate(contexts):
                        # Skip empty or None contexts
                        if not context or not isinstance(context, str):
                            logger.warning(f"Skipping invalid context for {image_name} at index {i}")
                            continue

                        # Create Document object with metadata
                        document = Document(
                            page_content=context,
                            metadata={
                                "source_type": "image",
                                "file_id": file_id,
                                "image_name": image_name,
                                "image_url": image_url,
                                "chunk_id": f"{file_id}_{image_name}_{i}",
                                "context_index": i,
                                "source": "image"  # For compatibility with other document sources
                            }
                        )
                        documents.append(document)

        except Exception as e:
            logger.warning(f"Error processing files in folder {folder_path}: {str(e)}")
            continue

    # Create FAISS database from documents
    if not documents:
        logger.warning("No valid image contexts found. Returning empty FAISS database.")
        # Create an empty FAISS database (may not work as expected)
        db = FAISS.from_documents(
            [Document(page_content="Empty placeholder", metadata={"source": "placeholder"})],
            embeddings
        )
        truncated_db = create_truncated_db(db)
        return db, truncated_db

    try:
        # Create the FAISS database with the documents
        db = FAISS.from_documents(documents, embeddings)
        logger.info(f"Created FAISS database with {len(documents)} image context embeddings")
        truncated_db = create_truncated_db(db)
        return db, truncated_db
    except Exception as e:
        logger.error(f"Error creating FAISS database: {str(e)}")
        # Fallback to smaller models if available
        try:
            logger.info("Trying with 'small' embedding model...")
            embeddings_small = get_embedding_models("small", para)
            db = FAISS.from_documents(documents, embeddings_small)
            truncated_db = create_truncated_db(db)
            return db, truncated_db
        except Exception as e2:
            logger.error(f"Error with small embedding model: {str(e2)}")
            logger.info("Trying with 'lite' embedding model...")
            embeddings_lite = get_embedding_models("lite", para)
            db = FAISS.from_documents(documents, embeddings_lite)
            truncated_db = create_truncated_db(db)
            return db, truncated_db


if __name__ == "__main__":
    # Simple check to make sure all needed imports are working
    print("Imports successful. Running the main function...")

    # folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/markdown/"
    folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/c5e6dadde391f97ac2ba65acf827248e/markdown"
    # file_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/16005aaa19145334b5605c6bf61661a0.pdf"
    # extract_image_context(folder_dir, file_path)
    # # upload_images_to_azure(folder_dir)

    # Test the aggregate_image_contexts_to_urls function
    test_folders = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/c5e6dadde391f97ac2ba65acf827248e/markdown",
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/7ebfb3495a81793a0daa2246d0ed24db/markdown"
        # Add additional test folders if available
        # "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/markdown/"
    ]

    print("\nTesting aggregate_image_contexts_to_urls function:")
    try:
        context_url_mapping = aggregate_image_contexts_to_urls(test_folders)
        print(f"Found {len(context_url_mapping)} context-to-URL mappings")

        # Print a sample of the mapping (up to 3 entries)
        for i, (context, url) in enumerate(list(context_url_mapping.items())):
            # Truncate long contexts for display
            display_context = context[:100] + "..." if len(context) > 100 else context
            print(f"Sample {i+1}:")
            print(f"Context: {display_context}")
            print(f"URL: {url}")
            print("")
    except Exception as e:
        print(f"Error testing aggregate_image_contexts_to_urls: {str(e)}")

    # Test the create_image_context_embeddings_text function
    print("\nTesting create_image_context_embeddings_text function (raw data):")
    try:
        embedding_chunks = create_image_context_embeddings_text(test_folders)
        print(f"Created {len(embedding_chunks)} embedding chunks")

        # Print a sample of the embedding chunks (up to all entries)
        for i, chunk in enumerate(embedding_chunks):
            # Truncate long texts for display
            display_text = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            print(f"Sample Chunk {i+1}:")
            print(f"Text: {display_text}")
            print(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}")
            print("")
    except Exception as e:
        print(f"Error testing create_image_context_embeddings_text: {str(e)}")

    # Test the create_image_context_embeddings_db function
    print("\nTesting create_image_context_embeddings_db function (FAISS database):")
    try:
        db, truncated_db = create_image_context_embeddings_db(test_folders)
        print(f"Created FAISS database with image context embeddings")

        # Test a simple similarity search
        if db:
            test_query = "Image 3"
            results = db.similarity_search(test_query, k=1)
            print(f"\nSample similarity search results for query: '{test_query}'")
            for i, doc in enumerate(results):
                print(f"Result {i+1}:")
                print(f"Content: {doc.page_content[:100]}..." if len(doc.page_content) > 100 else doc.page_content)
                print(f"Metadata: {doc.metadata}")
                print("")
    except Exception as e:
        print(f"Error testing create_image_context_embeddings_db: {str(e)}")

    # Test the create_truncated_db function
    print("\nTesting create_truncated_db function:")
    try:
        # First create a regular database
        db, truncated_db = create_image_context_embeddings_db(test_folders)
        if db:
            # Create a truncated version
            truncated_db = create_truncated_db(db)
            print(f"Created truncated FAISS database with first 3 sentences of each chunk")
            
            # Test a simple similarity search on the truncated database
            test_query = "Image 3"
            results = truncated_db.similarity_search(test_query, k=1)
            print(f"\nSample similarity search results for query: '{test_query}' on truncated database")
            for i, doc in enumerate(results):
                print(f"Result {i+1}:")
                print(f"Content: {doc.page_content[:100]}..." if len(doc.page_content) > 100 else doc.page_content)
                print(f"Metadata: {doc.metadata}")
                print("")
    except Exception as e:
        print(f"Error testing create_truncated_db: {str(e)}")