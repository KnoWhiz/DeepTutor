import os
import sys
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
import requests
import base64
import openai

import logging
# Configure logger to display to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("images_understanding_test.py")

load_dotenv()

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Path to science directory
science_dir = os.path.dirname(current_dir)
# Path to pipeline directory
pipeline_dir = os.path.dirname(science_dir)
# Path to DeepTutor project root
project_root = os.path.dirname(pipeline_dir)
sys.path.append(project_root)

def analyze_image(image_url=None, system_prompt=None, user_prompt=None, context=None, stream=True):
    """
    Analyze an image using Azure OpenAI's vision model.

    Args:
        image_url (str, optional): URL of the image to analyze. Defaults to a test image if none provided.
        system_prompt (str, optional): System prompt for the vision model. Defaults to a test image if none provided.
        user_prompt (str, optional): User prompt for the vision model. Defaults to a test image if none provided.
        context (str, optional): Context for the image. Defaults to None.
        stream (bool, optional): Whether to stream the response incrementally. Defaults to True.

    Returns:
        str or generator: The analysis result as text, or a generator of response chunks if stream=True
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

CRITICAL: Do NOT speculate beyond what is explicitly shown. If information is unclear or not provided, state this plainly rather than making assumptions. Deliver your analysis in a scientific, precise tone.

Start the response with "##Image Analysis:"
"""

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

4. SCIENTIFIC:
   - Use formal, technical language appropriate for a scientific publication
   - Maintain objectivity throughout your description
   - Be concise yet thorough

If certain details are unclear or not visible, simply state "This information is not visible in the image" rather than making educated guesses.

Start the response with "##Image Analysis:"
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
            max_tokens=2000,
            stream=stream
        )
        
        # Handle streaming response
        if stream:
            return response  # Return the streaming response generator
        
        # Handle non-streaming response
        result = response.choices[0].message.content
        return result

    except Exception as e:
        logger.exception(f"Error occurred in analyze_image with Azure OpenAI: {str(e)}")

        # Try another model for image understanding
        result = process_image_with_llama(image_url, system_prompt, stream)
        return result


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
                        logger.info(f"Getting image analysis for saved image: {image_name}")
                        image_url = urls[image_name]
                        analysis = analyze_image(image_url, context=context, stream=True)
                        if isinstance(analysis, str):
                            logger.info("\n\n")
                            logger.info(analysis)
                            logger.info("\n\n")
                            # Update context with analysis
                            contexts[image_name][i] = f"{context}\nImage Analysis: {analysis}"
                        else:
                            logger.info("\n\n")
                            analysis_text = ""
                            for chunk in analysis:
                                if hasattr(chunk, "choices") and chunk.choices:
                                    # Extract the actual text content from the ChatCompletionChunk
                                    delta = chunk.choices[0].delta
                                    if hasattr(delta, "content") and delta.content:
                                        chunk_text = delta.content
                                        analysis_text += chunk_text
                                        # Use sys.stdout directly for real-time streaming output
                                        sys.stdout.write(chunk_text)
                                        sys.stdout.flush()
                                else:
                                    # Skip empty or malformed chunks
                                    continue
                            logger.info("\n\n")
                            # Update context with analysis text
                            contexts[image_name][i] = f"{context}\nImage Analysis: {analysis_text}"

        # Save updated contexts back to file
        with open(context_file, 'w') as f:
            json.dump(contexts, f, indent=2)

        return contexts

    except Exception as e:
        logger.info(f"Error processing folder images: {str(e)}")
        raise

if __name__ == '__main__':
    # Example usage
    try:
        # Test single image analysis
        logger.info("Testing single image analysis with streaming...")
        result = analyze_image("https://knowhiztutorrag.blob.core.windows.net/knowhiztutorrag/file_appendix/3671da1e844b53ffbdccac7bc8c57341/images/_page_1_Figure_1.jpeg")
        logger.info("\nSingle Image Analysis Result:")
        if isinstance(result, str):
            logger.info(result)
        else:
            # Handle streaming response
            for chunk in result:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        # Use sys.stdout for streaming output
                        sys.stdout.write(delta.content)
                        sys.stdout.flush()
            logger.info("\n")

        # Test non-streaming mode
        logger.info("\nTesting single image analysis without streaming...")
        non_stream_result = analyze_image(
            "https://knowhiztutorrag.blob.core.windows.net/knowhiztutorrag/file_appendix/3671da1e844b53ffbdccac7bc8c57341/images/_page_1_Figure_1.jpeg",
            stream=False
        )
        logger.info("Non-streaming result:")
        logger.info(non_stream_result)

        # Test folder processing
        # Define the path relative to the project root
        embedded_content_path = os.path.join(os.path.dirname(project_root), 
                                           "embedded_content/3671da1e844b53ffbdccac7bc8c57341/markdown")
        logger.info(f"\nProcessing folder images at path: {embedded_content_path}")
        
        # Check if directory exists
        if not os.path.exists(embedded_content_path):
            logger.info(f"Directory does not exist: {embedded_content_path}")
            # Create directory structure for testing
            os.makedirs(embedded_content_path, exist_ok=True)
            # Create empty image context and URL files
            with open(os.path.join(embedded_content_path, 'image_context.json'), 'w') as f:
                json.dump({}, f)
            with open(os.path.join(embedded_content_path, 'image_urls.json'), 'w') as f:
                json.dump({}, f)
            logger.info(f"Created empty directory structure for future testing")
        else:
            updated_contexts = process_folder_images(embedded_content_path)
            logger.info("Folder processing completed. Updated contexts saved.")

    except Exception as e:
        logger.info(f"Failed to process: {str(e)}")
