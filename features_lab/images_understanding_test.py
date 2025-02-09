import os
import sys
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
load_dotenv()

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

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
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

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
        print(f"Error processing folder images: {str(e)}")
        raise

if __name__ == '__main__':
    # Example usage
    try:
        # Test single image analysis
        # result = analyze_image("https://knowhiztutorrag.blob.core.windows.net/knowhiztutorrag/file_appendix/3671da1e844b53ffbdccac7bc8c57341/images/_page_1_Figure_1.jpeg")
        # print("\nSingle Image Analysis Result:")
        # print(result)
        
        # Test folder processing
        folder_path = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/3671da1e844b53ffbdccac7bc8c57341/markdown"
        print("\nProcessing folder images...")
        updated_contexts = process_folder_images(folder_path)
        print("Folder processing completed. Updated contexts saved.")
        
    except Exception as e:
        print(f"Failed to process: {str(e)}")
