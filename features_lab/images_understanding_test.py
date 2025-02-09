import os
import sys
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

if __name__ == '__main__':
    # Example usage
    try:
        result = analyze_image("https://knowhiztutorrag.blob.core.windows.net/knowhiztutorrag/file_appendix/3671da1e844b53ffbdccac7bc8c57341/images/_page_1_Figure_1.jpeg")
        print("\nImage Analysis Result:")
        print(result)
    except Exception as e:
        print(f"Failed to analyze image: {str(e)}")
