import os
import openai
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

def process_image_with_llama(image_url, prompt_text, stream=False):
    """
    Process an image with Llama-3.2-90B-Vision-Instruct model.
    
    Args:
        image_url (str): URL of the image to process
        prompt_text (str): Text prompt to send along with the image
        stream (bool): Whether to stream the response incrementally (default: False)
        
    Returns:
        str or generator: The model's response as text, or a generator of response chunks if stream=True
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

# Example usage
if __name__ == "__main__":
    # Example image URL and prompt
    image_url = "https://knowhiztutorrag.blob.core.windows.net/knowhiztutorrag/file_appendix/4a003e263d89a1e4fabff70111df076b/images/_page_2_Figure_0.jpeg"
    prompt = "What do you see in this image"
    
    # Get the regular response
    result = process_image_with_llama(image_url, prompt)
    print(result)
    
    # Example with streaming
    print("\nStreaming response:")
    stream_response = process_image_with_llama(image_url, prompt, stream=True)
    
    # Process the streaming response
    try:
        for chunk in stream_response:
            if hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
        print()  # Add a newline at the end
    except Exception as e:
        print(f"Error processing stream: {e}")