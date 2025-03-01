
import os  
import base64
import dotenv
dotenv.load_dotenv()
from openai import AzureOpenAI  

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP")
deployment = "o3-mini"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY_BACKUP")  

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-12-01-preview",
)
    
    
# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

#Prepare the chat prompt 
chat_prompt = [
    {
        "role": "developer",
        "content": [
            {
                "type": "text",
                "text": "You are a deep thinking AI assistant that helps people find information."
            }
        ]
    }
] 
    
# Include speech result if speech is enabled  
messages = chat_prompt  
    
# Generate the completion  
completion = client.chat.completions.create(  
    model=deployment,
    messages=messages,
    max_completion_tokens=100000,
    stop=None,  
    stream=False
)

print(completion.to_json())  
    