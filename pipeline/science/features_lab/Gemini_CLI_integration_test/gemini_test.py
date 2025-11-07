from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# print(f"Using API key: {api_key}")
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-pro", contents="Explain how AI works in a few words"
)
print(response.text)