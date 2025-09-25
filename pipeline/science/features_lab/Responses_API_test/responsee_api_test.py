from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
stream = client.responses.create(
    model="gpt-5",
    input="Say hello and then count to 50.",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")