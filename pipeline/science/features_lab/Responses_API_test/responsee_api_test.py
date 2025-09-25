from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

stream = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    tools=[{"type": "web_search"}],
    # tool_choice="auto",  # default
    instructions="Search the web as needed (multiple searches OK) and cite sources.",
    input="What are the latest results on trapped-ion photonics integration? Summarize with citations.",
    stream=True,
)

for event in stream:
    if event.type == "response.tool_call.created":
        print("🔎 tool call:", event.tool.name)
    elif event.type == "response.tool_call.delta":
        print(event.delta, end="")
    elif event.type == "response.output_text.delta":
        print(event.delta, end="")