# from langchain_deepseek import ChatDeepSeek

# import getpass
# import os

# if not os.getenv("DEEPSEEK_API_KEY"):
#     os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")

# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     # max_retries=2,
# )

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg.content

# from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "{input}"),
#     ]
# )

# chain = prompt | llm
# chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "German",
#         "input": "I love programming.",
#     }
# )

# print(chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "German",
#         "input": "I love programming.",
#     }
# ))

import requests
import json

# Replace with your SambaNova Cloud API key
API_KEY = "b114b9b6-efa1-4815-bca9-27f35ea2910e"

# Replace with the actual endpoint URL for DeepSeek R1 streaming.
# For example, it might look like: "https://cloud.sambanova.ai/v1/models/deepseek/r1/infer"
url = "https://api.sambanova.ai/v1/chat/completions"

# Set up the headers, including the authorization header
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Define the payload. In this example we set "stream": true.
# You can adjust the parameters (like prompt, temperature, etc.) per your model’s specification.
payload = {
    "prompt": "Once upon a time in a land far, far away,",
    "max_tokens": 1000000,
    "temperature": 0.0,
    "stream": True
}

def stream_inference():
    # Send the POST request with stream=True
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    # Raise an exception if the request was unsuccessful
    response.raise_for_status()
    
    # Iterate over the streaming response line by line.
    # Each line should contain a JSON object with a part of the model's output.
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                # Sometimes the API may prefix lines with "data:" so remove that if present
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                # Parse the JSON payload
                token_data = json.loads(line)
                # Here we assume the token text is under a key, for example "token"
                token = token_data.get("token", "")
                print(token, end="", flush=True)
            except json.JSONDecodeError:
                # If a line isn't valid JSON, skip it or log the error.
                continue

if __name__ == "__main__":
    try:
        print("Streaming output:")
        stream_inference()
        print("\nDone.")
    except Exception as e:
        print("Error during streaming inference:", str(e))