import os
import openai
from dotenv import load_dotenv
from typing import Optional

import logging
logger = logging.getLogger("deepseek_test.py")

load_dotenv()

# def deepseek_inference(
#     prompt: str,
#     system_message: str = "You are a deep thinking assistant",
#     stream: bool = False,
#     temperature: float = 0.0,
#     top_p: float = 0.1,
#     max_tokens: int = 5000
# ) -> Optional[str]:
#     """
#     Get completion from the DeepSeek model with optional streaming support.

#     Args:
#         prompt: The user's input prompt
#         system_message: The system message to set the AI's behavior
#         stream: Whether to stream the output or not
#         temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
#         top_p: Controls diversity via nucleus sampling
#         max_tokens: Maximum number of tokens to generate

#     Returns:
#         The generated text if streaming is False, None if streaming is True
#     """
#     client = openai.OpenAI(
#         api_key=os.environ.get("SAMBANOVA_API_KEY"),
#         base_url="https://preview.snova.ai/v1",
#     )

#     try:
#         response = client.chat.completions.create(
#             model="DeepSeek-R1",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#             stream=stream
#         )

#         if stream:
#             # Process the streaming response
#             for chunk in response:
#                 if chunk.choices[0].delta.content is not None:
#                     print(chunk.choices[0].delta.content, end="", flush=True)
#             print()  # Add a newline at the end
#             return None
#         else:
#             # Return the complete response
#             return response.choices[0].message.content

#     except openai.APIError as e:
#         logger.info(f"API Error: {str(e)}")
#         return None
#     except Exception as e:
#         logger.info(f"An error occurred: {str(e)}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     # Example with streaming
#     print("Streaming response:")
#     deepseek_inference("what is 1+1?", stream = True)

#     print("\nNon-streaming response:")
#     response = deepseek_inference("what is 1+1?", stream = False)
#     if response:
#         print(response)

from sambanova import SambaNova

# client = SambaNova(
#     api_key=os.environ.get("SAMBANOVA_API_KEY"),
#     base_url="https://api.sambanova.ai/v1",
# )

# response = client.chat.completions.create(
#     model="DeepSeek-R1-0528",
#     messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"}],
#     temperature=0.0,
#     top_p=0.1
# )

# print(response.choices[0].message.content)

client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url=os.environ.get("SAMBANOVA_API_ENDPOINT"),
)

response = client.chat.completions.create(
    model="DeepSeek-R1",
    messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"}],
    temperature=0.0,
    top_p=0.1,
    # max_tokens=10000,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
