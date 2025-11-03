# from openai import OpenAI

# # Initialize the client to use Lemonade Server
# client = OpenAI(
#     base_url="http://localhost:8000/api/v1",
#     api_key="lemonade"  # required but unused
# )

# # Create a chat completion
# completion = client.chat.completions.create(
#     model="DeepSeek-Qwen3-8B-GGUF",  # or any other available model
#     messages=[
#         {"role": "user", "content": "What is the capital of France?"}
#     ]
# )

# # Print the response
# print(completion.choices[0].message.content)

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="lemonade",
)

stream = client.chat.completions.create(
    model="DeepSeek-Qwen3-8B-GGUF",
    messages=[{"role": "user", "content": "What is the capital of France? Explain it in detail."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)

print()  # final newline

from openai import OpenAI

# Lemonade runs at http://localhost:8000/api/v1 by default
client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="lemonade",  # required by the client but unused by Lemonade
)

texts = [
    "search_query: multiplexed ion–photon interfaces",
    "document: temporally multiplexed ion transport in a linear Paul trap",
]

resp = client.embeddings.create(
    model="nomic-embed-text-v1-GGUF",  # Lemonade model name you pulled
    input=texts,
)

vectors = [item.embedding for item in resp.data]
print(len(vectors), "embeddings,", "dim =", len(vectors[0]))